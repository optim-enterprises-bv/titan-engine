#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::sync::Arc;

use crate::attention::{AttentionMask, SdpaParams};
use crate::device_map::{DeviceMappedMask, DeviceMapper};
use crate::gguf::Content;
use crate::layers::{CausalMaskConfig, CausalMasker, QRmsNorm, RotaryEmbedding, Sdpa};
use crate::layers_masker::PastKvLenCache;
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::{extract_logits, EitherCache, KvCache, NormalCache};
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};
use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Embedding, Module};
#[cfg(feature = "cuda")]
use mistralrs_quant::TieredExperts;
use mistralrs_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

// Default fallback for models that don't specify context_length
const DEFAULT_MAX_SEQ_LEN: u32 = 4096;

struct Mlp {
    feed_forward_w1: Arc<dyn QuantMethod>,
    feed_forward_w2: Arc<dyn QuantMethod>,
    feed_forward_w3: Arc<dyn QuantMethod>,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        let y = crate::ops::mul_and_act(&w1, &w3, crate::layers::Activation::Silu)?;
        self.feed_forward_w2.forward(&y)
    }
}

/// The routed experts of one MoE layer: candle's stacked `QMatMul`s (stock), or
/// titan-engine's slotted experts read by cuda-oxide kernels (`TITAN_TIERED=1`),
/// which produce bit-identical outputs.
pub(crate) enum Experts {
    Stock {
        gate: QMatMul,
        up: QMatMul,
        down: QMatMul,
    },
    #[cfg(feature = "cuda")]
    Tiered {
        gate: TieredExperts,
        up: TieredExperts,
        down: TieredExperts,
    },
}

impl Experts {
    /// SiLU-gated routed experts: `xs` is `(num_tokens, 1, hidden)`, `indices` is
    /// `(num_tokens, top_k)`; returns `(num_tokens, top_k, hidden)` (unweighted).
    pub(crate) fn forward(&self, xs: &Tensor, indices: &Tensor) -> Result<Tensor> {
        self.forward_lookahead(xs, indices, &[])
    }

    /// `forward`, first handing the tiered prefetcher each `(experts, depth, predicted ids)` of a later
    /// layer; the predictions share the routing ids' device sync and never touch this layer's output.
    pub(crate) fn forward_lookahead(
        &self,
        xs: &Tensor,
        indices: &Tensor,
        preds: &[(&Experts, usize, Tensor)],
    ) -> Result<Tensor> {
        self.forward_act_lookahead(
            xs,
            indices,
            preds,
            |gate, up| crate::ops::mul_and_act(gate, up, crate::layers::Activation::Silu),
            true,
        )
    }

    /// `forward` with `act(gate, up)` in place of SiLU gating, applied to the assembled
    /// `(num_tokens, top_k, n)` projections (GPU slots and CPU twin rows alike).
    pub(crate) fn forward_act(
        &self,
        xs: &Tensor,
        indices: &Tensor,
        act: impl Fn(&Tensor, &Tensor) -> Result<Tensor>,
    ) -> Result<Tensor> {
        self.forward_act_lookahead(xs, indices, &[], act, false)
    }

    /// `forward` with the routed outputs reduced piecewise: `reduce(first_token, y)` maps the `(n, top_k, hidden)`
    /// outputs of tokens `first_token..first_token + n` to `(n, ..)`, and the results are concatenated along dim 0.
    /// A streamed prefill chunk (`titan_pfs`, MMQ) of more than `piece` tokens runs its down projection in
    /// `piece`-token pieces against one staged copy of the experts, reducing each piece as it lands, so the
    /// chunk's `(tokens, top_k, hidden)` outputs never exist at once; anything else is `reduce(0, forward(..))`.
    pub(crate) fn forward_reduced(
        &self,
        xs: &Tensor,
        indices: &Tensor,
        piece: usize,
        reduce: impl Fn(usize, &Tensor) -> Result<Tensor>,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let Experts::Tiered { gate, up, down } = self {
            let rows = xs.dim(0)?;
            if piece > 0 && rows > piece && [gate, up, down].iter().all(|p| p.streams(rows) && p.streams_mmq()) {
                if let Some(y) = Self::forward_streamed_reduced(gate, up, down, xs, indices, piece, &reduce)? {
                    return Ok(y);
                }
            }
        }
        reduce(0, &self.forward(xs, indices)?)
    }

    /// `forward_lookahead` and `forward_act` combined; `silu`: `act` is SiLU gating (the CPU one-pass path computes it).
    pub(crate) fn forward_act_lookahead(
        &self,
        xs: &Tensor,
        indices: &Tensor,
        preds: &[(&Experts, usize, Tensor)],
        act: impl Fn(&Tensor, &Tensor) -> Result<Tensor>,
        silu: bool,
    ) -> Result<Tensor> {
        match self {
            Experts::Stock { gate, up, down } => {
                let gate = gate.indexed_moe_forward(xs, indices)?;
                let up = up.indexed_moe_forward(xs, indices)?;
                let activated = act(&gate, &up)?;
                down.indexed_moe_forward(&activated, indices)
            }
            #[cfg(feature = "cuda")]
            Experts::Tiered { gate, up, down } => {
                use mistralrs_quant::TieredExperts;
                let rows = xs.dim(0)?;
                if [gate, up, down].iter().all(|p| p.streams(rows)) {
                    if let Some(y) = Self::forward_streamed(gate, up, down, xs, indices, &act)? {
                        return Ok(y);
                    }
                }
                if silu {
                    let fast: Vec<([&TieredExperts; 3], usize, Tensor)> = preds
                        .iter()
                        .filter_map(|(e, d, t)| match e {
                            Experts::Tiered { gate, up, down } => Some(([gate, up, down], *d, t.clone())),
                            Experts::Stock { .. } => None,
                        })
                        .collect();
                    if fast.len() == preds.len() {
                        if let Some(y) = TieredExperts::forward_fast(gate, up, down, xs, indices, &fast, &act)? {
                            return Ok(y);
                        }
                    }
                }
                // Quantize once for gate and up, then one device sync fetches the routing ids; the
                // shared input goes to the host only if some routed expert is not GPU-resident.
                let mut qx = gate.quantize_input(xs)?;
                let ids_host = if preds.is_empty() {
                    TieredExperts::routing_to_host(indices)?
                } else {
                    let mut all = vec![indices.flatten_all()?];
                    for (_, _, p) in preds {
                        all.push(p.flatten_all()?);
                    }
                    let mut v = TieredExperts::routing_to_host(&Tensor::cat(&all, 0)?)?;
                    let mut rest = v.split_off(indices.elem_count());
                    for (next, depth, p) in preds {
                        let tail = rest.split_off(p.elem_count());
                        if let Experts::Tiered { gate, up, down } = next {
                            TieredExperts::lookahead(&[gate, up, down], *depth, &rest);
                        }
                        rest = tail;
                    }
                    v
                };
                if gate.any_miss(&ids_host) || up.any_miss(&ids_host) {
                    qx.fetch(gate.device())?;
                }
                for p in [gate, up, down] {
                    p.prefetch(&ids_host);
                }
                let (gate, up) = TieredExperts::forward_gate_up(gate, up, &qx, indices, &ids_host)?;
                let activated = act(&gate, &up)?;
                let mut qa = down.quantize_input(&activated)?;
                if down.any_miss(&ids_host) {
                    qa.fetch(down.device())?;
                }
                down.forward_q(&qa, indices, Some(&ids_host))
            }
        }
    }
}

#[cfg(feature = "cuda")]
impl Experts {
    /// A prefill chunk with every routed expert on the GPU (`titan_pfs`): the (token, expert) tasks run in
    /// expert-major order, so each expert's weights are read from L2 by all its tokens in turn; `act` sees
    /// the projections in token order, as in `forward`. Each task is computed exactly as `forward` computes
    /// it. `None` when the staging ring could not be allocated.
    fn forward_streamed(
        gate: &TieredExperts,
        up: &TieredExperts,
        down: &TieredExperts,
        xs: &Tensor,
        indices: &Tensor,
        act: &impl Fn(&Tensor, &Tensor) -> Result<Tensor>,
    ) -> Result<Option<Tensor>> {
        let (tokens, topk) = indices.dims2()?;
        let hidden = xs.dim(2)?;
        let tasks = tokens * topk;
        let ids_host = TieredExperts::routing_to_host(indices)?;
        if [gate, up, down].iter().all(|p| p.streams_mmq()) {
            let x = xs.reshape((tokens, hidden))?;
            let Some(g) = gate.forward_streamed_mmq(&x, &ids_host, topk, false)? else {
                return Ok(None);
            };
            let Some(u) = up.forward_streamed_mmq(&x, &ids_host, topk, false)? else {
                candle_core::bail!("titan pfs: staging ring lost within a layer");
            };
            let to_tokens = |t: Tensor| -> Result<Tensor> {
                let n = t.dim(2)?;
                t.reshape((tokens, topk, n))
            };
            let activated = act(&to_tokens(g)?, &to_tokens(u)?)?;
            let n = activated.dim(2)?;
            let Some(y) = down.forward_streamed_mmq(&activated.reshape((tasks, n))?, &ids_host, topk, true)? else {
                candle_core::bail!("titan pfs: staging ring lost within a layer");
            };
            return Ok(Some(to_tokens(y)?));
        }
        let mut order: Vec<u32> = (0..tasks as u32).collect();
        order.sort_by_key(|&t| ids_host[t as usize]);
        let sorted: Vec<u32> = order.iter().map(|&t| ids_host[t as usize]).collect();
        let src_rows: Vec<u32> = order.iter().map(|&t| t / topk as u32).collect();
        let mut back = vec![0u32; tasks];
        for (i, &t) in order.iter().enumerate() {
            back[t as usize] = i as u32;
        }
        let dev = xs.device();
        let ids = Tensor::from_vec(sorted.clone(), (tasks, 1), dev)?;
        let src_rows = Tensor::from_vec(src_rows, tasks, dev)?;
        let order = Tensor::from_vec(order, tasks, dev)?;
        let back = Tensor::from_vec(back, tasks, dev)?;
        // sorted task rows <-> (tokens, topk, n) in token order
        let unsort = |t: &Tensor| -> Result<Tensor> {
            let n = t.dim(2)?;
            t.reshape((tasks, n))?.index_select(&back, 0)?.reshape((tokens, topk, n))
        };
        let x = xs
            .reshape((tokens, hidden))?
            .index_select(&src_rows, 0)?
            .reshape((tasks, 1, hidden))?;
        let qx = gate.quantize_input(&x)?;
        let Some(g) = gate.forward_streamed(&qx, &ids, &sorted)? else {
            return Ok(None);
        };
        let Some(u) = up.forward_streamed(&qx, &ids, &sorted)? else {
            candle_core::bail!("titan pfs: staging ring lost within a layer");
        };
        drop((qx, x));
        let activated = act(&unsort(&g)?, &unsort(&u)?)?;
        let n = activated.dim(2)?;
        let activated = activated
            .reshape((tasks, n))?
            .index_select(&order, 0)?
            .reshape((tasks, 1, n))?;
        let qa = down.quantize_input(&activated)?;
        let Some(y) = down.forward_streamed(&qa, &ids, &sorted)? else {
            candle_core::bail!("titan pfs: staging ring lost within a layer");
        };
        Ok(Some(unsort(&y)?))
    }
}

#[cfg(feature = "cuda")]
impl Experts {
    /// `forward_reduced` for a streamed MMQ chunk: gate and up over the whole chunk as `forward_streamed`, the
    /// down projection in pieces of `piece` tokens (`forward_streamed_mmq_pieces`). `None` when the staging ring
    /// could not be allocated.
    fn forward_streamed_reduced(
        gate: &TieredExperts,
        up: &TieredExperts,
        down: &TieredExperts,
        xs: &Tensor,
        indices: &Tensor,
        piece: usize,
        reduce: &impl Fn(usize, &Tensor) -> Result<Tensor>,
    ) -> Result<Option<Tensor>> {
        let (tokens, topk) = indices.dims2()?;
        let hidden = xs.dim(2)?;
        let tasks = tokens * topk;
        let ids_host = TieredExperts::routing_to_host(indices)?;
        let x = xs.reshape((tokens, hidden))?;
        let Some(g) = gate.forward_streamed_mmq(&x, &ids_host, topk, false)? else {
            return Ok(None);
        };
        let Some(u) = up.forward_streamed_mmq(&x, &ids_host, topk, false)? else {
            candle_core::bail!("titan pfs: staging ring lost within a layer");
        };
        let to_tokens = |t: Tensor| -> Result<Tensor> {
            let n = t.dim(2)?;
            t.reshape((t.dim(0)? / topk, topk, n))
        };
        let activated = crate::ops::mul_and_act(&to_tokens(g)?, &to_tokens(u)?, crate::layers::Activation::Silu)?;
        let n = activated.dim(2)?;
        let activated = activated.reshape((tasks, n))?;
        let mut outs = Vec::with_capacity(tokens.div_ceil(piece));
        let ok = down.forward_streamed_mmq_pieces(&activated, &ids_host, topk, true, piece * topk, |t0, y| {
            outs.push(reduce(t0 / topk, &to_tokens(y)?)?);
            Ok(())
        })?;
        if !ok {
            candle_core::bail!("titan pfs: staging ring lost within a layer");
        }
        Ok(Some(if outs.len() == 1 { outs.pop().expect("one piece") } else { Tensor::cat(&outs, 0)? }))
    }
}

struct FusedMoe {
    gate: QMatMul,
    experts: Experts,
    norm_topk_prob: bool,
    num_experts_per_tok: usize,
}

impl FusedMoe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, hidden_dim) = xs.dims3()?;
        let xs = xs.reshape(((), hidden_dim))?;
        let original_dtype = xs.dtype();
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let router_logits = self.gate.forward(&xs.to_dtype(DType::F32)?)?;
        let topk = crate::ops::moe_router_topk(
            &router_logits,
            crate::ops::MoeRouterTopKConfig {
                top_k: self.num_experts_per_tok,
                score_function: crate::ops::MoeRouterScoreFunction::Softmax,
                selected_weight: crate::ops::MoeRouterSelectedWeight::Score,
                renormalize: self.norm_topk_prob,
                norm_min: 0.0,
                output_scale: 1.0,
                logit_clip: None,
            },
            None,
            None,
        )?;
        let (scores, indices) = (topk.values, topk.indices);

        let ys = self
            .experts
            .forward(&xs.reshape((num_tokens, 1, hidden_dim))?, &indices)?;
        ys.broadcast_mul(&scores.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?
            .reshape((batch, seq_len, hidden_dim))?
            .to_dtype(original_dtype)
    }
}

/// Load a layer's routed experts: slotted for titan-engine when `TITAN_TIERED=1`, the layer is on
/// CUDA and all three projections are tierable; host experts in the GGUF mmap when the files are mapped.
pub(crate) fn load_experts<R: std::io::Seek + std::io::Read>(
    ct: &mut Content<'_, R>,
    prefix: &str,
    device: &Device,
) -> Result<Experts> {
    let names = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];
    #[cfg(feature = "cuda")]
    if let Device::Cuda(dev) = device {
        if TieredExperts::enabled() {
            let layer: usize = prefix
                .rsplit('.')
                .next()
                .and_then(|l| l.parse().ok())
                .unwrap_or(0);
            let full = names.map(|n| format!("{prefix}.{n}.weight"));
            let mapped = full
                .iter()
                .map(|n| {
                    let info = ct.tensor_info(n)?;
                    Ok((info.shape.dims3()?, info.ggml_dtype, ct.tensor_mapped(n)))
                })
                .collect::<Result<Vec<_>>>()?;
            if mapped.iter().all(|(_, dtype, m)| m.is_some() && TieredExperts::supports(*dtype)) {
                let [g, u, d] = [(0, "gate"), (1, "up"), (2, "down")].map(|(i, proj)| {
                    let (shape, dtype, m) = &mapped[i];
                    let (map, offset) = m.clone().expect("checked");
                    TieredExperts::from_mapped(map, offset, *shape, *dtype, dev, layer, proj)
                });
                return Ok(Experts::Tiered {
                    gate: g?,
                    up: u?,
                    down: d?,
                });
            }
            let host: Vec<_> = names
                .iter()
                .map(|n| ct.tensor(&format!("{prefix}.{n}.weight"), &Device::Cpu))
                .collect::<Result<_>>()?;
            if host.iter().all(|q| TieredExperts::supports(q.dtype())) {
                let g = TieredExperts::from_qtensor(&host[0], dev, layer, "gate");
                let u = TieredExperts::from_qtensor(&host[1], dev, layer, "up");
                let d = TieredExperts::from_qtensor(&host[2], dev, layer, "down");
                return Ok(Experts::Tiered {
                    gate: g?,
                    up: u?,
                    down: d?,
                });
            }
            tracing::warn!("{prefix}: expert dtypes not all Q4_K/Q5_K; using stock experts");
        }
    }
    let [gate, up, down] = names.map(|n| ct.tensor(&format!("{prefix}.{n}.weight"), device));
    Ok(Experts::Stock {
        gate: QMatMul::from_qtensor(gate?)?,
        up: QMatMul::from_qtensor(up?)?,
        down: QMatMul::from_qtensor(down?)?,
    })
}

enum MoeOrMlp {
    FusedMoe(FusedMoe),
    Mlp(Mlp),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(m) => m.forward(xs),
            Self::FusedMoe(m) => m.forward(xs),
        }
    }
}

struct LayerWeights {
    attention_wq: Arc<dyn QuantMethod>,
    attention_wk: Arc<dyn QuantMethod>,
    attention_wv: Arc<dyn QuantMethod>,
    attention_wo: Arc<dyn QuantMethod>,
    attention_norm: QRmsNorm,
    q_norm: QRmsNorm,
    k_norm: QRmsNorm,
    mlp: MoeOrMlp,
    ffn_norm: QRmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rotary: Arc<RotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    dtype: DType,
}

impl LayerWeights {
    fn forward_attn(
        &self,
        x: &Tensor,
        mask: &AttentionMask,
        start_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let (q, k, v) = if seq_len != 1 {
            let q = q
                .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.n_head, seq_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;
            (q, k, v)
        };

        let positions =
            crate::pipeline::text_positions_tensor(start_offsets, q.dim(2)?, q.device())?;
        let (q, k) = self.rotary.forward_qk_norm(
            &q,
            &k,
            self.q_norm.weight(),
            self.k_norm.weight(),
            self.q_norm.eps(),
            self.k_norm.eps(),
            &positions,
        )?;

        let (q, k, v) = (
            q.to_dtype(self.dtype)?,
            k.to_dtype(self.dtype)?,
            v.to_dtype(self.dtype)?,
        );

        let y = match &self.paged_attn {
            Some(paged_attn) => {
                let ((key_cache, value_cache), input_metadata) = metadata.unwrap();
                paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    mask,
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    None,
                )?
            }
            None => {
                let (k, v) = kv_cache.append(&k, &v)?;

                Sdpa.run_attention(&q, &k, &v, mask, None, &self.sdpa_params)?
            }
        };

        let y = if mask.is_custom() {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };

        let y = self.attention_wo.forward(&y.to_dtype(x.dtype())?)?;
        Ok(y)
    }
}

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: QRmsNorm,
    output: Arc<dyn QuantMethod>,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct QwenMoEConfig {
    pub moe_intermediate_size: usize,
    pub num_experts: Option<usize>,
    pub mlp_only_layers: Option<Vec<usize>>,
    pub decoder_sparse_step: Option<usize>,
    pub norm_topk_prob: bool,
    pub num_experts_per_tok: usize,
}

pub(crate) struct PropsGGUF {
    pub head_count: usize,
    pub head_count_kv: usize,
    pub block_count: usize,
    pub embedding_length: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    pub rope_freq_base: f32,
    pub key_length: usize,
    pub value_length: usize,
    pub moe_cfg: QwenMoEConfig,
}

fn verify_qwen3_arch(
    metadata: &HashMap<String, candle_core::quantized::gguf_file::Value>,
) -> Result<String> {
    use crate::utils::gguf_metadata::TryValueInto;
    let actual_arch: String = metadata
        .get("general.architecture")
        .cloned()
        .try_value_into()?;

    if actual_arch != "qwen3" && actual_arch != "qwen3moe" {
        candle_core::bail!("Expected `qwen3` architecture, got `{actual_arch}`.");
    }
    Ok(actual_arch)
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> std::result::Result<Self, Self::Error> {
        let _ = verify_qwen3_arch(c.metadata)?;

        let required = [
            "attention.head_count",
            "attention.head_count_kv",
            "block_count",
            "embedding_length",
            "attention.layer_norm_rms_epsilon",
        ];
        c.has_required_keys(&required)?;

        let embed_len = c.get_value::<u32>("embedding_length")? as usize;
        let head_count = c.get_value::<u32>("attention.head_count")? as usize;

        // NOTE: Values are not aligned with GGUFv3 types
        // TODO: Normalize value types to spec

        let moe_cfg = QwenMoEConfig {
            moe_intermediate_size: c.get_value::<u32>("expert_feed_forward_length")? as usize,
            num_experts: Some(c.get_value::<u32>("expert_count")? as usize),
            mlp_only_layers: Some(vec![]),
            decoder_sparse_step: Some(1),
            norm_topk_prob: true,
            num_experts_per_tok: c.get_value::<u32>("expert_used_count")? as usize,
        };

        let props = Self {
            head_count,
            head_count_kv: c.get_value::<u32>("attention.head_count_kv")? as usize,
            block_count: c.get_value::<u32>("block_count")? as usize,
            embedding_length: embed_len,
            rms_norm_eps: c.get_value("attention.layer_norm_rms_epsilon")?,
            max_seq_len: c
                .get_value::<u64>("context_length")
                .ok()
                .unwrap_or(DEFAULT_MAX_SEQ_LEN as u64) as usize,
            rope_freq_base: c.get_value("rope.freq_base").ok().unwrap_or(10_000_f32),
            key_length: c
                .get_value::<u32>("attention.key_length")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(embed_len / head_count),
            value_length: c
                .get_value::<u32>("attention.value_length")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(embed_len / head_count),
            moe_cfg,
        };

        Ok(props)
    }
}

impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>,
        device: &Device,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        attention_mechanism: AttentionImplementation,
        dtype: DType,
    ) -> Result<Self> {
        // Parameter extraction from metadata.
        let meta = ct.get_metadata();
        let actual_arch = verify_qwen3_arch(meta)?;

        let metadata = ContentMetadata {
            path_prefix: &actual_arch,
            metadata: meta,
        };
        let PropsGGUF {
            head_count,
            head_count_kv,
            block_count,
            embedding_length,
            rms_norm_eps,
            max_seq_len,
            rope_freq_base,
            key_length,
            value_length,
            moe_cfg,
        } = PropsGGUF::try_from(metadata).or_else(|err| candle_core::bail!("{err}"))?;

        let qtok_embeddings = ct.tensor("token_embd.weight", device)?;
        let tok_embeddings = qtok_embeddings.dequantize(device)?;
        let norm = QRmsNorm::new(ct.tensor("output_norm.weight", device)?, rms_norm_eps)?;
        let output = if !ct.has_tensor("output.weight") {
            ct.tensor("token_embd.weight", device)?
        } else {
            ct.tensor("output.weight", device)?
        };
        let mut layers = Vec::with_capacity(block_count);

        let head_dim = key_length;
        if key_length != value_length {
            candle_core::bail!(
                "Expected key_length == value_length, got {key_length} != {value_length}"
            );
        }

        let mut ropes = HashMap::new();
        for layer_idx in 0..block_count {
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            ropes.insert(
                device.location(),
                Arc::new(RotaryEmbedding::new(
                    rope_freq_base,
                    head_dim,
                    max_seq_len,
                    device,
                    true,
                    DType::F32,
                )?),
            );
        }

        for layer_idx in NiceProgressBar::<_, 'b'>(
            0..block_count,
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let prefix = format!("blk.{layer_idx}");
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            let rotary = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();

            let attention_wq = ct.tensor(&format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(&format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(&format!("{prefix}.attn_v.weight"), device)?;
            let attention_wo = ct.tensor(&format!("{prefix}.attn_output.weight"), device)?;

            let mlp = if !moe_cfg
                .mlp_only_layers
                .as_ref()
                .unwrap()
                .contains(&layer_idx)
                && (moe_cfg.num_experts.unwrap() > 0
                    && (layer_idx + 1) % moe_cfg.decoder_sparse_step.unwrap() == 0)
            {
                let gate = ct.tensor(&format!("{prefix}.ffn_gate_inp.weight"), device)?;
                let experts = load_experts(&mut ct, &prefix, device)?;
                let moe = FusedMoe {
                    gate: QMatMul::from_qtensor(gate)?,
                    experts,
                    norm_topk_prob: moe_cfg.norm_topk_prob,
                    num_experts_per_tok: moe_cfg.num_experts_per_tok,
                };

                MoeOrMlp::FusedMoe(moe)
            } else {
                let feed_forward_w1 = ct.tensor(&format!("{prefix}.ffn_gate.weight"), device)?;
                let feed_forward_w2 = ct.tensor(&format!("{prefix}.ffn_down.weight"), device)?;
                let feed_forward_w3 = ct.tensor(&format!("{prefix}.ffn_up.weight"), device)?;
                let mlp = Mlp {
                    feed_forward_w1: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(feed_forward_w1),
                        b: None,
                    })?),
                    feed_forward_w2: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(feed_forward_w2),
                        b: None,
                    })?),
                    feed_forward_w3: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(feed_forward_w3),
                        b: None,
                    })?),
                };
                MoeOrMlp::Mlp(mlp)
            };

            // Qwen3 always has q_norm and k_norm
            let q_norm = QRmsNorm::new(
                ct.tensor(&format!("{prefix}.attn_q_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let k_norm = QRmsNorm::new(
                ct.tensor(&format!("{prefix}.attn_k_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let attention_norm = ct.tensor(&format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(&format!("{prefix}.ffn_norm.weight"), device)?;
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(head_dim, device, None)?)
                }
            };
            layers.push(LayerWeights {
                attention_wq: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(attention_wq),
                    b: None,
                })?),
                attention_wk: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(attention_wk),
                    b: None,
                })?),
                attention_wv: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(attention_wv),
                    b: None,
                })?),
                attention_wo: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(attention_wo),
                    b: None,
                })?),
                attention_norm: QRmsNorm::new(attention_norm, rms_norm_eps)?,
                q_norm,
                k_norm,
                mlp,
                ffn_norm: QRmsNorm::new(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                rotary: rotary.clone(),
                paged_attn,
                sdpa_params: SdpaParams {
                    n_kv_groups: head_count / head_count_kv,
                    softcap: None,
                    softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                    sliding_window: None,
                    sinks: None,
                },
                dtype,
            })
        }
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(output),
                b: None,
            })?),
            device: device.clone(),
            cache: EitherCache::Normal(NormalCache::new(block_count, max_seq_len)),
            max_seq_len,
            mapper: Some(mapper),
            dtype,
        })
    }
}

impl ModelWeights {
    pub fn forward(
        &self,
        x: &Tensor,
        start_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let mut layer_in = self.tok_embeddings.forward(x)?;
        let cache = &mut self.cache.normal().0;
        let mask = CausalMasker.make_causal_mask(
            x,
            metadata
                .as_ref()
                .map(|(_, _)| &start_offsets as &dyn PastKvLenCache)
                .unwrap_or(cache as &dyn PastKvLenCache),
            self.dtype,
            &CausalMaskConfig::default(),
        )?;
        let mask = if metadata
            .as_ref()
            .map(|(_, meta)| meta.is_first_prompt_chunk)
            .unwrap_or(true)
        {
            mask
        } else {
            AttentionMask::None
        };
        let mask = if let Some(ref mapper) = self.mapper {
            DeviceMappedMask::new(mask, &**mapper)?
        } else {
            DeviceMappedMask::from_single(mask)
        };
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
                layer_in = mapper.map(layer_in, i)?;
            }
            let x = layer_in;
            let residual = &x;
            let x = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn(
                &x,
                &mask.get(x.device()),
                start_offsets,
                &mut cache[i],
                metadata
                    .as_ref()
                    .map(|(kv_cache, metadata)| (kv_cache[i].clone(), *metadata)),
            )?;
            let x = (attn + residual)?;

            // MLP
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp.forward(&x)?;
            let x = (x + residual)?;
            layer_in = x;
        }
        let x = self.norm.forward(&layer_in)?;
        let x = extract_logits(&x, context_lens)?;
        self.output.forward(&x.contiguous()?)
    }
}
