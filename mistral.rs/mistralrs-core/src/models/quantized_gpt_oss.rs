#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! GGUF arch `gpt-oss` (llama.cpp `LLM_ARCH_OPENAI_MOE`, src/models/openai-moe.cpp).

use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor, D};
use mistralrs_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

use super::quantized_qwen35_moe::QEmbedding;
use super::quantized_qwen3_moe::{load_experts, Experts};
use crate::attention::{AttentionMask, SdpaParams};
use crate::device_map::DeviceMapper;
use crate::gguf::Content;
use crate::layers::{GptOssRotaryEmbedding, QRmsNorm, Sdpa};
use crate::paged_attention::AttentionImplementation;
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::{extract_logits, EitherCache, KvCache, NormalCache};
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};

/// llama.cpp LLM_FFN_SWIGLU_OAI_MOE constants (not in the GGUF).
const SWIGLU_ALPHA: f32 = 1.702;
const SWIGLU_LIMIT: f32 = 7.0;
/// llama.cpp default when `{arch}.attention.sliding_window_pattern` is absent: even layers slide.
const DEFAULT_SWA_PERIOD: usize = 2;
const DEFAULT_YARN_BETA_FAST: f32 = 32.0;
const DEFAULT_YARN_BETA_SLOW: f32 = 1.0;

pub(crate) struct PropsGGUF {
    head_count: usize,
    head_count_kv: usize,
    block_count: usize,
    head_dim: usize,
    rms_norm_eps: f32,
    max_seq_len: usize,
    rope_freq_base: f32,
    yarn: Option<(f32, usize, f32, f32)>,
    expert_used_count: usize,
    sliding_window: usize,
    swa_period: usize,
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> std::result::Result<Self, Self::Error> {
        c.has_required_keys(&[
            "attention.head_count",
            "attention.head_count_kv",
            "block_count",
            "embedding_length",
            "attention.layer_norm_rms_epsilon",
            "expert_used_count",
            "attention.sliding_window",
        ])?;
        let embedding_length = c.get_value::<u32>("embedding_length")? as usize;
        let head_count = c.get_value::<u32>("attention.head_count")? as usize;
        let head_dim = c
            .get_value::<u32>("attention.key_length")
            .map(|v| v as usize)
            .unwrap_or(embedding_length / head_count);
        let yarn = match c.get_value::<String>("rope.scaling.type").ok().as_deref() {
            Some("yarn") => Some((
                c.get_value::<f32>("rope.scaling.factor")?,
                c.get_value::<u32>("rope.scaling.original_context_length")? as usize,
                c.get_value::<f32>("rope.scaling.yarn_beta_fast")
                    .unwrap_or(DEFAULT_YARN_BETA_FAST),
                c.get_value::<f32>("rope.scaling.yarn_beta_slow")
                    .unwrap_or(DEFAULT_YARN_BETA_SLOW),
            )),
            _ => None,
        };
        Ok(Self {
            head_count,
            head_count_kv: c.get_value::<u32>("attention.head_count_kv")? as usize,
            block_count: c.get_value::<u32>("block_count")? as usize,
            head_dim,
            rms_norm_eps: c.get_value("attention.layer_norm_rms_epsilon")?,
            max_seq_len: c.get_value::<u64>("context_length")? as usize,
            rope_freq_base: c.get_value("rope.freq_base")?,
            yarn,
            expert_used_count: c.get_value::<u32>("expert_used_count")? as usize,
            sliding_window: c.get_value::<u32>("attention.sliding_window")? as usize,
            swa_period: c
                .get_value::<u32>("attention.sliding_window_pattern")
                .map(|v| v as usize)
                .unwrap_or(DEFAULT_SWA_PERIOD),
        })
    }
}

fn linear(w: candle_core::quantized::QTensor, b: Option<Tensor>) -> Result<Arc<dyn QuantMethod>> {
    Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
        q_weight: Arc::new(w),
        b,
    })?))
}

fn f32_tensor<R: std::io::Seek + std::io::Read>(
    ct: &mut Content<'_, R>,
    name: &str,
    device: &Device,
) -> Result<Tensor> {
    ct.tensor(name, &Device::Cpu)?
        .dequantize(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .to_device(device)
}

fn swiglu_oai(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    return mistralrs_quant::gptoss_swiglu_fused(gate, up, SWIGLU_ALPHA, SWIGLU_LIMIT);
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (gate, up);
        candle_core::bail!("gpt-oss GGUF needs CUDA");
    }
}

struct Attention {
    wq: Arc<dyn QuantMethod>,
    wk: Arc<dyn QuantMethod>,
    wv: Arc<dyn QuantMethod>,
    wo: Arc<dyn QuantMethod>,
    rotary: Arc<GptOssRotaryEmbedding>,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    /// q / k / v and KV cache dtype (the pipeline's; `--dtype f16` matches llama.cpp's default cache)
    dtype: DType,
    sdpa_params: SdpaParams,
}

impl Attention {
    fn forward(&self, x: &Tensor, positions: &Tensor, kv_cache: &mut KvCache) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let q = self.wq.forward(x)?;
        let k = self.wk.forward(x)?;
        let v = self.wv.forward(x)?;
        let (q, k, v) = if seq_len != 1 {
            (
                q.reshape((b_sz, seq_len, self.n_head, self.head_dim))?
                    .transpose(1, 2)?,
                k.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                    .transpose(1, 2)?,
                v.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                    .transpose(1, 2)?,
            )
        } else {
            (
                q.reshape((b_sz, self.n_head, seq_len, self.head_dim))?,
                k.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?,
                v.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?,
            )
        };
        let (q, k) = self.rotary.forward(&q, &k, positions)?;
        let (q, k, v) = (
            q.to_dtype(self.dtype)?,
            k.to_dtype(self.dtype)?,
            v.to_dtype(self.dtype)?,
        );
        let (k, v) = kv_cache.append(&k, &v)?;
        // the sinks kernel applies causality and the window itself (key i visible iff pos - i < window)
        let y = Sdpa.run_attention(&q, &k, &v, &AttentionMask::None, None, &self.sdpa_params)?;
        let y = if seq_len != 1 {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };
        self.wo.forward(&y.to_dtype(DType::F32)?)
    }
}

struct Moe {
    router: Arc<dyn QuantMethod>,
    experts: Experts,
    /// `(experts, n)` each, added per routed (token, expert) like llama.cpp's ggml_add_id
    gate_b: Tensor,
    up_b: Tensor,
    down_b: Tensor,
    top_k: usize,
}

impl Moe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, hidden) = xs.dims3()?;
        let tokens = b_sz * seq_len;
        let flat = xs.reshape((tokens, hidden))?;
        let logits = self.router.forward(&flat)?;
        // llama.cpp SOFTMAX_WEIGHT: top-k on the raw logits, softmax over the selected ones
        let topk = crate::ops::moe_router_topk(
            &logits,
            crate::ops::MoeRouterTopKConfig {
                top_k: self.top_k,
                score_function: crate::ops::MoeRouterScoreFunction::Raw,
                selected_weight: crate::ops::MoeRouterSelectedWeight::Softmax,
                renormalize: false,
                norm_min: 0.0,
                output_scale: 1.0,
                logit_clip: None,
            },
            None,
            None,
        )?;
        let (weights, ids) = (topk.values, topk.indices);
        let ids_flat = ids.flatten_all()?;
        let routed_bias = |b: &Tensor| -> Result<Tensor> {
            let n = b.dim(1)?;
            b.index_select(&ids_flat, 0)?
                .reshape((tokens, self.top_k, n))
        };
        // Bias and activation run on the assembled (GPU slots + CPU twin rows) projections, so any
        // tiered split sees the same inputs to them.
        let ys =
            self.experts
                .forward_act(&flat.reshape((tokens, 1, hidden))?, &ids, |gate, up| {
                    let gate = (gate + routed_bias(&self.gate_b)?)?;
                    let up = (up + routed_bias(&self.up_b)?)?;
                    swiglu_oai(&gate, &up)
                })?;
        let ys = (ys + routed_bias(&self.down_b)?)?;
        ys.broadcast_mul(&weights.to_dtype(DType::F32)?.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?
            .reshape((b_sz, seq_len, hidden))
    }
}

struct Layer {
    attn_norm: QRmsNorm,
    attn: Attention,
    post_attn_norm: QRmsNorm,
    moe: Moe,
}

pub struct ModelWeights {
    embed: QEmbedding,
    layers: Vec<Layer>,
    norm: QRmsNorm,
    output: Arc<dyn QuantMethod>,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
}

impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>,
        device: &Device,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        attention_mechanism: AttentionImplementation,
        dtype: DType,
    ) -> Result<Self> {
        if matches!(attention_mechanism, AttentionImplementation::PagedAttention) {
            candle_core::bail!("gpt-oss GGUF: run with --paged-attn off");
        }
        let arch = ct.arch().to_string();
        let PropsGGUF {
            head_count,
            head_count_kv,
            block_count,
            head_dim,
            rms_norm_eps,
            max_seq_len,
            rope_freq_base,
            yarn,
            expert_used_count,
            sliding_window,
            swa_period,
        } = PropsGGUF::try_from(ContentMetadata {
            path_prefix: &arch,
            metadata: ct.get_metadata(),
        })
        .or_else(|err| candle_core::bail!("{err}"))?;

        let embed = QEmbedding::new(ct.tensor("token_embd.weight", &Device::Cpu)?)?;
        let norm = QRmsNorm::new(ct.tensor("output_norm.weight", device)?, rms_norm_eps)?;
        let output = linear(ct.tensor("output.weight", device)?, None)?;

        let mut ropes: std::collections::HashMap<_, Arc<GptOssRotaryEmbedding>> =
            std::collections::HashMap::new();
        let mut layers = Vec::with_capacity(block_count);
        for il in NiceProgressBar::<_, 'b'>(
            0..block_count,
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let device = mapper.device_for(il, false).unwrap_or(device);
            if !device.is_cuda() {
                candle_core::bail!(
                    "gpt-oss GGUF: layer {il} was mapped to {device:?}; every layer must be on the GPU \
                     (spill experts with TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION instead)"
                );
            }
            let loc = device.location();
            if !ropes.contains_key(&loc) {
                // truncate: llama.cpp's ggml_rope_yarn_corr_dims floors / ceils the ramp ends
                let (factor, orig_ctx, beta_fast, beta_slow) = yarn.unwrap_or((
                    1.0,
                    max_seq_len,
                    DEFAULT_YARN_BETA_FAST,
                    DEFAULT_YARN_BETA_SLOW,
                ));
                let r = GptOssRotaryEmbedding::new(
                    rope_freq_base as f64,
                    head_dim,
                    max_seq_len,
                    factor as f64,
                    orig_ctx,
                    beta_fast as f64,
                    beta_slow as f64,
                    true,
                    device,
                    DType::F32,
                )?;
                ropes.insert(loc, Arc::new(r));
            }
            let rotary = ropes[&loc].clone();
            let p = format!("blk.{il}");
            let mut lin = |name: &str| -> Result<Arc<dyn QuantMethod>> {
                let w = ct.tensor(&format!("{p}.{name}.weight"), device)?;
                let b = f32_tensor(&mut ct, &format!("{p}.{name}.bias"), device)?;
                linear(w, Some(b))
            };
            let (wq, wk, wv, wo) = (
                lin("attn_q")?,
                lin("attn_k")?,
                lin("attn_v")?,
                lin("attn_output")?,
            );
            let router = lin("ffn_gate_inp")?;
            let is_swa = swa_period > 0 && il % swa_period < swa_period - 1;
            let attn = Attention {
                wq,
                wk,
                wv,
                wo,
                rotary,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                dtype,
                sdpa_params: SdpaParams {
                    n_kv_groups: head_count / head_count_kv,
                    softcap: None,
                    softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                    sliding_window: is_swa.then_some(sliding_window),
                    sinks: Some(f32_tensor(
                        &mut ct,
                        &format!("{p}.attn_sinks.weight"),
                        device,
                    )?),
                },
            };
            let moe = Moe {
                router,
                experts: load_gpt_oss_experts(&mut ct, &p, device)?,
                gate_b: f32_tensor(&mut ct, &format!("{p}.ffn_gate_exps.bias"), device)?,
                up_b: f32_tensor(&mut ct, &format!("{p}.ffn_up_exps.bias"), device)?,
                down_b: f32_tensor(&mut ct, &format!("{p}.ffn_down_exps.bias"), device)?,
                top_k: expert_used_count,
            };
            layers.push(Layer {
                attn_norm: QRmsNorm::new(
                    ct.tensor(&format!("{p}.attn_norm.weight"), device)?,
                    rms_norm_eps,
                )?,
                attn,
                post_attn_norm: QRmsNorm::new(
                    ct.tensor(&format!("{p}.post_attention_norm.weight"), device)?,
                    rms_norm_eps,
                )?,
                moe,
            });
        }
        Ok(Self {
            embed,
            layers,
            norm,
            output,
            device: device.clone(),
            // full caches on the sliding layers too: the kernel windows them
            cache: EitherCache::Normal(NormalCache::new(block_count, max_seq_len)),
            max_seq_len,
            mapper: Some(mapper),
        })
    }
}

/// candle has no MXFP4 `indexed_moe_forward`, so without `TITAN_TIERED=1` every expert still goes
/// in titan tiered GPU slots (the fraction defaults to 1: nothing on the CPU).
fn load_gpt_oss_experts<R: std::io::Seek + std::io::Read>(
    ct: &mut Content<'_, R>,
    prefix: &str,
    device: &Device,
) -> Result<Experts> {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(dev) = device {
        use mistralrs_quant::TieredExperts;
        if !TieredExperts::enabled() {
            let layer: usize = prefix
                .rsplit('.')
                .next()
                .and_then(|l| l.parse().ok())
                .unwrap_or(0);
            let [gate, up, down] = [
                ("ffn_gate_exps", "gate"),
                ("ffn_up_exps", "up"),
                ("ffn_down_exps", "down"),
            ]
            .map(|(n, proj)| {
                let q = ct.tensor(&format!("{prefix}.{n}.weight"), &Device::Cpu)?;
                TieredExperts::from_qtensor(&q, dev, layer, proj)
            });
            return Ok(Experts::Tiered {
                gate: gate?,
                up: up?,
                down: down?,
            });
        }
    }
    load_experts(ct, prefix, device)
}

impl ModelWeights {
    pub fn forward(
        &self,
        x: &Tensor,
        start_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        if metadata.is_some() {
            candle_core::bail!("gpt-oss GGUF: PagedAttention is not supported");
        }
        let mut xs = self.embed.forward(x, &self.device)?;
        let seq_len = x.dim(1)?;
        let cache = &mut self.cache.normal().0;
        let mut positions: Option<Tensor> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
                xs = mapper.map(xs, i)?;
            }
            let pos = match &positions {
                Some(p) if p.device().same_device(xs.device()) => p.clone(),
                _ => {
                    let p = crate::pipeline::text_positions_tensor(
                        start_offsets,
                        seq_len,
                        xs.device(),
                    )?;
                    positions = Some(p.clone());
                    p
                }
            };
            let residual = &xs;
            let h = layer
                .attn
                .forward(&layer.attn_norm.forward(&xs)?, &pos, &mut cache[i])?;
            let xs1 = (h + residual)?;
            let h = layer.moe.forward(&layer.post_attn_norm.forward(&xs1)?)?;
            xs = (h + &xs1)?;
        }
        let xs = xs.to_device(&self.device)?;
        let xs = self.norm.forward(&xs)?;
        let xs = extract_logits(&xs, context_lens)?;
        self.output.forward(&xs.contiguous()?)
    }
}
