#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! GGUF `kimi-linear` (Kimi Linear 48B-A3B), after llama.cpp `src/models/kimi-linear.cpp`: KDA delta-rule
//! layers with a per-channel decay, NoPE MLA layers kept as a 576-wide latent cache, and sigmoid-routed MoE.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use candle_core::quantized::gguf_file::Value;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Result, Tensor, D};
use mistralrs_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

use super::quantized_qwen3_moe::{load_experts, Experts};
use super::quantized_qwen35_moe::QEmbedding;
use crate::device_map::DeviceMapper;
use crate::gguf::Content;
use crate::kv_cache::{
    HybridCache, HybridCacheConfig, HybridLayerCache, HybridLayerType, KvCache, RecurrentLayerConfig,
    RecurrentStatePool, RecurrentStateSpec,
};
use crate::layers::QRmsNorm;
use crate::pipeline::{EitherCache, ModelForwardContext, RecurrentBatchKind, RecurrentMetadata};
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};
use crate::paged_attention::AttentionImplementation;

const ARCH: &str = "kimi-linear";
const PREFILL_CHUNK: usize = 512;
/// Largest attention score block (bytes) one query block may build.
const SCORES_BYTES: usize = 512 << 20;
const L2_EPS: f64 = 1e-6;
const RECURRENT_SLOTS: usize = 2;
const GATING_SIGMOID: u64 = 2;

fn md_u64(md: &HashMap<String, Value>, key: &str) -> Result<u64> {
    match md.get(key) {
        Some(v) => value_u64(v).ok_or_else(|| candle_core::Error::msg(format!("{ARCH}: `{key}` is not an integer"))),
        None => candle_core::bail!("{ARCH}: missing metadata `{key}`"),
    }
}

fn value_u64(v: &Value) -> Option<u64> {
    Some(match v {
        Value::U8(x) => u64::from(*x),
        Value::U16(x) => u64::from(*x),
        Value::U32(x) => u64::from(*x),
        Value::U64(x) => *x,
        Value::I8(x) => u64::try_from(*x).ok()?,
        Value::I16(x) => u64::try_from(*x).ok()?,
        Value::I32(x) => u64::try_from(*x).ok()?,
        Value::I64(x) => u64::try_from(*x).ok()?,
        _ => return None,
    })
}

fn md_f32(md: &HashMap<String, Value>, key: &str) -> Result<f32> {
    match md.get(key) {
        Some(Value::F32(x)) => Ok(*x),
        Some(Value::F64(x)) => Ok(*x as f32),
        _ => candle_core::bail!("{ARCH}: missing float metadata `{key}`"),
    }
}

struct Hp {
    n_layer: usize,
    n_head: usize,
    kda_dim: usize,
    conv_k: usize,
    kv_lora: usize,
    rope_dim: usize,
    nope_dim: usize,
    v_dim: usize,
    top_k: usize,
    expert_scale: f32,
    rms_eps: f32,
    max_seq_len: usize,
    is_mla: Vec<bool>,
    dense_lead: usize,
    sigmoid_gating: bool,
}

impl Hp {
    fn from_metadata(md: &HashMap<String, Value>) -> Result<Self> {
        let key = |s: &str| format!("{ARCH}.{s}");
        let n_layer = md_u64(md, &key("block_count"))? as usize;
        let kv_heads = match md.get(&key("attention.head_count_kv")) {
            Some(Value::Array(a)) => a.iter().map(|v| value_u64(v).unwrap_or(0)).collect::<Vec<_>>(),
            _ => candle_core::bail!("{ARCH}: head_count_kv must list every layer"),
        };
        if kv_heads.len() != n_layer {
            candle_core::bail!("{ARCH}: head_count_kv has {} entries for {n_layer} layers", kv_heads.len());
        }
        let rope_dim = md_u64(md, &key("rope.dimension_count"))? as usize;
        let k_mla = md_u64(md, &key("attention.key_length_mla"))? as usize;
        Ok(Self {
            n_layer,
            n_head: md_u64(md, &key("attention.head_count"))? as usize,
            kda_dim: md_u64(md, &key("kda.head_dim"))? as usize,
            conv_k: md_u64(md, &key("ssm.conv_kernel"))? as usize,
            kv_lora: md_u64(md, &key("attention.kv_lora_rank"))? as usize,
            rope_dim,
            nope_dim: k_mla - rope_dim,
            v_dim: md_u64(md, &key("attention.value_length_mla"))? as usize,
            top_k: md_u64(md, &key("expert_used_count"))? as usize,
            expert_scale: md_f32(md, &key("expert_weights_scale"))?,
            rms_eps: md_f32(md, &key("attention.layer_norm_rms_epsilon"))?,
            max_seq_len: md_u64(md, &key("context_length"))? as usize,
            is_mla: kv_heads.iter().map(|&h| h > 0).collect(),
            dense_lead: md_u64(md, &key("leading_dense_block_count"))? as usize,
            sigmoid_gating: md_u64(md, &key("expert_gating_func"))? == GATING_SIGMOID,
        })
    }

    fn inner(&self) -> usize {
        self.n_head * self.kda_dim
    }

    fn mla_head(&self) -> usize {
        self.nope_dim + self.rope_dim
    }
}

fn linear(q: QTensor) -> Result<Arc<dyn QuantMethod>> {
    Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
        q_weight: Arc::new(q),
        b: None,
    })?))
}

fn load_linear<R: std::io::Seek + std::io::Read>(
    ct: &mut Content<'_, R>,
    name: &str,
    device: &Device,
) -> Result<Arc<dyn QuantMethod>> {
    linear(ct.tensor(name, device)?)
}

fn load_f32<R: std::io::Seek + std::io::Read>(ct: &mut Content<'_, R>, name: &str, device: &Device) -> Result<Tensor> {
    ct.tensor(name, device)?.dequantize(device)?.to_dtype(DType::F32)
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    x.relu()? + (x.abs()?.neg()?.exp()? + 1.0)?.log()?
}

fn l2_normalize(x: &Tensor) -> Result<Tensor> {
    let ss = x.sqr()?.sum_keepdim(D::Minus1)?;
    x.broadcast_mul(&(ss + L2_EPS)?.sqrt()?.recip()?)
}

struct Kda {
    wq: Arc<dyn QuantMethod>,
    wk: Arc<dyn QuantMethod>,
    wv: Arc<dyn QuantMethod>,
    /// q, k, v depthwise conv taps stacked on the channel axis: `(3 * inner, conv_k)`
    conv: Tensor,
    f_a: Arc<dyn QuantMethod>,
    f_b: Arc<dyn QuantMethod>,
    beta: Arc<dyn QuantMethod>,
    g_a: Arc<dyn QuantMethod>,
    g_b: Arc<dyn QuantMethod>,
    /// `-exp(A_log)` per head, `(n_head,)`
    a: Tensor,
    dt_bias: Tensor,
    o_norm: Tensor,
    wo: Arc<dyn QuantMethod>,
}

impl Kda {
    fn load<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        prefix: &str,
        hp: &Hp,
        device: &Device,
    ) -> Result<Self> {
        let inner = hp.inner();
        let mut taps = Vec::with_capacity(3);
        for n in ["q", "k", "v"] {
            taps.push(load_f32(ct, &format!("{prefix}.ssm_conv1d_{n}.weight"), device)?.reshape((inner, hp.conv_k))?);
        }
        let conv = Tensor::cat(&taps, 0)?;
        Ok(Self {
            wq: load_linear(ct, &format!("{prefix}.attn_q.weight"), device)?,
            wk: load_linear(ct, &format!("{prefix}.attn_k.weight"), device)?,
            wv: load_linear(ct, &format!("{prefix}.attn_v.weight"), device)?,
            conv,
            f_a: load_linear(ct, &format!("{prefix}.ssm_f_a.weight"), device)?,
            f_b: load_linear(ct, &format!("{prefix}.ssm_f_b.weight"), device)?,
            beta: load_linear(ct, &format!("{prefix}.ssm_beta.weight"), device)?,
            g_a: load_linear(ct, &format!("{prefix}.ssm_g_a.weight"), device)?,
            g_b: load_linear(ct, &format!("{prefix}.ssm_g_b.weight"), device)?,
            a: load_f32(ct, &format!("{prefix}.ssm_a"), device)?.reshape(hp.n_head)?,
            dt_bias: load_f32(ct, &format!("{prefix}.ssm_dt.bias"), device)?,
            o_norm: load_f32(ct, &format!("{prefix}.ssm_norm.weight"), device)?,
            wo: load_linear(ct, &format!("{prefix}.attn_output.weight"), device)?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        pool: &mut RecurrentStatePool,
        md: &RecurrentMetadata,
        fresh: bool,
        hp: &Hp,
    ) -> Result<Tensor> {
        let (n, t, _) = x.dims3()?;
        let (heads, dim, inner) = (hp.n_head, hp.kda_dim, hp.inner());
        let indices = if md.state_indices().device().same_device(pool.device()) {
            md.state_indices().clone()
        } else {
            md.state_indices().to_device(pool.device())?
        };
        let (conv_state, rec_state) = if fresh {
            (
                Tensor::zeros((n, 3 * inner, hp.conv_k - 1), DType::F32, x.device())?,
                Tensor::zeros((n, heads, dim, dim), DType::F32, x.device())?,
            )
        } else {
            (
                pool.gather_conv_state(&indices)?.to_dtype(DType::F32)?,
                pool.gather_recurrent_state(&indices)?.to_dtype(DType::F32)?,
            )
        };

        let proj = Tensor::cat(&[self.wq.forward(x)?, self.wk.forward(x)?, self.wv.forward(x)?], D::Minus1)?
            .to_dtype(DType::F32)?;
        let (qkv, new_conv) = self.causal_conv(&proj, &conv_state, hp)?;
        let q = l2_normalize(&qkv.narrow(D::Minus1, 0, inner)?.reshape((n, t, heads, dim))?.contiguous()?)?;
        let q = (q * (dim as f64).powf(-0.5))?;
        let k = l2_normalize(&qkv.narrow(D::Minus1, inner, inner)?.reshape((n, t, heads, dim))?.contiguous()?)?;
        let v = qkv.narrow(D::Minus1, 2 * inner, inner)?.reshape((n, t, heads, dim))?.contiguous()?;

        let g = softplus(&self.f_b.forward(&self.f_a.forward(x)?)?.to_dtype(DType::F32)?.broadcast_add(&self.dt_bias)?)?
            .reshape((n, t, heads, dim))?
            .broadcast_mul(&self.a.reshape((1, 1, heads, 1))?)?;
        let beta = candle_nn::ops::sigmoid(&self.beta.forward(x)?.to_dtype(DType::F32)?)?;

        let (o, new_rec) = delta_scan(&q, &k, &v, &g, &beta, rec_state)?;

        let o = candle_nn::ops::rms_norm(&o.contiguous()?, &self.o_norm, hp.rms_eps)?;
        let gate = candle_nn::ops::sigmoid(&self.g_b.forward(&self.g_a.forward(x)?)?.to_dtype(DType::F32)?)?
            .reshape((n, t, heads, dim))?;
        let out = self.wo.forward(&(o * gate)?.reshape((n, t, inner))?)?;

        pool.scatter_conv_state_with_host_indices(&indices, md.state_indices_host(), &new_conv.to_dtype(pool.conv_dtype())?)?;
        pool.scatter_recurrent_state_with_host_indices(
            &indices,
            md.state_indices_host(),
            &new_rec.to_dtype(pool.recurrent_dtype())?,
        )?;
        out.to_dtype(DType::F32)
    }

    /// SiLU of a causal depthwise conv over the stacked q, k, v projections `(n, t, 3 * inner)`; the last
    /// `conv_k - 1` raw inputs per channel come back as the next state.
    fn causal_conv(&self, proj: &Tensor, state: &Tensor, hp: &Hp) -> Result<(Tensor, Tensor)> {
        let t = proj.dim(1)?;
        let hist = hp.conv_k - 1;
        let xcat = Tensor::cat(&[state, &proj.transpose(1, 2)?.contiguous()?], 2)?;
        let mut acc: Option<Tensor> = None;
        for j in 0..hp.conv_k {
            let w = self.conv.narrow(1, j, 1)?.unsqueeze(0)?;
            let term = xcat.narrow(2, j, t)?.broadcast_mul(&w)?;
            acc = Some(match acc {
                Some(a) => (a + term)?,
                None => term,
            });
        }
        let y = acc.expect("conv_k > 0").silu()?.transpose(1, 2)?.contiguous()?;
        Ok((y, xcat.narrow(2, t, hist)?.contiguous()?))
    }
}

/// Gated delta rule with a per-key-channel log-decay `g`; `s` is `(n, heads, k, v)`. Returns `(o, s)` with
/// `o` shaped `(n, t, heads, v)`.
fn delta_scan(q: &Tensor, k: &Tensor, v: &Tensor, g: &Tensor, beta: &Tensor, mut s: Tensor) -> Result<(Tensor, Tensor)> {
    let t = q.dim(1)?;
    let decay = g.exp()?;
    let step = |x: &Tensor, i: usize| x.narrow(1, i, 1)?.squeeze(1)?.contiguous();
    let mut rows = Vec::with_capacity(t);
    for i in 0..t {
        s = s.broadcast_mul(&step(&decay, i)?.unsqueeze(3)?)?;
        let ki = step(k, i)?;
        let pred = ki.unsqueeze(2)?.matmul(&s)?.squeeze(2)?;
        let delta = (step(v, i)? - pred)?.broadcast_mul(&step(beta, i)?.unsqueeze(2)?)?;
        s = (&s + ki.unsqueeze(3)?.matmul(&delta.unsqueeze(2)?)?)?;
        rows.push(step(q, i)?.unsqueeze(2)?.matmul(&s)?);
    }
    Ok((Tensor::cat(&rows, 2)?.transpose(1, 2)?.contiguous()?, s))
}

struct Mla {
    wq: Arc<dyn QuantMethod>,
    wkv_a: Arc<dyn QuantMethod>,
    kv_norm: QRmsNorm,
    /// per head `(nope_dim, kv_lora)` absorbed into the query
    wk_b: Tensor,
    /// per head `(kv_lora, v_dim)` applied to the attended latent
    wv_b: Tensor,
    wo: Arc<dyn QuantMethod>,
}

impl Mla {
    fn load<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        prefix: &str,
        hp: &Hp,
        device: &Device,
    ) -> Result<Self> {
        if ct.has_tensor(&format!("{prefix}.attn_q_a.weight")) {
            candle_core::bail!("{ARCH}: {prefix} compresses the query, which no Kimi Linear checkpoint does");
        }
        if !ct.has_tensor(&format!("{prefix}.attn_k_b.weight")) {
            candle_core::bail!("{ARCH}: {prefix} lacks split attn_k_b / attn_v_b; re-convert with a current llama.cpp");
        }
        Ok(Self {
            wq: load_linear(ct, &format!("{prefix}.attn_q.weight"), device)?,
            wkv_a: load_linear(ct, &format!("{prefix}.attn_kv_a_mqa.weight"), device)?,
            kv_norm: QRmsNorm::new(ct.tensor(&format!("{prefix}.attn_kv_a_norm.weight"), device)?, hp.rms_eps)?,
            wk_b: load_f32(ct, &format!("{prefix}.attn_k_b.weight"), device)?.transpose(1, 2)?.contiguous()?,
            wv_b: load_f32(ct, &format!("{prefix}.attn_v_b.weight"), device)?.transpose(1, 2)?.contiguous()?,
            wo: load_linear(ct, &format!("{prefix}.attn_output.weight"), device)?,
        })
    }

    fn forward(&self, x: &Tensor, kv: &mut KvCache, tails: &TailMasks, hp: &Hp) -> Result<Tensor> {
        let (n, t, _) = x.dims3()?;
        let heads = hp.n_head;
        let latent_dim = hp.kv_lora + hp.rope_dim;

        let q = self.wq.forward(x)?.to_dtype(DType::F32)?.reshape((n, t, heads, hp.mla_head()))?;
        let q_nope = q.narrow(3, 0, hp.nope_dim)?.transpose(1, 2)?.contiguous()?;
        let q_pe = q.narrow(3, hp.nope_dim, hp.rope_dim)?.transpose(1, 2)?.contiguous()?;
        let q_full = Tensor::cat(&[q_nope.broadcast_matmul(&self.wk_b.unsqueeze(0)?)?, q_pe], 3)?;

        let kv_a = self.wkv_a.forward(x)?.to_dtype(DType::F32)?;
        let c = self.kv_norm.forward(&kv_a.narrow(2, 0, hp.kv_lora)?.contiguous()?)?;
        let latent = Tensor::cat(&[&c, &kv_a.narrow(2, hp.kv_lora, hp.rope_dim)?], 2)?;
        let dummy = Tensor::zeros((n, 1, t, 1), DType::BF16, x.device())?;
        let (keys, _) = kv.append(&latent.to_dtype(DType::BF16)?.unsqueeze(1)?, &dummy)?;
        let total = keys.dim(2)?;
        let past = total - t;
        let keys = keys.squeeze(1)?.to_dtype(DType::F32)?;
        let keys_t = keys.transpose(1, 2)?;

        let scale = (hp.mla_head() as f64).powf(-0.5);
        let block = (SCORES_BYTES / (4 * n * heads * total)).clamp(1, t);
        let mut outs = Vec::with_capacity(t.div_ceil(block));
        for b0 in (0..t).step_by(block) {
            let l = block.min(t - b0);
            let end = past + b0 + l;
            let qb = q_full.narrow(2, b0, l)?.contiguous()?.reshape((n, heads * l, latent_dim))?;
            let mut scores = (qb.matmul(&keys_t.narrow(2, 0, end)?)? * scale)?.reshape((n, heads, l, end))?;
            if l > 1 {
                let mask = Tensor::cat(&[Tensor::zeros((l, end - l), DType::F32, x.device())?, tails.get(l, x.device())?], 1)?;
                scores = scores.broadcast_add(&mask)?;
            }
            let probs = candle_nn::ops::softmax_last_dim(&scores)?.reshape((n, heads * l, end))?;
            let o = probs.matmul(&keys.narrow(1, 0, end)?)?.narrow(2, 0, hp.kv_lora)?;
            outs.push(o.reshape((n, heads, l, hp.kv_lora))?);
        }
        let o = if outs.len() == 1 { outs.remove(0) } else { Tensor::cat(&outs, 2)? };
        let o = o.broadcast_matmul(&self.wv_b.unsqueeze(0)?)?.transpose(1, 2)?.contiguous()?;
        self.wo.forward(&o.reshape((n, t, heads * hp.v_dim))?)?.to_dtype(DType::F32)
    }
}

/// Causal `(l, l)` masks (0 on and below the diagonal, -inf above), built once per block length.
struct TailMasks(Mutex<HashMap<usize, Tensor>>);

impl TailMasks {
    fn get(&self, l: usize, device: &Device) -> Result<Tensor> {
        let mut cache = self.0.lock().expect("tail masks poisoned");
        if let Some(m) = cache.get(&l) {
            return Ok(m.clone());
        }
        let data = (0..l * l).map(|i| if i % l > i / l { f32::NEG_INFINITY } else { 0.0 }).collect::<Vec<_>>();
        let m = Tensor::from_vec(data, (l, l), device)?;
        cache.insert(l, m.clone());
        Ok(m)
    }
}

struct Mlp {
    gate: Arc<dyn QuantMethod>,
    up: Arc<dyn QuantMethod>,
    down: Arc<dyn QuantMethod>,
}

impl Mlp {
    fn load<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        prefix: &str,
        suffix: &str,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            gate: load_linear(ct, &format!("{prefix}.ffn_gate{suffix}.weight"), device)?,
            up: load_linear(ct, &format!("{prefix}.ffn_up{suffix}.weight"), device)?,
            down: load_linear(ct, &format!("{prefix}.ffn_down{suffix}.weight"), device)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = crate::ops::mul_and_act(&self.gate.forward(x)?, &self.up.forward(x)?, crate::layers::Activation::Silu)?;
        self.down.forward(&h)?.to_dtype(DType::F32)
    }
}

struct Moe {
    /// `(hidden, n_expert)`
    router: Tensor,
    select_bias: Tensor,
    experts: Experts,
    shared: Mlp,
    top_k: usize,
    scale: f32,
}

impl Moe {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let hidden = *dims.last().expect("rank > 0");
        let flat = x.reshape(((), hidden))?.to_dtype(DType::F32)?;
        let tokens = flat.dim(0)?;
        let topk = crate::ops::moe_router_topk(
            &flat.matmul(&self.router)?,
            crate::ops::MoeRouterTopKConfig {
                top_k: self.top_k,
                score_function: crate::ops::MoeRouterScoreFunction::Sigmoid,
                selected_weight: crate::ops::MoeRouterSelectedWeight::Score,
                renormalize: true,
                norm_min: ROUTER_NORM_MIN,
                output_scale: self.scale,
                logit_clip: None,
            },
            Some(&self.select_bias),
            None,
        )?;
        let routed = self.experts.forward(&flat.reshape((tokens, 1, hidden))?, &topk.indices)?;
        let routed = routed
            .broadcast_mul(&topk.values.to_dtype(routed.dtype())?.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?
            .to_dtype(DType::F32)?;
        (routed + self.shared.forward(&flat)?)?.reshape(dims)
    }
}

/// llama.cpp clamps the routing weight sum at this before dividing.
const ROUTER_NORM_MIN: f32 = 6.103_515_6e-5;

enum Mixer {
    Kda(Kda),
    Mla(Mla),
}

enum Ffn {
    Dense(Mlp),
    Moe(Moe),
}

struct Layer {
    attn_norm: QRmsNorm,
    mixer: Mixer,
    ffn_norm: QRmsNorm,
    ffn: Ffn,
}

pub struct ModelWeights {
    tok_embeddings: QEmbedding,
    layers: Vec<Layer>,
    norm: QRmsNorm,
    output: Arc<dyn QuantMethod>,
    pub(crate) device: Device,
    pub(crate) cache: EitherCache,
    pub(crate) max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    hp: Hp,
    tails: TailMasks,
}

impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>,
        device: &Device,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        _attention_mechanism: AttentionImplementation,
        _dtype: DType,
    ) -> Result<Self> {
        let hp = Hp::from_metadata(ct.get_metadata())?;
        let n_expert = md_u64(ct.get_metadata(), &format!("{ARCH}.expert_count"))? as usize;
        if !hp.sigmoid_gating {
            candle_core::bail!("{ARCH}: only sigmoid expert gating is implemented");
        }
        let tok_embeddings = QEmbedding::new(ct.tensor("token_embd.weight", &Device::Cpu)?)?;
        let norm = QRmsNorm::new(ct.tensor("output_norm.weight", device)?, hp.rms_eps)?;
        let output = load_linear(&mut ct, "output.weight", device)?;

        let mut layers = Vec::with_capacity(hp.n_layer);
        for i in NiceProgressBar::<_, 'b'>(0..hp.n_layer, "Loading repeating layers", &new_multi_progress()) {
            let prefix = format!("blk.{i}");
            let device = mapper.device_for(i, false).unwrap_or(device);
            let mixer = if hp.is_mla[i] {
                Mixer::Mla(Mla::load(&mut ct, &prefix, &hp, device)?)
            } else {
                Mixer::Kda(Kda::load(&mut ct, &prefix, &hp, device)?)
            };
            let ffn = if i < hp.dense_lead {
                Ffn::Dense(Mlp::load(&mut ct, &prefix, "", device)?)
            } else {
                let router = load_f32(&mut ct, &format!("{prefix}.ffn_gate_inp.weight"), device)?;
                if router.dim(0)? != n_expert {
                    candle_core::bail!("{ARCH}: {prefix} router has {} rows for {n_expert} experts", router.dim(0)?);
                }
                Ffn::Moe(Moe {
                    router: router.t()?.contiguous()?,
                    select_bias: load_f32(&mut ct, &format!("{prefix}.exp_probs_b.bias"), device)?,
                    experts: load_experts(&mut ct, &prefix, device)?,
                    shared: Mlp::load(&mut ct, &prefix, "_shexp", device)?,
                    top_k: hp.top_k,
                    scale: hp.expert_scale,
                })
            };
            layers.push(Layer {
                attn_norm: QRmsNorm::new(ct.tensor(&format!("{prefix}.attn_norm.weight"), device)?, hp.rms_eps)?,
                mixer,
                ffn_norm: QRmsNorm::new(ct.tensor(&format!("{prefix}.ffn_norm.weight"), device)?, hp.rms_eps)?,
                ffn,
            });
        }

        let layer_devices = (0..hp.n_layer)
            .map(|i| mapper.device_for(i, false).unwrap_or(device).clone())
            .collect::<Vec<_>>();
        let n_mla = hp.is_mla.iter().filter(|m| **m).count();
        tracing::info!("{ARCH}: {n_mla} MLA layers, {} KDA layers", hp.n_layer - n_mla);
        let cache = HybridCache::with_slots(
            HybridCacheConfig {
                layer_types: hp
                    .is_mla
                    .iter()
                    .map(|&m| if m { HybridLayerType::Attention } else { HybridLayerType::Recurrent })
                    .collect(),
                max_seq_len: hp.max_seq_len,
                recurrent: RecurrentLayerConfig {
                    conv_dim: 3 * hp.inner(),
                    conv_width: hp.conv_k - 1,
                    state: RecurrentStateSpec::Opaque { dims: vec![hp.n_head, hp.kda_dim, hp.kda_dim] },
                    recurrent_dtype: Some(DType::F32),
                },
            },
            DType::F32,
            &layer_devices,
            RECURRENT_SLOTS,
        )?;

        Ok(Self {
            tok_embeddings,
            layers,
            norm,
            output,
            device: device.clone(),
            cache: EitherCache::Hybrid(Arc::new(Mutex::new(cache))),
            max_seq_len: hp.max_seq_len,
            mapper: Some(mapper),
            hp,
            tails: TailMasks(Mutex::new(HashMap::new())),
        })
    }
}

impl ModelWeights {
    pub fn forward(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        let prefill = matches!(ctx.recurrent_batch_kind(), Some(RecurrentBatchKind::Prefill));
        if prefill && input_ids.dim(1)? > PREFILL_CHUNK {
            return self.forward_chunked(input_ids, ctx);
        }
        let fresh = prefill && ctx.seqlen_offsets().first().copied().unwrap_or(0) == 0;
        let x = self.forward_trunk(input_ids, ctx, fresh)?;
        let x = ctx.logits(&x)?;
        self.output.forward(&x.contiguous()?)
    }

    fn forward_chunked(&self, input_ids: &Tensor, ctx: &ModelForwardContext<'_>) -> Result<Tensor> {
        let (b, total) = input_ids.dims2()?;
        let base = ctx.seqlen_offsets().to_vec();
        let base0 = base.first().copied().unwrap_or(0);
        let position_ids = ctx.position_ids_vec();
        let recurrent_metadata = ctx.recurrent_metadata().cloned();
        let kind = ctx.recurrent_batch_kind().expect("prefill");
        let last_start = (total - 1) / PREFILL_CHUNK * PREFILL_CHUNK;
        let last_lens = ctx
            .context_lens()
            .iter()
            .map(|(start, len)| (start - last_start, *len))
            .collect::<Vec<_>>();
        let end = base.iter().max().copied().unwrap_or(0) + total + 1;
        self.cache.hybrid().reserve_attention(end)?;
        #[cfg(feature = "cuda")]
        mistralrs_quant::TieredExperts::set_prefill_context(end);

        let mut out = None;
        for start in (0..total).step_by(PREFILL_CHUNK) {
            let len = PREFILL_CHUNK.min(total - start);
            let is_last = start + len == total;
            let offsets = base.iter().map(|o| o + start).collect::<Vec<_>>();
            let lens = if is_last { last_lens.clone() } else { vec![(len - 1, 1); b] };
            let mut chunk_ctx = ModelForwardContext::new(&offsets, &lens, &position_ids, None, ctx.flash_params())
                .with_recurrent_batch_kind(kind)
                .with_recurrent_metadata(recurrent_metadata.clone());
            let x = self.forward_trunk(&input_ids.narrow(1, start, len)?, &mut chunk_ctx, start == 0 && base0 == 0)?;
            if is_last {
                out = Some(self.output.forward(&chunk_ctx.logits(&x)?.contiguous()?)?);
            }
        }
        Ok(out.expect("at least one chunk"))
    }

    fn forward_trunk(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>, fresh: bool) -> Result<Tensor> {
        let mut x = self.tok_embeddings.forward(input_ids, &self.device)?.to_dtype(DType::F32)?;
        let md = ctx
            .recurrent_metadata()
            .cloned()
            .ok_or_else(|| candle_core::Error::msg("kimi-linear: recurrent metadata is required"))?;
        let mut hybrid = self.cache.hybrid();
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(mapper) = &self.mapper {
                x = mapper.map(x, i)?;
            }
            let h = layer.attn_norm.forward(&x)?;
            let h = match (&layer.mixer, hybrid.get_mut(i)) {
                (Mixer::Mla(mla), Some(HybridLayerCache::Attention(kv))) => mla.forward(&h, kv, &self.tails, &self.hp)?,
                (Mixer::Kda(kda), Some(HybridLayerCache::Recurrent(pool))) => kda.forward(&h, pool, &md, fresh, &self.hp)?,
                _ => candle_core::bail!("kimi-linear: layer {i} cache kind does not match its mixer"),
            };
            let x_attn = (h + &x)?;
            let h = layer.ffn_norm.forward(&x_attn)?;
            let h = match &layer.ffn {
                Ffn::Dense(mlp) => mlp.forward(&h)?,
                Ffn::Moe(moe) => moe.forward(&h)?,
            };
            x = (h + x_attn)?;
        }
        drop(hybrid);
        self.norm.forward(&x.to_device(&self.device)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Per-token reference of the KDA recurrence with explicit loops.
    fn scan_reference(q: &[f32], k: &[f32], v: &[f32], g: &[f32], beta: &[f32], t: usize, d: usize) -> Vec<f32> {
        let mut s = vec![0f32; d * d];
        let mut out = vec![0f32; t * d];
        for i in 0..t {
            for a in 0..d {
                let decay = g[i * d + a].exp();
                for b in 0..d {
                    s[a * d + b] *= decay;
                }
            }
            let mut pred = vec![0f32; d];
            for a in 0..d {
                for b in 0..d {
                    pred[b] += s[a * d + b] * k[i * d + a];
                }
            }
            for a in 0..d {
                for b in 0..d {
                    s[a * d + b] += k[i * d + a] * (v[i * d + b] - pred[b]) * beta[i];
                }
            }
            for b in 0..d {
                out[i * d + b] = (0..d).map(|a| s[a * d + b] * q[i * d + a]).sum();
            }
        }
        out
    }

    #[test]
    fn delta_scan_matches_loops() -> Result<()> {
        let (t, d) = (5usize, 4usize);
        let dev = Device::Cpu;
        let make = |seed: f32| (0..t * d).map(|i| ((i as f32 + seed) * 0.37).sin() * 0.5).collect::<Vec<_>>();
        let (q, k, v) = (make(1.0), make(2.0), make(3.0));
        let g = make(4.0).iter().map(|x| -x.abs()).collect::<Vec<_>>();
        let beta = (0..t).map(|i| 0.2 + 0.1 * i as f32).collect::<Vec<_>>();
        let tt = |x: &[f32]| Tensor::from_vec(x.to_vec(), (1, t, 1, d), &dev);
        let (o, _) = delta_scan(
            &tt(&q)?,
            &tt(&k)?,
            &tt(&v)?,
            &tt(&g)?,
            &Tensor::from_vec(beta.clone(), (1, t, 1), &dev)?,
            Tensor::zeros((1, 1, d, d), DType::F32, &dev)?,
        )?;
        let got = o.flatten_all()?.to_vec1::<f32>()?;
        let want = scan_reference(&q, &k, &v, &g, &beta, t, d);
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        Ok(())
    }
}
