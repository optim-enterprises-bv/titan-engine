#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::too_many_arguments)]

//! GGUF `inkling2` / `inkling2_loop` (Thinking Machines Inkling 2 Mini), after llama.cpp
//! `src/models/inkling2.cpp` and `inkling2-loop.cpp` (PR #32694 layout, with #32704's per-expert scales).
//!
//! No RoPE: a learned relative-position bias per head (projected from the query input onto a distance
//! table) and a per-query sink logit. Keys, values and the attention/FFN outputs pass causal short
//! convolutions (4 taps, input-dependent taps for K and V on the full-attention layers). Each layer's
//! inputs are a softmax mix ("attention residuals") over the finished blocks of `attn_res.block_size`
//! layers. The loop variant runs a core of layers `loop.start..loop.end` `loop.count` times; passes after
//! the first restart from the prelude output and mix in messages from the previous pass through low-rank
//! controllers. Every visit of a layer has its own KV cache and convolution state ("slot").
//!
//! Routed experts only (sigmoid routing with a selection bias, top-k, renormalized, scaled), fused
//! gate/up, SiTU activation. This first version keeps every weight on the GPU and runs each routed
//! expert as its own quantized matmul, which covers every GGUF expert format candle can multiply.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use candle_core::quantized::gguf_file::Value;
use candle_core::quantized::{QStorage, QTensor};
use candle_core::{DType, Device, Result, Tensor, D};
use mistralrs_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

use super::quantized_qwen35_moe::QEmbedding;
use crate::device_map::DeviceMapper;
use crate::gguf::Content;
use crate::kv_cache::{
    HybridCache, HybridCacheConfig, HybridLayerCache, HybridLayerType, KvCache, RecurrentLayerConfig,
    RecurrentStatePool, RecurrentStateSpec,
};
use crate::layers::QRmsNorm;
use crate::paged_attention::AttentionImplementation;
use crate::pipeline::{EitherCache, ModelForwardContext, RecurrentBatchKind, RecurrentMetadata};
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};

const PREFILL_CHUNK: usize = 512;
/// Largest attention score block (bytes) one query block may build.
const SCORES_BYTES: usize = 64 << 20;
const RECURRENT_SLOTS: usize = 2;
/// llama.cpp `expert_weights_norm_eps` for inkling2.
const ROUTER_NORM_MIN: f32 = 1e-20;
/// Unweighted RMS norms (attention residual mixing, loop controllers).
const MIX_EPS: f64 = 1e-6;
const CTRL_Q: usize = 0;
const CTRL_FFN: usize = 3;

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

struct Md<'a> {
    md: &'a HashMap<String, Value>,
    arch: String,
}

impl Md<'_> {
    fn key(&self, k: &str) -> String {
        format!("{}.{k}", self.arch)
    }
    fn u(&self, k: &str) -> Result<usize> {
        let key = self.key(k);
        match self.md.get(&key) {
            Some(Value::Array(a)) => a
                .first()
                .and_then(value_u64)
                .map(|v| v as usize)
                .ok_or_else(|| candle_core::Error::msg(format!("inkling2: `{key}` is empty"))),
            Some(v) => value_u64(v)
                .map(|v| v as usize)
                .ok_or_else(|| candle_core::Error::msg(format!("inkling2: `{key}` is not an integer"))),
            None => candle_core::bail!("inkling2: missing metadata `{key}`"),
        }
    }
    fn u_or(&self, k: &str, d: usize) -> Result<usize> {
        if self.md.contains_key(&self.key(k)) {
            self.u(k)
        } else {
            Ok(d)
        }
    }
    fn f(&self, k: &str) -> Result<f32> {
        let key = self.key(k);
        match self.md.get(&key) {
            Some(Value::F32(x)) => Ok(*x),
            Some(Value::F64(x)) => Ok(*x as f32),
            _ => candle_core::bail!("inkling2: missing float metadata `{key}`"),
        }
    }
    fn f_or(&self, k: &str, d: f32) -> Result<f32> {
        if self.md.contains_key(&self.key(k)) {
            self.f(k)
        } else {
            Ok(d)
        }
    }
    fn bools(&self, k: &str) -> Result<Vec<bool>> {
        let key = self.key(k);
        match self.md.get(&key) {
            Some(Value::Array(a)) => a
                .iter()
                .map(|v| match v {
                    Value::Bool(b) => Ok(*b),
                    other => value_u64(other)
                        .map(|x| x != 0)
                        .ok_or_else(|| candle_core::Error::msg(format!("inkling2: `{key}` is not a bool array"))),
                })
                .collect(),
            _ => candle_core::bail!("inkling2: missing array metadata `{key}`"),
        }
    }
}

#[derive(Clone, Debug)]
struct LoopHp {
    start: usize,
    end: usize,
    passes: usize,
}

#[derive(Clone, Debug)]
struct Hp {
    n_phys: usize,
    n_embd: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    top_k: usize,
    n_ff_exp: usize,
    rms_eps: f64,
    qk_clip: f64,
    n_rel: usize,
    rel_extent: usize,
    rel_clip: f64,
    log_floor: f64,
    log_alpha: f64,
    n_swa: usize,
    is_swa: Vec<bool>,
    conv_k: usize,
    res_block: usize,
    situ_beta: f64,
    situ_lb: f64,
    expert_scale: f32,
    softcap: f64,
    vocab_unpadded: usize,
    max_seq_len: usize,
    lp: Option<LoopHp>,
}

impl Hp {
    fn from_metadata(md: &HashMap<String, Value>, arch: &str) -> Result<Self> {
        let m = Md { md, arch: arch.to_string() };
        let n_phys = m.u("block_count")?;
        let is_swa = m.bools("attention.sliding_window_pattern")?;
        if is_swa.len() != n_phys {
            candle_core::bail!("inkling2: sliding_window_pattern has {} entries for {n_phys} layers", is_swa.len());
        }
        let lp = if arch == "inkling2_loop" {
            if matches!(md.get(&m.key("loop.varying")), Some(Value::Bool(true))) {
                candle_core::bail!("inkling2_loop: variable loop counts are not supported");
            }
            Some(LoopHp { start: m.u("loop.start")?, end: m.u("loop.end")?, passes: m.u("loop.count")? })
        } else {
            None
        };
        let head_dim = m.u("attention.key_length")?;
        if m.u("attention.value_length")? != head_dim {
            candle_core::bail!("inkling2: key and value head sizes differ");
        }
        Ok(Self {
            n_phys,
            n_embd: m.u("embedding_length")?,
            n_head: m.u("attention.head_count")?,
            n_head_kv: m.u("attention.head_count_kv")?,
            head_dim,
            top_k: m.u("expert_used_count")?,
            n_ff_exp: m.u("expert_feed_forward_length")?,
            rms_eps: f64::from(m.f("attention.layer_norm_rms_epsilon")?),
            qk_clip: f64::from(m.f("attention.qk_clip")?),
            n_rel: m.u("attention.relative_length")?,
            rel_extent: m.u("attention.relative_extent")?,
            rel_clip: f64::from(m.f("attention.relative_clip")?),
            log_floor: m.u("attention.log_scale_floor")? as f64,
            log_alpha: f64::from(m.f("attention.log_scale_alpha")?),
            n_swa: m.u("attention.sliding_window")?,
            is_swa,
            conv_k: m.u("shortconv.l_cache")?,
            res_block: m.u("attn_res.block_size")?,
            situ_beta: f64::from(m.f("activation.situ_beta")?),
            situ_lb: f64::from(m.f("activation.situ_linear_beta")?),
            expert_scale: m.f_or("expert_weights_scale", 1.0)?,
            softcap: f64::from(m.f_or("final_logit_softcapping", 0.0)?),
            vocab_unpadded: m.u("vocab_size_unpadded")?,
            max_seq_len: m.u("context_length")?,
            lp,
        })
    }

    fn k_gqa(&self) -> usize {
        self.n_head_kv * self.head_dim
    }

    /// Convolution state channels per slot: K, V, the attention output and the FFN output.
    fn conv_channels(&self) -> usize {
        2 * self.k_gqa() + 2 * self.n_embd
    }

    fn n_slots(&self) -> usize {
        match &self.lp {
            Some(l) => self.n_phys + (l.passes - 1) * (l.end - l.start),
            None => self.n_phys,
        }
    }

    fn phys(&self, slot: usize) -> usize {
        match &self.lp {
            Some(l) if slot >= self.n_phys => l.start + (slot - self.n_phys) % (l.end - l.start),
            _ => slot,
        }
    }
}

fn linear(q: QTensor) -> Result<Arc<dyn QuantMethod>> {
    Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf { q_weight: Arc::new(q), b: None })?))
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

fn fwd(l: &Arc<dyn QuantMethod>, x: &Tensor) -> Result<Tensor> {
    l.forward(x)?.to_dtype(DType::F32)
}

/// `x / sqrt(mean(x^2) + eps)` over the last dimension, no weight.
fn rms(x: &Tensor, eps: f64) -> Result<Tensor> {
    let ms = x.sqr()?.mean_keepdim(D::Minus1)?;
    x.broadcast_div(&(ms + eps)?.sqrt()?)
}

/// llama.cpp `graph::clip`: optional per-channel scale, then each row scaled down to an L2 norm of at
/// most `thr`.
fn clip(x: &Tensor, scale: Option<&Tensor>, thr: f64) -> Result<Tensor> {
    let x = match scale {
        Some(s) => x.broadcast_mul(s)?,
        None => x.clone(),
    };
    let norm = (x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()? + 1e-6)?;
    let factor = norm.recip()?.affine(thr, 0.0)?.clamp(0f32, 1f32)?;
    x.broadcast_mul(&factor)
}

/// Attention over depth: `blocks` and `cur` are `(n, t, E)`; `query` is `(q, E)`. Returns `(n, t, q, E)`,
/// each query's softmax-weighted sum of the RMS-normalized states.
fn mix(blocks: &[Tensor], cur: &Tensor, query: &Tensor) -> Result<Tensor> {
    let mut all: Vec<&Tensor> = blocks.iter().collect();
    all.push(cur);
    let states = rms(&Tensor::stack(&all, 2)?, MIX_EPS)?; // (n, t, nb, E)
    let e = states.dim(D::Minus1)?;
    let scores = (states.broadcast_matmul(&query.t()?)? / (e as f64).sqrt())?; // (n, t, nb, q)
    let probs = candle_nn::ops::softmax_last_dim(&scores.transpose(2, 3)?.contiguous()?)?; // (n, t, q, nb)
    probs.matmul(&states)
}

fn take(x: &Tensor, i: usize) -> Result<Tensor> {
    x.narrow(2, i, 1)?.squeeze(2)?.contiguous()
}

struct Conv {
    /// `(channels, taps)`
    w: Tensor,
}

impl Conv {
    /// Causal short convolution with a residual. `cur` `(n, t, C)`, `hist` `(n, C, taps - 1)` (the previous
    /// inputs), `dynamic` `(n, t, heads * taps)` adds input-dependent taps per head. Returns the output and
    /// the next history.
    fn forward(&self, cur: &Tensor, hist: &Tensor, dynamic: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let (n, t, c) = cur.dims3()?;
        let taps = self.w.dim(1)?;
        let x = Tensor::cat(&[hist, &cur.transpose(1, 2)?.contiguous()?], 2)?; // (n, C, taps - 1 + t)
        let mut acc: Option<Tensor> = None;
        for k in 0..taps {
            let term = x.narrow(2, k, t)?.broadcast_mul(&self.w.narrow(1, k, 1)?.unsqueeze(0)?)?;
            acc = Some(match acc {
                Some(a) => (a + term)?,
                None => term,
            });
        }
        let mut out = acc.expect("taps > 0").transpose(1, 2)?.contiguous()?; // (n, t, C)
        if let Some(dynamic) = dynamic {
            let heads = dynamic.dim(D::Minus1)? / taps;
            let d = dynamic.reshape((n, t, heads, taps))?;
            for lag in 0..taps {
                let tap = d.narrow(3, lag, 1)?; // (n, t, heads, 1)
                let window = x.narrow(2, taps - 1 - lag, t)?.transpose(1, 2)?.contiguous()?.reshape((n, t, heads, c / heads))?;
                out = (out + window.broadcast_mul(&tap)?.reshape((n, t, c))?)?;
            }
        }
        let denom = (((if dynamic.is_some() { 2 } else { 1 }) * taps + 1) as f64).sqrt();
        let y = ((cur + out)? / denom)?;
        let next = x.narrow(2, t, taps - 1)?.contiguous()?;
        Ok((y, next))
    }
}

struct Attn {
    wq: Arc<dyn QuantMethod>,
    wk: Arc<dyn QuantMethod>,
    wv: Arc<dyn QuantMethod>,
    wo: Arc<dyn QuantMethod>,
    wgate: Arc<dyn QuantMethod>,
    /// `(n_head, head_dim)`
    q_clip: Tensor,
    sink: Arc<dyn QuantMethod>,
    rel: Arc<dyn QuantMethod>,
    /// `(n_rel, extent + 1)`: the distance table, one zero column for distances past the extent.
    rel_bias_t: Tensor,
    /// `(n_head, n_rel)`
    rel_clip: Tensor,
    k_conv: Conv,
    v_conv: Conv,
    post_conv: Conv,
    /// SWA layers: one input norm; full layers: separate Q, K, V norms and dynamic K/V taps.
    norm: Option<QRmsNorm>,
    qkv_norms: Option<[QRmsNorm; 3]>,
    k_conv_proj: Option<Arc<dyn QuantMethod>>,
    v_conv_proj: Option<Arc<dyn QuantMethod>>,
}

struct Experts {
    /// per expert: fused gate/up `(2 * n_ff, E)` and down `(E, n_ff)`
    gate_up: Vec<Arc<dyn QuantMethod>>,
    down: Vec<Arc<dyn QuantMethod>>,
    gate_up_scale: Option<Vec<f32>>,
    down_scale: Option<Vec<f32>>,
}

struct Moe {
    /// `(E, n_expert)`
    router_t: Tensor,
    select_bias: Option<Tensor>,
    experts: Experts,
}

struct LoopLayer {
    /// `(4, E)`: Q, K, V and FFN message queries
    msg_query: Tensor,
    /// per controller: down `(2E, rank)`, gain `(rank, E)`, forget `(rank, E)`
    ctrl: Vec<(Tensor, Tensor, Tensor)>,
}

struct Layer {
    attn: Attn,
    /// `(q, E)`: 1 query on SWA layers, 3 (Q, K, V) on full layers; absent on layer 0.
    attn_res_query: Option<Tensor>,
    /// `(1, E)`
    ffn_res_query: Tensor,
    ffn_norm: QRmsNorm,
    moe: Moe,
    ffn_post_conv: Conv,
}

pub struct ModelWeights {
    tok_embeddings: QEmbedding,
    tok_norm: QRmsNorm,
    layers: Vec<Layer>,
    loop_layers: Vec<LoopLayer>,
    output_res_query: Tensor,
    output_norm: QRmsNorm,
    output: Arc<dyn QuantMethod>,
    pub(crate) device: Device,
    pub(crate) cache: EitherCache,
    pub(crate) max_seq_len: usize,
    hp: Hp,
}

fn split_rows(q: &QTensor, rows: usize, cols: usize, device: &Device) -> Result<Vec<QTensor>> {
    let (experts, r, c) = q.shape().dims3()?;
    if r != rows || c != cols {
        candle_core::bail!("inkling2: expert tensor {:?}, want (_, {rows}, {cols})", q.shape());
    }
    let dtype = q.dtype();
    let row_bytes = cols / dtype.block_size() * dtype.type_size();
    let stride = rows * row_bytes;
    let data = q.data()?;
    if data.len() < experts * stride {
        candle_core::bail!("inkling2: expert tensor has {} bytes, want {}", data.len(), experts * stride);
    }
    (0..experts)
        .map(|e| {
            let bytes = &data[e * stride..(e + 1) * stride];
            QTensor::new(QStorage::from_data(Cow::Borrowed(bytes), device, dtype)?, (rows, cols))
        })
        .collect()
}

fn load_scale<R: std::io::Seek + std::io::Read>(ct: &mut Content<'_, R>, name: &str) -> Result<Option<Vec<f32>>> {
    if !ct.has_tensor(name) {
        return Ok(None);
    }
    Ok(Some(load_f32(ct, name, &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?))
}

impl Experts {
    fn load<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        prefix: &str,
        hp: &Hp,
        device: &Device,
    ) -> Result<Self> {
        let gu = ct.tensor(&format!("{prefix}.ffn_gate_up_exps.weight"), &Device::Cpu)?;
        let down = ct.tensor(&format!("{prefix}.ffn_down_exps.weight"), &Device::Cpu)?;
        let gate_up = split_rows(&gu, 2 * hp.n_ff_exp, hp.n_embd, device)?
            .into_iter()
            .map(linear)
            .collect::<Result<Vec<_>>>()?;
        let down = split_rows(&down, hp.n_embd, hp.n_ff_exp, device)?
            .into_iter()
            .map(linear)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            gate_up,
            down,
            gate_up_scale: load_scale(ct, &format!("{prefix}.ffn_gate_up_exps.scale"))?,
            down_scale: load_scale(ct, &format!("{prefix}.ffn_down_exps.scale"))?,
        })
    }
}

impl Moe {
    /// `x` `(n, t, E)` F32.
    fn forward(&self, x: &Tensor, hp: &Hp) -> Result<Tensor> {
        let (n, t, e) = x.dims3()?;
        let flat = x.reshape((n * t, e))?.contiguous()?;
        let tokens = n * t;
        let topk = crate::ops::moe_router_topk(
            &flat.matmul(&self.router_t)?,
            crate::ops::MoeRouterTopKConfig {
                top_k: hp.top_k,
                score_function: crate::ops::MoeRouterScoreFunction::Sigmoid,
                selected_weight: crate::ops::MoeRouterSelectedWeight::Score,
                renormalize: true,
                norm_min: ROUTER_NORM_MIN,
                output_scale: hp.expert_scale,
                logit_clip: None,
            },
            self.select_bias.as_ref(),
            None,
        )?;
        let ids = topk.indices.to_device(&Device::Cpu)?.to_vec2::<u32>()?;
        let weights = topk.values.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
        let mut per_expert: HashMap<usize, (Vec<u32>, Vec<f32>)> = HashMap::new();
        for (tok, (row, wrow)) in ids.iter().zip(&weights).enumerate() {
            for (&id, &w) in row.iter().zip(wrow) {
                let entry = per_expert.entry(id as usize).or_default();
                entry.0.push(tok as u32);
                entry.1.push(w);
            }
        }
        let ff = hp.n_ff_exp;
        let mut out = Tensor::zeros((tokens, e), DType::F32, x.device())?;
        let mut keys: Vec<usize> = per_expert.keys().copied().collect();
        keys.sort_unstable();
        for id in keys {
            let (toks, ws) = &per_expert[&id];
            let idx = Tensor::from_slice(toks, toks.len(), x.device())?;
            let xe = flat.index_select(&idx, 0)?;
            let mut gu = fwd(&self.experts.gate_up[id], &xe)?;
            if let Some(s) = &self.experts.gate_up_scale {
                gu = (gu * f64::from(s[id]))?;
            }
            let gate = gu.narrow(1, 0, ff)?;
            let up = gu.narrow(1, ff, ff)?;
            let act = situ(&gate, &up, hp.situ_beta, hp.situ_lb)?;
            let mut y = fwd(&self.experts.down[id], &act.contiguous()?)?;
            let mut scale = Tensor::from_slice(ws, (ws.len(), 1), x.device())?;
            if let Some(s) = &self.experts.down_scale {
                scale = (scale * f64::from(s[id]))?;
            }
            y = y.broadcast_mul(&scale)?;
            out = out.index_add(&idx, &y, 0)?;
        }
        out.reshape((n, t, e))
    }
}

/// `situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb)` (no linear clamp when lb == 0).
fn situ(gate: &Tensor, up: &Tensor, beta: f64, lb: f64) -> Result<Tensor> {
    let act = ((gate / beta)?.tanh()? * beta)?.mul(&candle_nn::ops::sigmoid(gate)?)?;
    let up = if lb > 0.0 { ((up / lb)?.tanh()? * lb)? } else { up.clone() };
    act.mul(&up)
}

impl Attn {
    fn load<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        prefix: &str,
        hp: &Hp,
        swa: bool,
        device: &Device,
    ) -> Result<Self> {
        let p = |s: &str| format!("{prefix}.{s}");
        let conv = |ct: &mut Content<'_, R>, s: &str| -> Result<Conv> {
            // GGUF {taps, channels} -> (channels, taps)
            Ok(Conv { w: load_f32(ct, &p(s), device)? })
        };
        let rel_bias = load_f32(ct, &p("attn_rel_bias.weight"), device)?; // (extent, n_rel)
        let rel_bias_t = Tensor::cat(&[&rel_bias.t()?, &Tensor::zeros((hp.n_rel, 1), DType::F32, device)?], 1)?
            .contiguous()?;
        let (norm, qkv_norms, k_conv_proj, v_conv_proj) = if swa {
            (Some(QRmsNorm::new(ct.tensor(&p("attn_norm.weight"), device)?, hp.rms_eps as f32)?), None, None, None)
        } else {
            let n = |ct: &mut Content<'_, R>, s: &str| QRmsNorm::new(ct.tensor(&p(s), device)?, hp.rms_eps as f32);
            (
                None,
                Some([n(ct, "attn_q_inp_norm.weight")?, n(ct, "attn_k_inp_norm.weight")?, n(ct, "attn_v_inp_norm.weight")?]),
                Some(load_linear(ct, &p("attn_k_conv_proj.weight"), device)?),
                Some(load_linear(ct, &p("attn_v_conv_proj.weight"), device)?),
            )
        };
        Ok(Self {
            wq: load_linear(ct, &p("attn_q.weight"), device)?,
            wk: load_linear(ct, &p("attn_k.weight"), device)?,
            wv: load_linear(ct, &p("attn_v.weight"), device)?,
            wo: load_linear(ct, &p("attn_output.weight"), device)?,
            wgate: load_linear(ct, &p("attn_gate.weight"), device)?,
            q_clip: load_f32(ct, &p("attn_q_clip.weight"), device)?,
            sink: load_linear(ct, &p("attn_sink_proj.weight"), device)?,
            rel: load_linear(ct, &p("attn_rel.weight"), device)?,
            rel_bias_t,
            rel_clip: load_f32(ct, &p("attn_rel_clip.weight"), device)?,
            k_conv: conv(ct, "attn_k_conv.weight")?,
            v_conv: conv(ct, "attn_v_conv.weight")?,
            post_conv: conv(ct, "attn_post_conv.weight")?,
            norm,
            qkv_norms,
            k_conv_proj,
            v_conv_proj,
        })
    }
}

/// Per forward: the absolute positions of the batch rows and the log-scale factors of the full layers.
struct Positions {
    /// first position of each sequence in this forward
    starts: Vec<usize>,
    t: usize,
    /// `(n, t, 1, 1)`
    tau: Tensor,
}

impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>,
        device: &Device,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        _attention_mechanism: AttentionImplementation,
        _dtype: DType,
    ) -> Result<Self> {
        let arch = match ct.get_metadata().get("general.architecture") {
            Some(Value::String(s)) => s.clone(),
            _ => candle_core::bail!("inkling2: missing general.architecture"),
        };
        let hp = Hp::from_metadata(ct.get_metadata(), &arch)?;
        // every layer and slot on the main device in this version
        if (0..hp.n_phys).any(|i| mapper.device_for(i, false).is_some_and(|d| !d.same_device(device))) {
            candle_core::bail!("inkling2: every layer must be on the main device (no CPU offload in this version)");
        }
        let tok_embeddings = QEmbedding::new(ct.tensor("token_embd.weight", &Device::Cpu)?)?;
        let tok_norm = QRmsNorm::new(ct.tensor("token_embd_norm.weight", device)?, hp.rms_eps as f32)?;
        let output_norm = QRmsNorm::new(ct.tensor("output_norm.weight", device)?, hp.rms_eps as f32)?;
        let output = load_linear(&mut ct, "output.weight", device)?;
        let output_res_query = load_f32(&mut ct, "output_res_query.weight", device)?.reshape((1, hp.n_embd))?;

        let mut layers = Vec::with_capacity(hp.n_phys);
        for i in NiceProgressBar::<_, 'b'>(0..hp.n_phys, "Loading repeating layers", &new_multi_progress()) {
            let prefix = format!("blk.{i}");
            let swa = hp.is_swa[i];
            let attn = Attn::load(&mut ct, &prefix, &hp, swa, device)?;
            let attn_res_query = if i > 0 {
                let q = load_f32(&mut ct, &format!("{prefix}.attn_res_query.weight"), device)?;
                Some(q.reshape(((), hp.n_embd))?)
            } else {
                None
            };
            let router = load_f32(&mut ct, &format!("{prefix}.ffn_gate_inp.weight"), device)?; // (n_expert, E)
            let bias_name = format!("{prefix}.exp_probs_b.bias");
            let select_bias = if ct.has_tensor(&bias_name) { Some(load_f32(&mut ct, &bias_name, device)?) } else { None };
            layers.push(Layer {
                attn,
                attn_res_query,
                ffn_res_query: load_f32(&mut ct, &format!("{prefix}.ffn_res_query.weight"), device)?.reshape((1, hp.n_embd))?,
                ffn_norm: QRmsNorm::new(ct.tensor(&format!("{prefix}.ffn_norm.weight"), device)?, hp.rms_eps as f32)?,
                moe: Moe {
                    router_t: router.t()?.contiguous()?,
                    select_bias,
                    experts: Experts::load(&mut ct, &prefix, &hp, device)?,
                },
                ffn_post_conv: Conv { w: load_f32(&mut ct, &format!("{prefix}.ffn_post_conv.weight"), device)? },
            });
        }

        let mut loop_layers = Vec::new();
        if let Some(lp) = &hp.lp {
            for i in lp.start..lp.end {
                let p = |s: &str| format!("blk.{i}.{s}");
                let mut ctrl = Vec::with_capacity(4);
                for c in ["q", "k", "v", "ffn"] {
                    let down = load_f32(&mut ct, &p(&format!("loop_{c}_ctrl_down.weight")), device)?; // (rank, 2E)
                    let gain = load_f32(&mut ct, &p(&format!("loop_{c}_ctrl_gain.weight")), device)?; // (E, rank)
                    let forget = load_f32(&mut ct, &p(&format!("loop_{c}_ctrl_forget.weight")), device)?;
                    ctrl.push((down.t()?.contiguous()?, gain.t()?.contiguous()?, forget.t()?.contiguous()?));
                }
                loop_layers.push(LoopLayer {
                    msg_query: load_f32(&mut ct, &p("loop_msg_query.weight"), device)?.reshape((4, hp.n_embd))?,
                    ctrl,
                });
            }
        }

        let n_slots = hp.n_slots();
        let mut layer_types = vec![HybridLayerType::Attention; n_slots];
        layer_types.extend(std::iter::repeat_n(HybridLayerType::Recurrent, n_slots));
        let mut cache = HybridCache::with_slots(
            HybridCacheConfig {
                layer_types,
                max_seq_len: hp.max_seq_len,
                recurrent: RecurrentLayerConfig {
                    conv_dim: hp.conv_channels(),
                    conv_width: hp.conv_k - 1,
                    state: RecurrentStateSpec::Opaque { dims: vec![1] },
                    recurrent_dtype: Some(DType::F32),
                },
            },
            DType::F32,
            &vec![device.clone(); 2 * n_slots],
            RECURRENT_SLOTS,
        )?;
        let n_swa_slots = (0..n_slots).filter(|&s| hp.is_swa[hp.phys(s)]).count();
        for slot in 0..n_slots {
            if hp.is_swa[hp.phys(slot)] {
                if let Some(c) = cache.get_mut(slot) {
                    *c = HybridLayerCache::Attention(KvCache::new_rotating(2, hp.n_swa, 512));
                }
            }
        }
        tracing::info!(
            "inkling2: {} physical layers, {n_slots} layer visits ({n_swa_slots} sliding-window of {}, {} full)",
            hp.n_phys,
            hp.n_swa,
            n_slots - n_swa_slots
        );

        Ok(Self {
            tok_embeddings,
            tok_norm,
            layers,
            loop_layers,
            output_res_query,
            output_norm,
            output,
            device: device.clone(),
            cache: EitherCache::Hybrid(Arc::new(Mutex::new(cache))),
            max_seq_len: hp.max_seq_len,
            hp,
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
        let starts = ctx.seqlen_offsets().to_vec();
        let x = self.forward_trunk(input_ids, &starts, ctx, fresh)?;
        let x = ctx.logits(&x)?;
        self.head(&x)
    }

    fn head(&self, x: &Tensor) -> Result<Tensor> {
        let mut logits = fwd(&self.output, &x.contiguous()?)?;
        if self.hp.softcap > 0.0 {
            logits = ((logits / self.hp.softcap)?.tanh()? * self.hp.softcap)?;
        }
        logits.narrow(D::Minus1, 0, self.hp.vocab_unpadded)?.contiguous()
    }

    fn forward_chunked(&self, input_ids: &Tensor, ctx: &ModelForwardContext<'_>) -> Result<Tensor> {
        let (b, total) = input_ids.dims2()?;
        let base = ctx.seqlen_offsets().to_vec();
        let base0 = base.first().copied().unwrap_or(0);
        let position_ids = ctx.position_ids_vec();
        let recurrent_metadata = ctx.recurrent_metadata().cloned();
        let kind = ctx.recurrent_batch_kind().expect("prefill");
        let last_start = (total - 1) / PREFILL_CHUNK * PREFILL_CHUNK;
        let last_lens = ctx.context_lens().iter().map(|(start, len)| (start - last_start, *len)).collect::<Vec<_>>();
        let end = base.iter().max().copied().unwrap_or(0) + total + 1;
        self.cache.hybrid().reserve_attention(end)?;

        let mut out = None;
        for start in (0..total).step_by(PREFILL_CHUNK) {
            let len = PREFILL_CHUNK.min(total - start);
            let is_last = start + len == total;
            let offsets = base.iter().map(|o| o + start).collect::<Vec<_>>();
            let lens = if is_last { last_lens.clone() } else { vec![(len - 1, 1); b] };
            let mut chunk_ctx = ModelForwardContext::new(&offsets, &lens, &position_ids, None, ctx.flash_params())
                .with_recurrent_batch_kind(kind)
                .with_recurrent_metadata(recurrent_metadata.clone());
            let x = self.forward_trunk(&input_ids.narrow(1, start, len)?, &offsets, &mut chunk_ctx, start == 0 && base0 == 0)?;
            if is_last {
                out = Some(self.head(&chunk_ctx.logits(&x)?)?);
            }
        }
        Ok(out.expect("at least one chunk"))
    }

    fn positions(&self, starts: &[usize], t: usize) -> Result<Positions> {
        let hp = &self.hp;
        let mut tau = Vec::with_capacity(starts.len() * t);
        for &s in starts {
            for i in 0..t {
                let pos = (s + i) as f64;
                tau.push((1.0 + hp.log_alpha * ((pos + 1.0) / hp.log_floor).max(1.0).ln()) as f32);
            }
        }
        Ok(Positions {
            starts: starts.to_vec(),
            t,
            tau: Tensor::from_vec(tau, (starts.len(), t, 1, 1), &self.device)?,
        })
    }

    fn forward_trunk(
        &self,
        input_ids: &Tensor,
        starts: &[usize],
        ctx: &mut ModelForwardContext<'_>,
        fresh: bool,
    ) -> Result<Tensor> {
        let hp = &self.hp;
        let t = input_ids.dim(1)?;
        let pos = self.positions(starts, t)?;
        let md = ctx
            .recurrent_metadata()
            .cloned()
            .ok_or_else(|| candle_core::Error::msg("inkling2: recurrent metadata is required"))?;
        let mut x = self.tok_embeddings.forward(input_ids, &self.device)?.to_dtype(DType::F32)?;
        x = self.tok_norm.forward(&x)?;
        let mut hybrid = self.cache.hybrid();
        let n_slots = hp.n_slots();
        let mut blocks: Vec<Tensor> = Vec::new();

        let mut run = |slot: usize, ip: usize, cur: Tensor, blocks: &mut Vec<Tensor>, msgs: Option<(&Tensor, &LoopLayer)>| -> Result<Tensor> {
            let (att, rec) = hybrid.caches.split_at_mut(n_slots);
            let (HybridLayerCache::Attention(kv), HybridLayerCache::Recurrent(pool)) = (&mut att[slot], &mut rec[slot]) else {
                candle_core::bail!("inkling2: slot {slot} caches are not an attention/recurrent pair");
            };
            self.layer(slot, ip, cur, blocks, msgs, kv, pool, &md, fresh, &pos)
        };

        match &hp.lp {
            None => {
                for il in 0..hp.n_phys {
                    x = run(il, il, x, &mut blocks, None)?;
                }
            }
            Some(lp) => {
                for il in 0..lp.start {
                    x = run(il, il, x, &mut blocks, None)?;
                }
                let prelude = x.clone();
                let prelude_blocks = blocks.clone();
                for pass in 0..lp.passes {
                    let mut messages = None;
                    if pass > 0 {
                        let bank: Vec<Tensor> = blocks[prelude_blocks.len() + 1..].to_vec();
                        let bank_out = x.clone();
                        x = prelude.clone();
                        blocks = prelude_blocks.clone();
                        messages = Some((bank, bank_out));
                    }
                    for ip in lp.start..lp.end {
                        let slot = if pass == 0 { ip } else { hp.n_phys + (pass - 1) * (lp.end - lp.start) + (ip - lp.start) };
                        x = match &messages {
                            None => run(slot, ip, x, &mut blocks, None)?,
                            Some((bank, bank_out)) => {
                                let ll = &self.loop_layers[ip - lp.start];
                                let msgs = mix(bank, bank_out, &ll.msg_query)?;
                                run(slot, ip, x, &mut blocks, Some((&msgs, ll)))?
                            }
                        };
                    }
                }
                for il in lp.end..hp.n_phys {
                    x = run(il, il, x, &mut blocks, None)?;
                }
            }
        }
        drop(hybrid);
        let out = take(&mix(&blocks, &x, &self.output_res_query)?, 0)?;
        self.output_norm.forward(&out)
    }

    /// One layer visit. `ip` is the physical layer index used for the block bookkeeping.
    fn layer(
        &self,
        slot: usize,
        ip: usize,
        cur: Tensor,
        blocks: &mut Vec<Tensor>,
        msgs: Option<(&Tensor, &LoopLayer)>,
        kv: &mut KvCache,
        pool: &mut RecurrentStatePool,
        md: &RecurrentMetadata,
        fresh: bool,
        pos: &Positions,
    ) -> Result<Tensor> {
        let hp = &self.hp;
        let il = hp.phys(slot);
        let layer = &self.layers[il];
        let a = &layer.attn;
        let swa = hp.is_swa[il];
        let (n, t, e) = cur.dims3()?;
        let (h, hkv, d) = (hp.n_head, hp.n_head_kv, hp.head_dim);
        let kg = hp.k_gqa();

        // convolution history of this slot: K | V | attention output | FFN output
        let indices = if md.state_indices().device().same_device(pool.device()) {
            md.state_indices().clone()
        } else {
            md.state_indices().to_device(pool.device())?
        };
        let hist = if fresh {
            Tensor::zeros((n, hp.conv_channels(), hp.conv_k - 1), DType::F32, cur.device())?
        } else {
            pool.gather_conv_state(&indices)?.to_dtype(DType::F32)?
        };
        let hist_part = |off: usize, len: usize| hist.narrow(1, off, len);

        let inp_sa = cur.clone();
        let mut inputs = [cur.clone(), cur.clone(), cur.clone()];
        if let Some(q) = &layer.attn_res_query {
            let mixed = mix(blocks, &cur, q)?; // (n, t, 1 or 3, E)
            for (j, inp) in inputs.iter_mut().enumerate() {
                *inp = take(&mixed, if swa { 0 } else { j })?;
            }
        }
        if let Some((m, ll)) = msgs {
            for (j, inp) in inputs.iter_mut().enumerate() {
                *inp = control(inp, &take(m, CTRL_Q + j)?, &ll.ctrl[CTRL_Q + j])?;
            }
        }
        match (&a.norm, &a.qkv_norms) {
            (Some(norm), _) if msgs.is_some() => {
                for inp in inputs.iter_mut() {
                    *inp = norm.forward(inp)?;
                }
            }
            (Some(norm), _) => {
                let x = norm.forward(&inputs[0])?;
                inputs = [x.clone(), x.clone(), x];
            }
            (None, Some(norms)) => {
                for (inp, nm) in inputs.iter_mut().zip(norms) {
                    *inp = nm.forward(inp)?;
                }
            }
            _ => candle_core::bail!("inkling2: layer {il} has no attention input norm"),
        }

        // projections and convolutions
        let q = fwd(&a.wq, &inputs[0])?;
        let k_dyn = match &a.k_conv_proj {
            Some(p) => Some(fwd(p, &inputs[1])?),
            None => None,
        };
        let v_dyn = match &a.v_conv_proj {
            Some(p) => Some(fwd(p, &inputs[2])?),
            None => None,
        };
        let (k, hk) = a.k_conv.forward(&fwd(&a.wk, &inputs[1])?, &hist_part(0, kg)?, k_dyn.as_ref())?;
        let (v, hv) = a.v_conv.forward(&fwd(&a.wv, &inputs[2])?, &hist_part(kg, kg)?, v_dyn.as_ref())?;
        let r = fwd(&a.rel, &inputs[0])?;

        let mut q = clip(&q.reshape((n, t, h, d))?, Some(&a.q_clip), hp.qk_clip)?;
        let k = clip(&k.reshape((n, t, hkv, d))?, None, hp.qk_clip)?;
        let v = rms(&v.reshape((n, t, hkv, d))?, hp.rms_eps)?;
        let mut r = clip(&r.reshape((n, t, h, hp.n_rel))?, Some(&a.rel_clip), hp.rel_clip)?;
        if !swa {
            q = q.broadcast_mul(&pos.tau)?;
            r = r.broadcast_mul(&pos.tau)?;
        }
        let sinks = fwd(&a.sink, &inputs[0])?; // (n, t, H)
        // distance table per (token, head): R . bias_table - sink; the last column (past the extent) is -sink
        let table = r.broadcast_matmul(&a.rel_bias_t)?.broadcast_sub(&sinks.unsqueeze(3)?)?; // (n, t, H, extent+1)

        let (k_all, v_all) = kv.append(
            &k.transpose(1, 2)?.contiguous()?.to_dtype(DType::BF16)?,
            &v.transpose(1, 2)?.contiguous()?.to_dtype(DType::BF16)?,
        )?;
        let attn = self.attend(&q, &k_all, &v_all, &table, swa, pos)?; // (n, t, H*D)

        let gate = candle_nn::ops::sigmoid(&fwd(&a.wgate, &inputs[0])?)?;
        let o = fwd(&a.wo, &attn.mul(&gate)?)?;
        let (o, ho) = a.post_conv.forward(&o, &hist_part(2 * kg, e)?, None)?;

        let mut cur = if ip % hp.res_block == 0 {
            blocks.push(inp_sa);
            o
        } else {
            (inp_sa + o)?
        };

        // MoE
        let ffn_inp = cur.clone();
        let mut f = take(&mix(blocks, &cur, &layer.ffn_res_query)?, 0)?;
        if let Some((m, ll)) = msgs {
            f = control(&f, &take(m, CTRL_FFN)?, &ll.ctrl[CTRL_FFN])?;
        }
        let f = layer.ffn_norm.forward(&f)?;
        let f = layer.moe.forward(&f, hp)?;
        let (f, hf) = layer.ffn_post_conv.forward(&f, &hist_part(2 * kg + e, e)?, None)?;
        cur = (ffn_inp + f)?;

        let new_hist = Tensor::cat(&[hk, hv, ho, hf], 1)?.contiguous()?;
        pool.scatter_conv_state_with_host_indices(&indices, md.state_indices_host(), &new_hist.to_dtype(pool.conv_dtype())?)?;
        Ok(cur)
    }

    /// Attention with the relative-position bias and a zero sink. `q` `(n, t, H, D)`, `k_all`/`v_all`
    /// `(n, Hkv, L, D)` (the most recent L positions, ending at this chunk's last token), `table`
    /// `(n, t, H, extent + 1)`. Returns `(n, t, H*D)`.
    fn attend(&self, q: &Tensor, k_all: &Tensor, v_all: &Tensor, table: &Tensor, swa: bool, pos: &Positions) -> Result<Tensor> {
        let hp = &self.hp;
        let (n, t, h, d) = q.dims4()?;
        let hkv = hp.n_head_kv;
        let g = h / hkv;
        let l = k_all.dim(2)?;
        let ext = hp.rel_extent;
        let scale = 1.0 / (d as f64).sqrt();
        let dev = q.device();
        let mut rows = Vec::with_capacity(n);
        for b in 0..n {
            let kb = k_all.get(b)?.to_dtype(DType::F32)?; // (Hkv, L, D)
            let vb = v_all.get(b)?.to_dtype(DType::F32)?;
            let qb = q.get(b)?.transpose(0, 1)?.contiguous()?; // (H, t, D)
            let tb = table.get(b)?.transpose(0, 1)?.contiguous()?; // (H, t, ext+1)
            let end = pos.starts[b] + pos.t; // one past the last position in the cache
            let first_key = end - l;
            let block = (SCORES_BYTES / (4 * h * l)).clamp(1, t);
            let mut outs = Vec::with_capacity(t.div_ceil(block));
            for i0 in (0..t).step_by(block) {
                let lq = block.min(t - i0);
                // (Hkv, g*lq, D) . (Hkv, D, L) -> (H, lq, L)
                let qg = qb.narrow(1, i0, lq)?.reshape((hkv, g * lq, d))?;
                let scores = (qg.matmul(&kb.t()?)? * scale)?.reshape((h, lq, l))?;
                let mut idx = Vec::with_capacity(lq * l);
                let mut mask = Vec::with_capacity(lq * l);
                for i in 0..lq {
                    let p = pos.starts[b] + i0 + i;
                    for j in 0..l {
                        let kp = first_key + j;
                        let valid = kp <= p && (!swa || p - kp < hp.n_swa);
                        let dist = p.saturating_sub(kp);
                        idx.push(if valid && dist < ext { dist as u32 } else { ext as u32 });
                        mask.push(if valid { 0f32 } else { f32::NEG_INFINITY });
                    }
                }
                let idx = Tensor::from_vec(idx, (1, lq, l), dev)?.broadcast_as((h, lq, l))?.contiguous()?;
                let mask = Tensor::from_vec(mask, (1, lq, l), dev)?;
                let bias = tb.narrow(1, i0, lq)?.contiguous()?.gather(&idx, 2)?; // (H, lq, L)
                let logits = (scores + bias)?.broadcast_add(&mask)?;
                // softmax over the keys and one zero-logit sink
                let m = logits.max_keepdim(2)?.clamp(0f32, f32::INFINITY)?;
                let ex = logits.broadcast_sub(&m)?.exp()?;
                let denom = (ex.sum_keepdim(2)? + m.neg()?.exp()?)?;
                let p = ex.broadcast_div(&denom)?.reshape((hkv, g * lq, l))?;
                let o = p.matmul(&vb)?.reshape((h, lq, d))?; // (H, lq, D)
                outs.push(o);
            }
            let o = if outs.len() == 1 { outs.remove(0) } else { Tensor::cat(&outs, 1)? };
            rows.push(o.transpose(0, 1)?.contiguous()?.reshape((t, h * d))?);
        }
        Tensor::stack(&rows, 0)
    }
}

/// Mix a cross-pass message into a looped layer input: `(1 - forget) * cur + gain * message`, both
/// gated on `cur` and the message.
fn control(cur: &Tensor, message: &Tensor, ctrl: &(Tensor, Tensor, Tensor)) -> Result<Tensor> {
    let (down, gain, forget) = ctrl;
    let gate = Tensor::cat(&[rms(cur, MIX_EPS)?, rms(message, MIX_EPS)?], D::Minus1)?.broadcast_matmul(down)?;
    let gain = gate.broadcast_matmul(gain)?.tanh()?;
    let forget = gate.broadcast_matmul(forget)?.tanh()?;
    (cur - forget.mul(cur)?)? + gain.mul(message)?
}
