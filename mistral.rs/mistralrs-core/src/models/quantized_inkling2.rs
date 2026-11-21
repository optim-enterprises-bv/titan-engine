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
use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
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
/// Query chunks up to this many tokens use the fused decode attention kernel.
const FUSED_ATTN_MAX_T: usize = 16;

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
    #[cfg(feature = "cuda")]
    if cur.device().is_cuda() && blocks.len() < crate::titan_inkling::MIX_MAX && fused_part("MIX") {
        let mut all: Vec<&Tensor> = blocks.iter().collect();
        all.push(cur);
        let y = crate::titan_inkling::mix(&all, query, MIX_EPS as f32)?;
        check("mix", &y, || mix_ref(blocks, cur, query))?;
        return Ok(y);
    }
    mix_ref(blocks, cur, query)
}

fn mix_ref(blocks: &[Tensor], cur: &Tensor, query: &Tensor) -> Result<Tensor> {
    let mut all: Vec<&Tensor> = blocks.iter().collect();
    all.push(cur);
    let states = rms(&Tensor::stack(&all, 2)?, MIX_EPS)?; // (n, t, nb, E)
    let e = states.dim(D::Minus1)?;
    let scores = (states.broadcast_matmul(&query.t()?)? / (e as f64).sqrt())?; // (n, t, nb, q)
    let probs = candle_nn::ops::softmax_last_dim(&scores.transpose(2, 3)?.contiguous()?)?; // (n, t, q, nb)
    probs.matmul(&states)
}

/// `INKLING_FUSED=0` falls back to the candle-op reference path.
fn fused() -> bool {
    mistralrs_quant::titan_cfg::flag("INKLING_FUSED", true)
}

/// `INKLING_CHECK=1`: run the reference beside each fused kernel and log the worst relative difference seen.
fn check(part: &'static str, fused: &Tensor, reference: impl FnOnce() -> Result<Tensor>) -> Result<()> {
    if !mistralrs_quant::titan_cfg::flag("INKLING_CHECK", false) {
        return Ok(());
    }
    static WORST: Mutex<Option<HashMap<&'static str, f32>>> = Mutex::new(None);
    let r = reference()?.to_dtype(DType::F32)?;
    let scale = r.abs()?.max_all()?.to_scalar::<f32>()?.max(1e-30);
    let diff = (fused.to_dtype(DType::F32)? - &r)?.abs()?.max_all()?.to_scalar::<f32>()? / scale;
    let mut g = WORST.lock().expect("check");
    let w = g.get_or_insert_with(HashMap::new).entry(part).or_insert(0.0);
    if diff > *w {
        *w = diff;
        tracing::info!("inkling2 check {part}: new worst relative max diff {diff:.3e}");
    }
    Ok(())
}

fn fused_part(part: &str) -> bool {
    fused() && mistralrs_quant::titan_cfg::flag(&format!("INKLING_FUSED_{part}"), true)
}

/// Row clip with an optional per-token scale after it (`tau` `(n, t, 1, 1)`).
fn clip_tau(x: &Tensor, scale: Option<&Tensor>, thr: f64, tau: Option<&Tensor>) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if x.device().is_cuda() && fused() {
        if fused_part("ROWS") {
            let y = crate::titan_inkling::rows(x, scale, thr as f32, tau, None)?;
            check("clip", &y, || {
                let y = clip(x, scale, thr)?;
                match tau {
                    Some(t) => y.broadcast_mul(t),
                    None => Ok(y),
                }
            })?;
            return Ok(y);
        }
    }
    let y = clip(x, scale, thr)?;
    match tau {
        Some(t) => y.broadcast_mul(t),
        None => Ok(y),
    }
}

fn rms_rows(x: &Tensor, eps: f64) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if x.device().is_cuda() && fused_part("ROWS") {
        let y = crate::titan_inkling::rows(x, None, 0.0, None, Some(eps as f32))?;
        check("rms", &y, || rms(x, eps))?;
        return Ok(y);
    }
    rms(x, eps)
}

/// One layer visit's convolution state: the gathered history (`None` when fresh) and the next one, `(n, C, taps - 1)`.
struct ConvCtx {
    input: Option<Tensor>,
    out: Tensor,
}

fn take(x: &Tensor, i: usize) -> Result<Tensor> {
    x.narrow(2, i, 1)?.squeeze(2)?.contiguous()
}

struct Conv {
    /// `(channels, taps)`
    w: Tensor,
}

impl Conv {
    /// [`Conv::forward`] on channels `off..off + C` of the visit's state.
    fn run(&self, cur: &Tensor, cx: &mut ConvCtx, off: usize, dynamic: Option<(&Tensor, usize)>) -> Result<Tensor> {
        let (n, _, c) = cur.dims3()?;
        let taps = self.w.dim(1)?;
        #[cfg(feature = "cuda")]
        if cur.device().is_cuda() && taps <= crate::titan_inkling::CONV_MAX_TAPS && fused_part("CONV") {
            let denom = (((if dynamic.is_some() { 2 } else { 1 }) * taps + 1) as f64).sqrt();
            let y = crate::titan_inkling::conv(cur, &self.w, cx.input.as_ref(), &cx.out, off, dynamic, (1.0 / denom) as f32)?;
            check("conv", &y, || {
                let hist = match &cx.input {
                    Some(s) => s.narrow(1, off, c)?.contiguous()?,
                    None => Tensor::zeros((n, c, taps - 1), DType::F32, cur.device())?,
                };
                let (y_ref, next) = self.forward(cur, &hist, dynamic.map(|d| d.0))?;
                check("conv_state", &cx.out.narrow(1, off, c)?, || Ok(next))?;
                Ok(y_ref)
            })?;
            return Ok(y);
        }
        let hist = match &cx.input {
            Some(s) => s.narrow(1, off, c)?.contiguous()?,
            None => Tensor::zeros((n, c, taps - 1), DType::F32, cur.device())?,
        };
        let (y, next) = self.forward(cur, &hist, dynamic.map(|d| d.0))?;
        cx.out = cx.out.slice_assign(&[0..n, off..off + c, 0..taps - 1], &next)?;
        Ok(y)
    }

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
    /// `(extent + 1, n_rel)`: the distance table with a zero row for distances past the extent (used transposed).
    rel_bias: Tensor,
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

enum ExpertWeights {
    /// whole `(n_expert, rows, cols)` tensors run by candle's indexed MoE (routing ids stay on the device)
    Indexed { gate_up: QMatMul, down: QMatMul },
    /// one matmul per routed expert: any GGUF format, one host sync per layer visit
    PerExpert { gate_up: Vec<Arc<dyn QuantMethod>>, down: Vec<Arc<dyn QuantMethod>> },
}

struct Experts {
    w: ExpertWeights,
    gate_up_scale: Option<Vec<f32>>,
    down_scale: Option<Vec<f32>>,
    /// device copies of the scales for the indexed path, `(n_expert,)`
    gate_up_scale_t: Option<Tensor>,
    down_scale_t: Option<Tensor>,
}

/// GGUF expert formats candle's CUDA indexed MoE covers.
fn indexed_moe_supported(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q8_0
            | GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
            | GgmlDType::IQ2XXS
            | GgmlDType::IQ2XS
            | GgmlDType::IQ2S
            | GgmlDType::IQ3XXS
            | GgmlDType::IQ3S
            | GgmlDType::IQ4XS
            | GgmlDType::IQ4NL
            | GgmlDType::NVFP4
    )
}

struct Moe {
    /// `(n_expert, E)`: used as `x @ router.t()` so the GEMV splits over E (a (E, n) weight gets n/32 blocks)
    router: Tensor,
    select_bias: Option<Tensor>,
    experts: Experts,
}

struct LoopLayer {
    /// `(4, E)`: Q, K, V and FFN message queries
    msg_query: Tensor,
    /// per controller, `(out, in)`: down `(rank, 2E)`, gain `(E, rank)`, forget `(E, rank)`
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
        let gu_name = format!("{prefix}.ffn_gate_up_exps.weight");
        let down_name = format!("{prefix}.ffn_down_exps.weight");
        let indexed = device.is_cuda()
            && !mistralrs_quant::titan_cfg::flag("INKLING_PER_EXPERT", false)
            && [&gu_name, &down_name].iter().all(|n| ct.tensor_info(n).is_ok_and(|i| indexed_moe_supported(i.ggml_dtype)));
        let w = if indexed {
            ExpertWeights::Indexed {
                gate_up: QMatMul::from_qtensor(ct.tensor(&gu_name, device)?)?,
                down: QMatMul::from_qtensor(ct.tensor(&down_name, device)?)?,
            }
        } else {
            let gu = ct.tensor(&gu_name, &Device::Cpu)?;
            let down = ct.tensor(&down_name, &Device::Cpu)?;
            ExpertWeights::PerExpert {
                gate_up: split_rows(&gu, 2 * hp.n_ff_exp, hp.n_embd, device)?.into_iter().map(linear).collect::<Result<Vec<_>>>()?,
                down: split_rows(&down, hp.n_embd, hp.n_ff_exp, device)?.into_iter().map(linear).collect::<Result<Vec<_>>>()?,
            }
        };
        let gate_up_scale = load_scale(ct, &format!("{prefix}.ffn_gate_up_exps.scale"))?;
        let down_scale = load_scale(ct, &format!("{prefix}.ffn_down_exps.scale"))?;
        let dev_scale = |s: &Option<Vec<f32>>| s.as_ref().map(|v| Tensor::from_slice(v, v.len(), device)).transpose();
        Ok(Self {
            w,
            gate_up_scale_t: dev_scale(&gate_up_scale)?,
            down_scale_t: dev_scale(&down_scale)?,
            gate_up_scale,
            down_scale,
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
            &flat.matmul(&self.router.t()?)?,
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
        let ExpertWeights::PerExpert { gate_up, down } = &self.experts.w else {
            return self.forward_indexed(&flat, &topk, hp)?.reshape((n, t, e));
        };
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
            let mut gu = fwd(&gate_up[id], &xe)?;
            if let Some(s) = &self.experts.gate_up_scale {
                gu = (gu * f64::from(s[id]))?;
            }
            let gate = gu.narrow(1, 0, ff)?;
            let up = gu.narrow(1, ff, ff)?;
            let act = situ(&gate, &up, hp.situ_beta, hp.situ_lb)?;
            let mut y = fwd(&down[id], &act.contiguous()?)?;
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

impl Moe {
    /// All routed experts of all tokens in two indexed launches; `flat` `(T, E)`, returns `(T, E)`.
    fn forward_indexed(&self, flat: &Tensor, topk: &crate::ops::TopKOutput, hp: &Hp) -> Result<Tensor> {
        let ExpertWeights::Indexed { gate_up, down } = &self.experts.w else {
            candle_core::bail!("inkling2: indexed experts expected");
        };
        let (tokens, e) = flat.dims2()?;
        let k = hp.top_k;
        let ff = hp.n_ff_exp;
        let ids = topk.indices.contiguous()?;
        let per_slot = |scale: &Option<Tensor>| -> Result<Option<Tensor>> {
            scale.as_ref().map(|s| s.index_select(&ids.flatten_all()?, 0)?.reshape((tokens, k, 1))).transpose()
        };
        let mut gu = gate_up.indexed_moe_forward(&flat.reshape((tokens, 1, e))?, &ids)?.to_dtype(DType::F32)?; // (T, k, 2F)
        if let Some(s) = per_slot(&self.experts.gate_up_scale_t)? {
            gu = gu.broadcast_mul(&s)?;
        }
        let act = situ(&gu.narrow(2, 0, ff)?, &gu.narrow(2, ff, ff)?, hp.situ_beta, hp.situ_lb)?.contiguous()?;
        let mut y = down.indexed_moe_forward(&act, &ids)?.to_dtype(DType::F32)?; // (T, k, E)
        let mut w = topk.values.to_dtype(DType::F32)?.reshape((tokens, k, 1))?;
        if let Some(s) = per_slot(&self.experts.down_scale_t)? {
            w = w.mul(&s)?;
        }
        y = y.broadcast_mul(&w)?;
        y.sum(1)
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
        let rel_bias = Tensor::cat(&[&rel_bias, &Tensor::zeros((1, hp.n_rel), DType::F32, device)?], 0)?.contiguous()?;
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
            rel_bias,
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
    /// distance-table index and mask per (sequence, sliding, keys, first query, queries): every visit of
    /// the same kind shares them, so each is uploaded once per forward
    bias_idx: Mutex<HashMap<(usize, bool, usize, usize, usize), (Tensor, Tensor)>>,
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
                    router,
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
                    ctrl.push((down, gain, forget));
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
            bias_idx: Mutex::new(HashMap::new()),
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
        let mut prof = Prof::new(cur.device());
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
        prof.mark("state");
        let mut cx = ConvCtx {
            input: if fresh { None } else { Some(pool.gather_conv_state(&indices)?.to_dtype(DType::F32)?.contiguous()?) },
            out: Tensor::zeros((n, hp.conv_channels(), hp.conv_k - 1), DType::F32, cur.device())?,
        };

        prof.mark("inputs");
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

        prof.mark("proj");
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
        let k = a.k_conv.run(&fwd(&a.wk, &inputs[1])?, &mut cx, 0, k_dyn.as_ref().map(|d| (d, hkv)))?;
        let v = a.v_conv.run(&fwd(&a.wv, &inputs[2])?, &mut cx, kg, v_dyn.as_ref().map(|d| (d, hkv)))?;
        let r = fwd(&a.rel, &inputs[0])?;

        let tau = (!swa).then_some(&pos.tau);
        let q = clip_tau(&q.reshape((n, t, h, d))?, Some(&a.q_clip), hp.qk_clip, tau)?;
        let k = clip_tau(&k.reshape((n, t, hkv, d))?, None, hp.qk_clip, None)?;
        let v = rms_rows(&v.reshape((n, t, hkv, d))?, hp.rms_eps)?;
        let r = clip_tau(&r.reshape((n, t, h, hp.n_rel))?, Some(&a.rel_clip), hp.rel_clip, tau)?;
        let sinks = fwd(&a.sink, &inputs[0])?; // (n, t, H)
        // distance table per (token, head): R . bias_table - sink; the last column (past the extent) is -sink
        let table = r.broadcast_matmul(&a.rel_bias.t()?)?.broadcast_sub(&sinks.unsqueeze(3)?)?; // (n, t, H, extent+1)

        let (k_all, v_all) = kv.append(
            &k.transpose(1, 2)?.contiguous()?.to_dtype(DType::BF16)?,
            &v.transpose(1, 2)?.contiguous()?.to_dtype(DType::BF16)?,
        )?;
        prof.mark("attend");
        let attn = self.attend(&q, &k_all, &v_all, &table, swa, pos)?; // (n, t, H*D)

        prof.mark("out");
        let gate = candle_nn::ops::sigmoid(&fwd(&a.wgate, &inputs[0])?)?;
        let o = fwd(&a.wo, &attn.mul(&gate)?)?;
        let o = a.post_conv.run(&o, &mut cx, 2 * kg, None)?;

        let mut cur = if ip % hp.res_block == 0 {
            blocks.push(inp_sa);
            o
        } else {
            (inp_sa + o)?
        };

        prof.mark("mix_ffn");
        // MoE
        let ffn_inp = cur.clone();
        let mut f = take(&mix(blocks, &cur, &layer.ffn_res_query)?, 0)?;
        if let Some((m, ll)) = msgs {
            f = control(&f, &take(m, CTRL_FFN)?, &ll.ctrl[CTRL_FFN])?;
        }
        let f = layer.ffn_norm.forward(&f)?;
        prof.mark("moe");
        let f = layer.moe.forward(&f, hp)?;
        prof.mark("conv");
        let f = layer.ffn_post_conv.run(&f, &mut cx, 2 * kg + e, None)?;
        cur = (ffn_inp + f)?;

        prof.mark("hist");
        pool.scatter_conv_state_with_host_indices(&indices, md.state_indices_host(), &cx.out.to_dtype(pool.conv_dtype())?)?;
        prof.mark("end");
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
            let end = pos.starts[b] + pos.t;
            #[cfg(feature = "cuda")]
            if dev.is_cuda() && t <= FUSED_ATTN_MAX_T && d == 128 && fused_part("ATTN") {
                let o = crate::titan_inkling::attn(
                    &q.get(b)?,
                    &k_all.get(b)?,
                    &v_all.get(b)?,
                    &table.get(b)?,
                    pos.starts[b],
                    end - l,
                    swa.then_some(hp.n_swa),
                    scale as f32,
                )?;
                check("attn", &o, || self.attend_ref_one(q, k_all, v_all, table, swa, pos, b))?;
                rows.push(o);
                continue;
            }
            rows.push(self.attend_ref_one(q, k_all, v_all, table, swa, pos, b)?);
        }
        Tensor::stack(&rows, 0)
    }

    fn attend_ref_one(&self, q: &Tensor, k_all: &Tensor, v_all: &Tensor, table: &Tensor, swa: bool, pos: &Positions, b: usize) -> Result<Tensor> {
        let hp = &self.hp;
        let (_, t, h, d) = q.dims4()?;
        let hkv = hp.n_head_kv;
        let g = h / hkv;
        let l = k_all.dim(2)?;
        let ext = hp.rel_extent;
        let scale = 1.0 / (d as f64).sqrt();
        let dev = q.device();
        {
            let kb = k_all.get(b)?.to_dtype(DType::F32)?; // (Hkv, L, D)
            let vb_t = v_all.get(b)?.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?; // (Hkv, D, L)
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
                let key = (b, swa, l, i0, lq);
                let cached = pos.bias_idx.lock().expect("bias index cache poisoned").get(&key).cloned();
                let (idx, mask) = match cached {
                    Some(v) => v,
                    None => {
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
                        pos.bias_idx.lock().expect("bias index cache poisoned").insert(key, (idx.clone(), mask.clone()));
                        (idx, mask)
                    }
                };
                let bias = tb.narrow(1, i0, lq)?.contiguous()?.gather(&idx, 2)?; // (H, lq, L)
                let logits = (scores + bias)?.broadcast_add(&mask)?;
                // softmax over the keys and one zero-logit sink
                let m = logits.max_keepdim(2)?.clamp(0f32, f32::INFINITY)?;
                let ex = logits.broadcast_sub(&m)?.exp()?;
                let denom = (ex.sum_keepdim(2)? + m.neg()?.exp()?)?;
                let p = ex.broadcast_div(&denom)?.reshape((hkv, g * lq, l))?;
                let o = p.matmul(&vb_t.t()?)?.reshape((h, lq, d))?; // (H, lq, D)
                outs.push(o);
            }
            let o = if outs.len() == 1 { outs.remove(0) } else { Tensor::cat(&outs, 1)? };
            o.transpose(0, 1)?.contiguous()?.reshape((t, h * d))
        }
    }
}

/// `INKLING_PROFILE=1`: device-synchronized wall time per section of a layer visit, printed every 2048 visits.
struct Prof {
    dev: Option<Device>,
    last: std::time::Instant,
    cur: &'static str,
}

static PROF_ACC: Mutex<Option<(HashMap<&'static str, f64>, u64)>> = Mutex::new(None);

impl Prof {
    fn new(dev: &Device) -> Self {
        let on = mistralrs_quant::titan_cfg::flag("INKLING_PROFILE", false);
        if on {
            let _ = dev.synchronize();
        }
        Self { dev: on.then(|| dev.clone()), last: std::time::Instant::now(), cur: "pre" }
    }
    fn mark(&mut self, next: &'static str) {
        let Some(dev) = &self.dev else { return };
        let _ = dev.synchronize();
        let now = std::time::Instant::now();
        let dt = now.duration_since(self.last).as_secs_f64();
        let mut acc = PROF_ACC.lock().expect("prof");
        let (map, visits) = acc.get_or_insert_with(|| (HashMap::new(), 0));
        *map.entry(self.cur).or_default() += dt;
        if next == "end" {
            *visits += 1;
            if *visits % 2048 == 0 {
                let mut v: Vec<_> = map.iter().map(|(k, t)| (*k, *t)).collect();
                v.sort_by(|a, b| b.1.total_cmp(&a.1));
                let total: f64 = v.iter().map(|x| x.1).sum();
                let parts: Vec<String> = v.iter().map(|(k, t)| format!("{k} {:.0}us", 1e6 * t / *visits as f64)).collect();
                tracing::info!("inkling2 profile ({visits} visits, {:.0}us/visit): {}", 1e6 * total / *visits as f64, parts.join(", "));
            }
        }
        self.last = now;
        self.cur = next;
    }
}

/// Mix a cross-pass message into a looped layer input: `(1 - forget) * cur + gain * message`, both
/// gated on `cur` and the message.
fn control(cur: &Tensor, message: &Tensor, ctrl: &(Tensor, Tensor, Tensor)) -> Result<Tensor> {
    let (down, gain, forget) = ctrl;
    let gate = Tensor::cat(&[rms(cur, MIX_EPS)?, rms(message, MIX_EPS)?], D::Minus1)?.broadcast_matmul(&down.t()?)?;
    let gain = gate.broadcast_matmul(&gain.t()?)?.tanh()?;
    let forget = gate.broadcast_matmul(&forget.t()?)?.tanh()?;
    (cur - forget.mul(cur)?)? + gain.mul(message)?
}
