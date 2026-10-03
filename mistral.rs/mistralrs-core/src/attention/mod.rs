#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::{attention::backends::cpu, pipeline::text_models_inputs_processor::FlashParams};

use candle_core::{DType, Device, Result, Tensor};

/// Attention mask passed to [`Sdpa::run_attention`].
///
/// Encodes both the mask data and the *intent*, whether the attention layer
/// should use flash attention (causal handled by the kernel), eager attention
/// with an explicit mask tensor, or no masking at all.
#[derive(Clone, Debug)]
pub enum AttentionMask {
    /// No masking. Used for single-token decode or truly unmasked attention.
    None,
    /// Flash attention with `is_causal = true`. No mask tensor is needed;
    /// the flash kernel applies causal masking internally. Also signals
    /// "this is a prefill" to the paged attention layer.
    CausalFlash,
    /// An explicit mask tensor (causal, sliding window, bidirectional, etc).
    /// CPU fused attention can consume it directly; other backends route to eager as needed.
    Custom(Tensor),
}

impl AttentionMask {
    /// Extract the inner tensor as `Option<&Tensor>`.
    ///
    /// Returns `Some(&tensor)` for [`Custom`](Self::Custom), `None` otherwise.
    /// Useful for interfacing with paged-attention and MLA helpers that still
    /// accept `Option<&Tensor>`.
    pub fn as_option_tensor(&self) -> Option<&Tensor> {
        match self {
            Self::Custom(t) => Some(t),
            _ => None,
        }
    }

    /// Returns `true` when the mask carries an explicit tensor
    /// ([`Custom`](Self::Custom) variant), mirroring the old
    /// `Option<Tensor>::is_some()` semantics.
    pub fn is_custom(&self) -> bool {
        matches!(self, Self::Custom(_))
    }

    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }
}

mod backends;
#[cfg(feature = "cuda")]
mod flash_decode;
#[cfg(feature = "cuda")]
mod flash_prefill;

#[allow(unused)]
pub(crate) use backends::cpu::fast_exp;
#[cfg(feature = "cuda")]
use backends::naive::maybe_synchronize;
use backends::naive::maybe_synchronize_for;
pub(crate) use backends::{
    flash_attn, flash_backend_supports, flash_backend_supports_sdpa, naive_sdpa, sinks_attn,
    sinks_backend_is_available, sinks_backend_supports,
};

/// Chunk size for attention computation to avoid OOM on long sequences
pub(crate) const ATTENTION_CHUNK_SIZE: usize = 1024;
pub(crate) const FLASH_ATTN_NATIVE_MAX_GQA_GROUP: usize = 8;

#[cfg(any(
    feature = "flash-attn",
    feature = "flash-attn-v3",
    all(feature = "cuda", target_family = "unix")
))]
pub(crate) fn sliding_window_left(sliding_window: Option<usize>) -> Option<usize> {
    sliding_window.map(|window| window.saturating_sub(1))
}

pub(crate) fn eager_attention_mask(
    query_len: usize,
    key_len: usize,
    causal: bool,
    sliding_window: Option<usize>,
    dtype: DType,
    device: &Device,
) -> Result<Option<Tensor>> {
    if !causal && sliding_window.is_none() {
        return Ok(None);
    }
    let prefix_len = key_len.saturating_sub(query_len);
    let mut mask = Vec::with_capacity(query_len * key_len);
    for query_idx in 0..query_len {
        let query_pos = prefix_len + query_idx;
        for key_idx in 0..key_len {
            let future = causal && key_idx > query_pos;
            let too_old = sliding_window
                .is_some_and(|window| query_pos >= window && key_idx <= query_pos - window);
            mask.push(if future || too_old {
                f32::NEG_INFINITY
            } else {
                0.0
            });
        }
    }
    Tensor::from_vec(mask, (query_len, key_len), device)
        .and_then(|mask| mask.to_dtype(dtype))
        .map(Some)
}

/// Generic chunked attention computation that can be used by different backends
pub(crate) fn chunked_attention<F>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    attention_fn: F,
) -> Result<Tensor>
where
    F: Fn(&Tensor, &Tensor, &Tensor, Option<&Tensor>) -> Result<Tensor>,
{
    chunked_attention_with_offset(q, k, v, mask, |q, k, v, mask, _offset| {
        attention_fn(q, k, v, mask)
    })
}

pub(crate) fn chunked_attention_with_offset<F>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    attention_fn: F,
) -> Result<Tensor>
where
    F: Fn(&Tensor, &Tensor, &Tensor, Option<&Tensor>, usize) -> Result<Tensor>,
{
    let seq_len = q.dim(2)?;
    let chunk_size = attention_chunk_size(q, k)?;

    if seq_len <= chunk_size {
        return attention_fn(q, k, v, mask, 0);
    }

    let num_chunks = seq_len.div_ceil(chunk_size);
    let mut attn_chunks = Vec::with_capacity(num_chunks);

    for chunk_idx in 0..num_chunks {
        let offset = chunk_idx * chunk_size;
        let chunk_len = chunk_size.min(seq_len - offset);

        // Extract query chunk
        let q_chunk = q.narrow(2, offset, chunk_len)?;

        // Extract mask chunk if present
        let mask_chunk = mask
            .map(|m| {
                match m.rank() {
                    2 => {
                        // For 2D masks (seq_len, seq_len), narrow along dimension 0
                        m.narrow(0, offset, chunk_len)
                    }
                    3 => {
                        // For 3D masks (batch, seq_len, seq_len), narrow along dimension 1
                        m.narrow(1, offset, chunk_len)
                    }
                    4 => {
                        // For 4D masks (batch, heads, seq_len, seq_len), narrow along dimension 2
                        m.narrow(2, offset, chunk_len)
                    }
                    _ => m.narrow(2, offset, chunk_len), // Default to dimension 2
                }
            })
            .transpose()?;

        let att_chunk = attention_fn(&q_chunk, k, v, mask_chunk.as_ref(), offset)?;

        attn_chunks.push(att_chunk);
    }

    Tensor::cat(&attn_chunks, 2)
}

/// Score elements (batch * heads * query rows * kv len) one attention chunk may hold beyond
/// `GQA_GROUPED_MIN_KV` keys: 16 Mi, 64 MiB as F32 softmax input.
const ATTENTION_CHUNK_SCORE_ELEMS: usize = 16 << 20;

/// Query rows per attention chunk: `ATTENTION_CHUNK_SIZE` up to `GQA_GROUPED_MIN_KV` keys (as
/// before, so shorter contexts keep their exact results), then as many as fit
/// `ATTENTION_CHUNK_SCORE_ELEMS` (a long prompt, or a prompt chunk late in a long context), at
/// least 16.
fn attention_chunk_size(q: &Tensor, k: &Tensor) -> Result<usize> {
    let (b, h, _, _) = q.dims4()?;
    let kv_len = k.dim(2)?.max(1);
    if kv_len < GQA_GROUPED_MIN_KV {
        return Ok(ATTENTION_CHUNK_SIZE);
    }
    let fit = ATTENTION_CHUNK_SCORE_ELEMS / (b * h * kv_len).max(1);
    Ok(ATTENTION_CHUNK_SIZE.min(fit.max(16)))
}

/// KV length from which eager GQA attention runs grouped (`gqa_grouped_sdpa`) instead of on
/// `repeat_kv` copies: the copies are `n_kv_groups` times the layer's KV, per layer and per step
/// (1 GiB each for K and V at 64k tokens, 16 query / 2 KV heads, BF16). Shorter contexts keep the
/// repeat_kv path and its exact results.
const GQA_GROUPED_MIN_KV: usize = 4096;

/// Eager attention with each KV head's query heads stacked as rows of one matmul against that
/// head's K and V (no repeated copies of the KV). `q` `(b, h, s, d)` with query head
/// `j * n_rep + i` served by KV head `j` (the `repeat_kv` order); `mask`, if any, `(s, kv)`.
/// Query rows run in chunks sized by `attention_chunk_size`.
fn gqa_grouped_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    maybe_synchronize_for(q)?;
    let (b, h, seq_len, d) = q.dims4()?;
    let (_, kv_heads, kv_len, _) = k.dims4()?;
    let n_rep = h / kv_heads;
    let v_d = v.dim(3)?;
    let chunk = attention_chunk_size(q, k)?;
    let k_t = k.t()?;
    let mut out = Vec::with_capacity(seq_len.div_ceil(chunk));
    for off in (0..seq_len).step_by(chunk) {
        let len = chunk.min(seq_len - off);
        let qc = (q.narrow(2, off, len)? * f64::from(sdpa_params.softmax_scale))?
            .contiguous()?
            .reshape((b, kv_heads, n_rep * len, d))?;
        let mut att = mistralrs_quant::MatMul.matmul(&qc, &k_t)?;
        if let Some(mask) = mask {
            att = att
                .reshape((b, kv_heads, n_rep, len, kv_len))?
                .broadcast_add(&mask.narrow(0, off, len)?)?
                .reshape((b, kv_heads, n_rep * len, kv_len))?;
        }
        let att_dtype = att.dtype();
        let att = candle_nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?.to_dtype(att_dtype)?;
        out.push(
            mistralrs_quant::MatMul
                .matmul(&att, v)?
                .reshape((b, h, len, v_d))?,
        );
    }
    if out.len() == 1 {
        Ok(out.pop().expect("one chunk"))
    } else {
        Tensor::cat(&out, 2)
    }
}

/// `TITAN_ATTN_NOCOPY`: how a CUDA GQA decode row (one query row, no mask tensor) below
/// `TITAN_ATTN_NOCOPY_MAX` keys (default `GQA_GROUPED_MIN_KV`) runs the cuBLASLt attention.
/// - `0`: `repeat_kv` K and V to all query heads, then a transposed copy of the expanded V (three
///   copies of `heads x kv_len x head_dim` per layer and step, the old path);
/// - `1`: the same two cuBLASLt GEMMs per query head on the same operands, read per KV head
///   through a batch stride of 0; only the 2-KV-head V is transposed (1/n_rep of one old copy);
/// - `2`: one GEMM per KV head with its query heads as the rows (no copy at all, the V GEMM without a
///   transpose); a different GEMM shape, so not bit-identical to `0`;
/// - `3` (default): mode 1 run with the cuBLASLt algorithm picked for the old all-heads batch (mode 1's own
///   pick for a batch of `n_rep` with stride 0 can differ, e.g. in split-K, and round differently).
#[cfg(feature = "cuda")]
fn nocopy_mode() -> (u8, usize) {
    static M: mistralrs_quant::titan_cfg::GenCell<(u8, usize)> = mistralrs_quant::titan_cfg::GenCell::new();
    *M.get_or_init(|| {
        let mode = mistralrs_quant::titan_cfg::var("TITAN_ATTN_NOCOPY").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
        let max = mistralrs_quant::titan_cfg::var("TITAN_ATTN_NOCOPY_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(GQA_GROUPED_MIN_KV);
        tracing::info!("titan attention: GQA decode nocopy mode {mode} below {max} keys");
        (mode, max)
    })
}

/// The cuBLASLt eager attention of `run_attention_noflash` for one query row of a GQA layer
/// without `repeat_kv` (see `nocopy_mode`). `None` where it does not apply: the caller runs the
/// old path.
#[cfg(feature = "cuda")]
fn gqa_decode_nocopy(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    sdpa_params: &SdpaParams,
    causal: bool,
) -> Result<Option<Tensor>> {
    let (mode, max_kv) = nocopy_mode();
    let n_rep = sdpa_params.n_kv_groups;
    if mode == 0 || n_rep <= 1 || mask.is_some() || !q.device().is_cuda() {
        return Ok(None);
    }
    let (b, h, seq_len, d) = q.dims4()?;
    let (_, kv_heads, kv_len, k_d) = k.dims4()?;
    let v_d = v.dim(3)?;
    let row_major = |t: &Tensor| {
        let st = t.stride();
        st[3] == 1 && st[2] == t.dim(3).unwrap_or(0)
    };
    if b != 1
        || seq_len != 1
        || kv_len >= max_kv
        || h != kv_heads * n_rep
        || k_d != d
        || !q.is_contiguous()
        || !row_major(k)
        || !row_major(v)
        || mistralrs_quant::distributed::use_nccl()
    {
        return Ok(None);
    }
    let Some(cublaslt) = mistralrs_quant::cublaslt::CUBLASLT_CONTROLLER.get_for_device(q.device())
    else {
        return Ok(None);
    };
    maybe_synchronize_for(q)?;
    let alpha = Some(sdpa_params.softmax_scale / sdpa_params.softcap.unwrap_or(1.0));
    // (h, 1, d), head `j * n_rep + i` served by KV head `j`
    let q_flat = q.flatten(0, 1)?;
    let (k0, v0) = (k.squeeze(0)?, v.squeeze(0)?); // (kv_heads, kv_len, d), rows contiguous
    // mode 3: the algorithm cuBLASLt picks for the old all-heads batch
    let heur = |stride: usize| (mode == 3).then(|| (h as i32, stride as i64));
    let mut scores = if mode == 1 || mode == 3 {
        let parts = (0..kv_heads)
            .map(|j| {
                let kj = k0.narrow(0, j, 1)?.broadcast_as((n_rep, kv_len, d))?;
                let qj = q_flat.narrow(0, j * n_rep, n_rep)?;
                match heur(kv_len * d) {
                    Some(hb) => cublaslt.batch_matmul_heur(&kj, &qj, alpha, hb),
                    None => cublaslt.batch_matmul(&kj, &qj, None, alpha, None, None, None),
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&parts, 0)?
    } else {
        cublaslt
            .batch_matmul(&k0, &q_flat.reshape((kv_heads, n_rep, d))?, None, alpha, None, None, None)?
            .reshape((h, 1, kv_len))?
    };
    if let Some(softcap) = sdpa_params.softcap {
        scores = (scores.tanh()? * softcap as f64)?;
    }
    let scores_dtype = scores.dtype();
    if scores_dtype == DType::BF16 || scores_dtype == DType::F16 {
        scores = scores.to_dtype(DType::F32)?;
    }
    if causal {
        crate::ops::cuda_apply_causal_mask_f32(&scores, 0, kv_len - 1)?;
    }
    scores = candle_nn::ops::softmax_last_dim(&scores)?;
    if scores.dtype() != scores_dtype {
        scores = scores.to_dtype(scores_dtype)?;
    }
    let y = if mode == 1 || mode == 3 {
        let vt = v0.t()?.contiguous()?; // (kv_heads, v_d, kv_len)
        let parts = (0..kv_heads)
            .map(|j| {
                let vj = vt.narrow(0, j, 1)?.broadcast_as((n_rep, v_d, kv_len))?;
                let sj = scores.narrow(0, j * n_rep, n_rep)?;
                match heur(v_d * kv_len) {
                    Some(hb) => cublaslt.batch_matmul_heur(&vj, &sj, None, hb),
                    None => cublaslt.batch_matmul(&vj, &sj, None, None, None, None, None),
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&parts, 0)?
    } else {
        cublaslt.batch_matmul_nn(&v0, &scores.reshape((kv_heads, n_rep, kv_len))?, None)?
    };
    Ok(Some(y.reshape((b, h, seq_len, v_d))?))
}

/// Decode rows `0..n` of `q` (b, h, n, d) that see keys `past + 1 ..= past + n` of `k`, `v`: the first
/// row the flash-decode kernel (`TITAN_ATTN_FLASH`, CUDA) takes, `n` if none. Rows from there on go to
/// `flash_decode_rows` in one call; earlier rows keep the per-row path.
pub(crate) fn flash_decode_first_row(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    past: usize,
    n: usize,
    sdpa_params: &SdpaParams,
) -> usize {
    #[cfg(feature = "cuda")]
    {
        flash_decode::first_row(q, k, v, past, n, sdpa_params)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (q, k, v, past, sdpa_params);
        n
    }
}

/// Whether a CUDA prompt chunk of `rows` query rows ending at `kv_len` keys runs the flash-prefill kernel
/// (`TITAN_ATTN_FLASH_PREFILL`); the model's shapes must also pass `flash_prefill_supported`.
pub(crate) fn flash_prefill_wanted(rows: usize, kv_len: usize) -> bool {
    #[cfg(feature = "cuda")]
    {
        flash_prefill::wanted(rows, kv_len)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (rows, kv_len);
        false
    }
}

/// Whether `flash_prefill` supports these tensors (device, dtype, shapes, layouts).
pub(crate) fn flash_prefill_supported(q: &Tensor, k: &Tensor, v: &Tensor, sdpa_params: &SdpaParams) -> bool {
    #[cfg(feature = "cuda")]
    {
        flash_prefill::supported(q, k, v, sdpa_params)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (q, k, v, sdpa_params);
        false
    }
}

/// Causal attention of the last `s` query positions `q` (1, h, s, d) over `k`, `v` (1, kvh, kv_len, d) with the
/// flash-prefill kernel: (1, h, s, d).
pub(crate) fn flash_prefill(q: &Tensor, k: &Tensor, v: &Tensor, sdpa_params: &SdpaParams) -> Result<Tensor> {
    flash_prefill_window(q, k, v, sdpa_params, 0)
}

/// Whether `flash_prefill_any` takes these tensors: bf16 or f16 q / k / v, head dim 256 or 512 (gemma4).
pub(crate) fn flash_prefill_any_supported(q: &Tensor, k: &Tensor, v: &Tensor) -> bool {
    #[cfg(feature = "cuda")]
    {
        flash_prefill::supported_any(q, k, v)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (q, k, v);
        false
    }
}

/// Causal flash-prefill attention for head dim 256 or 512 on bf16 or f16 q / k / v (1, h, s, d) / (1, kvh, kv_len, d),
/// scale `softmax_scale`, each query limited to its last `win` keys (`win == 0`: no limit); (1, h, s, d) in f32 when
/// `out32` (f16 inputs), else bf16.
pub(crate) fn flash_prefill_any(q: &Tensor, k: &Tensor, v: &Tensor, softmax_scale: f32, win: usize, out32: bool) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        flash_prefill::attend_any(q, k, v, softmax_scale, win, out32)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (q, k, v, softmax_scale, win, out32);
        candle_core::bail!("flash-prefill needs CUDA")
    }
}

/// `flash_prefill` with each query limited to its last `win` keys (itself included); `win == 0`: no limit.
pub(crate) fn flash_prefill_window(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sdpa_params: &SdpaParams,
    win: usize,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        flash_prefill::attend(q, k, v, sdpa_params, win)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (q, k, v, sdpa_params, win);
        candle_core::bail!("flash-prefill needs CUDA")
    }
}

/// The key count from which CUDA decode rows take the flash-decode kernel, `None` when it is off.
pub(crate) fn flash_decode_min_kv() -> Option<usize> {
    #[cfg(feature = "cuda")]
    {
        flash_decode::min_kv()
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}

/// Attention for decode rows `q` (1, h, m, d), row `i` seeing keys `< past + i + 1` of `k`, `v`, in one
/// flash-decode call: `(1, m, h * d)`. Only for rows `flash_decode_first_row` assigned to it.
pub(crate) fn flash_decode_rows(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    past: usize,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        flash_decode::decode_rows(q, k, v, past, sdpa_params)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (q, k, v, past, sdpa_params);
        candle_core::bail!("flash-decode needs CUDA")
    }
}

/// `TITAN_ATTN_F32REF=1`: see `run_attention_noflash` (a debug reference, off by default).
fn f32_reference_attention() -> bool {
    static ON: mistralrs_quant::titan_cfg::GenCell<bool> = mistralrs_quant::titan_cfg::GenCell::new();
    *ON.get_or_init(|| mistralrs_quant::titan_cfg::var("TITAN_ATTN_F32REF").is_ok_and(|v| v == "1"))
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(x)
    } else {
        // one broadcast copy instead of n_rep concatenated ones (same element order)
        let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
        x.unsqueeze(2)?
            .broadcast_as((b_sz, n_kv_head, n_rep, seq_len, head_dim))?
            .contiguous()?
            .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

fn run_flash_attn_cpu_for_dtype(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    // KV may be stored at lower precision than the activations (f16 CPU KV cache);
    // kernels accumulate in f32 either way, so convert q down and the output back up.
    let out_dtype = q.dtype();
    let q_conv;
    let q = if q.dtype() != k.dtype() {
        q_conv = q.to_dtype(k.dtype())?;
        &q_conv
    } else {
        q
    };
    let res = match k.dtype() {
        DType::F32 => cpu::run_flash_attn_cpu::<f32>(q, k, v, mask, sdpa_params),
        DType::F16 => cpu::run_flash_attn_cpu::<half::f16>(q, k, v, mask, sdpa_params),
        DType::BF16 => cpu::run_flash_attn_cpu::<half::bf16>(q, k, v, mask, sdpa_params),
        other => candle_core::bail!("Unsupported dtype for CPU flash attn: {other:?}"),
    }?;
    if res.dtype() != out_dtype {
        res.to_dtype(out_dtype)
    } else {
        Ok(res)
    }
}

fn packed_attention_backend_is_available(q: &Tensor, sdpa_params: &SdpaParams) -> Result<bool> {
    let head_dim = q.dim(3)?;
    if sdpa_params.sinks.is_some() {
        return Ok(q.dim(0)? > 1 && sinks_backend_is_available(q, head_dim));
    }
    Ok(q.device().is_cuda()
        && crate::using_flash_attn()
        && matches!(q.dtype(), DType::F16 | DType::BF16)
        && flash_backend_supports_sdpa(
            head_dim,
            sdpa_params.softcap.is_some(),
            sdpa_params.sliding_window.is_some(),
        ))
}

pub struct SdpaParams {
    pub n_kv_groups: usize,
    pub softcap: Option<f32>,
    pub softmax_scale: f32,
    pub sliding_window: Option<usize>,
    pub sinks: Option<Tensor>,
}

pub struct Sdpa;

impl Sdpa {
    /// Computes softmax(QK^T*sqrt(d_k))V
    ///
    /// Inputs:
    /// - q: (b_sz, n_attn_heads, q_len, head_dim)
    /// - k: (b_sz, n_kv_heads, q_len, head_dim)
    /// - v: (b_sz, n_kv_heads, q_len, head_dim)
    ///
    /// Dispatch attention based on the `AttentionMask` variant:
    ///
    /// - `AttentionMask::CausalFlash`: flash attention with `is_causal = true`
    /// - `AttentionMask::None`: flash if available (decode), else eager without mask
    /// - `AttentionMask::Custom`: CPU fused attention or eager attention with the explicit mask tensor
    #[allow(unused_variables, clippy::too_many_arguments)]
    pub fn run_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: &AttentionMask,
        flash_params: Option<&FlashParams>,
        sdpa_params: &SdpaParams,
    ) -> Result<Tensor> {
        if flash_params.is_some_and(|params| params.packed)
            && (!matches!(mask, AttentionMask::CausalFlash)
                || !flash_params.is_some_and(|params| params.causal)
                || !packed_attention_backend_is_available(q, sdpa_params)?)
        {
            candle_core::bail!("packed prefill requires causal varlen attention support");
        }

        // If sinks are present, dispatch to the sinks backend
        if let Some(sinks) = &sdpa_params.sinks {
            let mask_tensor = match mask {
                AttentionMask::Custom(t) => Some(t),
                _ => None,
            };
            return sinks_attn(q, k, v, sinks, mask_tensor, flash_params, sdpa_params);
        }

        // The mask carries causality already; the kernel-level do_causal
        // early-exit is safe to enable only when the request is known causal.
        let do_causal = flash_params.is_some_and(|p| p.causal);

        if let AttentionMask::Custom(mask_tensor) = mask {
            if q.device().is_cpu() {
                let q = q.transpose(1, 2)?;
                let k = k.transpose(1, 2)?;
                let v = v.transpose(1, 2)?;
                return run_flash_attn_cpu_for_dtype(&q, &k, &v, Some(mask_tensor), sdpa_params);
            }

            return self.run_attention_noflash(q, k, v, Some(mask_tensor), sdpa_params, do_causal);
        }

        // CausalFlash or None: try flash attention, fall back to eager
        let can_use_flash = q.device().is_cpu()
            || q.device().is_cuda() && crate::using_flash_attn() && q.dtype() != DType::F32;

        if can_use_flash {
            let expanded_kv = if q.device().is_cuda()
                && crate::using_flash_attn()
                && q.dtype() != DType::F32
                && sdpa_params.n_kv_groups > FLASH_ATTN_NATIVE_MAX_GQA_GROUP
            {
                Some((
                    repeat_kv(k.clone(), sdpa_params.n_kv_groups)?,
                    repeat_kv(v.clone(), sdpa_params.n_kv_groups)?,
                    SdpaParams {
                        n_kv_groups: 1,
                        softcap: sdpa_params.softcap,
                        softmax_scale: sdpa_params.softmax_scale,
                        sliding_window: sdpa_params.sliding_window,
                        sinks: sdpa_params.sinks.clone(),
                    },
                ))
            } else {
                None
            };
            let (k, v, sdpa_params) = match &expanded_kv {
                Some((k, v, sdpa_params)) => (k, v, sdpa_params),
                None => (k, v, sdpa_params),
            };

            let head_dim = q.dim(3)?;
            if q.device().is_cuda()
                && !flash_backend_supports_sdpa(
                    head_dim,
                    sdpa_params.softcap.is_some(),
                    sdpa_params.sliding_window.is_some(),
                )
            {
                if flash_params.is_some_and(|params| params.packed) {
                    candle_core::bail!(
                        "packed prefill requires FlashAttention support for head_dim={head_dim} \
                         with softcap={}, sliding_window={}",
                        sdpa_params.softcap.is_some(),
                        sdpa_params.sliding_window.is_some()
                    );
                }
                let causal = matches!(mask, AttentionMask::CausalFlash) || do_causal;
                let fallback_mask = eager_attention_mask(
                    q.dim(2)?,
                    k.dim(2)?,
                    causal,
                    sdpa_params.sliding_window,
                    q.dtype(),
                    q.device(),
                )?;
                return self.run_attention_noflash(
                    q,
                    k,
                    v,
                    fallback_mask.as_ref(),
                    sdpa_params,
                    causal,
                );
            }

            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;

            if q.device().is_cpu() {
                return run_flash_attn_cpu_for_dtype(&q, &k, &v, None, sdpa_params);
            } else {
                return flash_attn(&q, &k, &v, flash_params, sdpa_params)?.transpose(1, 2);
            }
        }

        self.run_attention_noflash(q, k, v, None, sdpa_params, do_causal)
    }

    /// Same as `run_attention`, but skips the flash-attention dispatch.
    ///
    /// `causal` tells the Metal SDPA-full kernel to enable its upper-triangle skip (`do_causal=true`).
    /// Pass `true` only when the caller's mask is causal-or-stricter.
    /// Pass false` for bidirectional masks (e.g. vision attention).
    #[allow(unused_variables, clippy::too_many_arguments)]
    pub fn run_attention_noflash(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        sdpa_params: &SdpaParams,
        causal: bool,
    ) -> Result<Tensor> {
        let (b_sz, n_attn_heads, seq_len, head_dim) = q.dims4()?;
        let (_, _, _, k_head_dim) = k.dims4()?;
        let (_, _, _, v_head_dim) = v.dims4()?;

        // We can use Metal SDPA (vector/full) if the mask is the correct size and head dims match.
        // If the mask is provided, then softcapping isn't allowed - default back to naive SDPA
        // Softcapping is implemented for vector SDPA.
        let all_head_dims_match = head_dim == k_head_dim && k_head_dim == v_head_dim;
        let tgt_mask_shape = vec![b_sz, n_attn_heads, seq_len, k.dim(2)?];
        let can_use_mask = mask.is_none_or(|mask| {
            mask.layout().broadcast_as(tgt_mask_shape.clone()).is_ok()
                && sdpa_params.softcap.is_none_or(|x| x == 1.0)
        });
        let valid_head_dims: &[usize] = &[32, 64, 72, 80, 96, 128, 256, 512];
        // Metal SDPA full kernel requires q_seq <= k_seq when a mask is present.
        let metal_supports_mask = mask.is_none() || seq_len <= k.dim(2)?;

        // Metal FA path for DK=512 BF16 with a mask. Two specializations:
        // prefill (seq_len > 8) goes through the BlockMMA kernel; decode
        // (seq_len == 1) uses a vector FA kernel ported from llama.cpp.
        if [q, k, v].into_iter().all(|x| x.device().is_metal())
            && head_dim == 512
            && k_head_dim == 512
            && v_head_dim == 512
            && q.dtype() == DType::BF16
            && k.dtype() == DType::BF16
            && v.dtype() == DType::BF16
            && seq_len == 1
            && mask.is_some()
            && sdpa_params.softcap.is_none_or(|x| x == 1.0)
        {
            if let Some(out) =
                crate::attention::backends::metal_flash_attn::try_flash_attn_ext_vec_bf16_dk512(
                    q,
                    k,
                    v,
                    mask,
                    sdpa_params.softmax_scale,
                )?
            {
                return Ok(out);
            }
        }
        if [q, k, v].into_iter().all(|x| x.device().is_metal())
            && head_dim == 512
            && k_head_dim == 512
            && v_head_dim == 512
            && q.dtype() == DType::BF16
            && k.dtype() == DType::BF16
            && v.dtype() == DType::BF16
            && seq_len > 8
            && sdpa_params.softcap.is_none_or(|x| x == 1.0)
        {
            if let Some(mask) = mask {
                if let Some(out) =
                    crate::attention::backends::metal_flash_attn::try_flash_attn_ext_bf16_dk512(
                        q,
                        k,
                        v,
                        mask,
                        sdpa_params.softmax_scale,
                    )?
                {
                    return Ok(out);
                }
            }
        }

        if [q, k, v].into_iter().all(|x| x.device().is_metal())
            && all_head_dims_match
            && valid_head_dims.contains(&head_dim)
            && can_use_mask
            && metal_supports_mask
            && !(head_dim == 512 && seq_len > 8)
        {
            let mask = match mask {
                Some(mask) => Some(mask.broadcast_as(tgt_mask_shape)?),
                None => None,
            };
            // do_causal lets the steel_attention kernel bound its kb-loop to
            // the per-query position, skipping the upper triangle of Q*K^T
            // entirely (roughly halves matmul cost for prefill).
            let do_causal = seq_len > 1 && causal;
            return candle_nn::ops::sdpa(
                q,
                k,
                v,
                mask.as_ref(),
                do_causal,
                sdpa_params.softmax_scale,
                sdpa_params.softcap.unwrap_or(1.0),
            );
        }

        // TITAN_ATTN_F32REF=1 (debug, CUDA): decode rows in f32 end to end (the grouped path on f32 copies),
        // the reference that the bf16 cuBLASLt path and the flash-decode kernel are compared against
        if seq_len == 1
            && mask.is_none()
            && q.device().is_cuda()
            && sdpa_params.n_kv_groups > 1
            && sdpa_params.softcap.is_none()
            && f32_reference_attention()
        {
            let f = |t: &Tensor| t.to_dtype(DType::F32);
            return gqa_grouped_sdpa(&f(q)?, &f(k)?, &f(v)?, None, sdpa_params)?.to_dtype(q.dtype());
        }

        // titan flash-decode: one decode row at or above TITAN_ATTN_FLASH_MIN keys (it sees them all)
        if mask.is_none() && seq_len == 1 {
            let kv_len = k.dim(2)?;
            if kv_len > 0 && flash_decode_first_row(q, k, v, kv_len - 1, 1, sdpa_params) == 0 {
                return flash_decode_rows(q, k, v, kv_len - 1, sdpa_params)?
                    .reshape((b_sz, n_attn_heads, 1, v_head_dim));
            }
        }

        #[cfg(feature = "cuda")]
        if let Some(y) = gqa_decode_nocopy(q, k, v, mask, sdpa_params, causal)? {
            return Ok(y);
        }

        if q.device().is_cuda()
            && sdpa_params.n_kv_groups > 1
            && k.dim(2)? >= GQA_GROUPED_MIN_KV
            && sdpa_params.softcap.is_none_or(|x| x == 1.0)
            && mask.is_none_or(|m| m.rank() == 2)
            && !mistralrs_quant::distributed::use_nccl()
        {
            return gqa_grouped_sdpa(q, k, v, mask, sdpa_params);
        }

        let k = repeat_kv(k.clone(), sdpa_params.n_kv_groups)?;
        let v = repeat_kv(v.clone(), sdpa_params.n_kv_groups)?;

        if mask.is_some_and(|x| x.rank() == 2) || mistralrs_quant::distributed::use_nccl() {
            return naive_sdpa(
                &q.contiguous()?,
                &k.contiguous()?,
                &v.contiguous()?,
                mask,
                sdpa_params,
            );
        }

        // TODO: bench?
        #[allow(unused)]
        if let (Device::Cuda(_), Some(cublaslt)) = (
            q.device(),
            mistralrs_quant::cublaslt::CUBLASLT_CONTROLLER.get_for_device(q.device()),
        ) {
            #[cfg(feature = "cuda")]
            {
                maybe_synchronize_for(q)?;

                // Use chunked attention for cuBLASLt path
                let k_flat = k.flatten(0, 1)?;
                let v_flat = v.flatten(0, 1)?;

                let kv_len = k.dim(2)?;
                let prefix_len = kv_len.saturating_sub(seq_len);
                chunked_attention_with_offset(
                    q,
                    &k,
                    &v,
                    mask,
                    |q_chunk, _k, _v, mask_chunk, q_offset| {
                        // cuBLASLt batch matmul implementation requires inputs to be dims3
                        let (chunk_b_sz, chunk_n_heads, chunk_seq_len, chunk_head_dim) =
                            q_chunk.dims4()?;
                        let q_flat = q_chunk.flatten(0, 1)?;

                        let attention_bias = match mask_chunk {
                            Some(mask) if mask.rank() == 3 && mask.dims()[0] == 1 => {
                                Some(mask.repeat((chunk_n_heads, 1, 1))?)
                            }
                            Some(mask) if mask.rank() == 3 => Some(mask.clone()),
                            Some(mask) if mask.rank() == 4 => {
                                let tgt_shape =
                                    vec![chunk_b_sz, chunk_n_heads, chunk_seq_len, k.dim(2)?];
                                Some(mask.broadcast_as(tgt_shape)?.flatten(0, 1)?)
                            }
                            Some(mask) => {
                                candle_core::bail!("cublaslt attn mask: rank must be 3 or 4")
                            }
                            None => None,
                        };

                        // If attention_bias is set, we fuse the add by giving it as the output matrix
                        // and setting beta to 1.0
                        let beta = match attention_bias.is_some() {
                            true => Some(1.0),
                            false => None,
                        };

                        // Batch matrix multiplication
                        // Fuse softmax scale and attention_bias add
                        let mut attention_scores = cublaslt.batch_matmul(
                            &k_flat,
                            &q_flat,
                            attention_bias.as_ref(),
                            Some(sdpa_params.softmax_scale / sdpa_params.softcap.unwrap_or(1.0)),
                            beta,
                            None,
                            None,
                        )?;
                        if let Some(softcap) = sdpa_params.softcap {
                            attention_scores = (attention_scores.tanh()? * softcap as f64)?;
                        }
                        // Compute softmax in F32 for precision. BF16's 7 mantissa
                        // bits cause exp() to lose information on long sequences.
                        // Flash attention already computes softmax in F32; this
                        // matches that behaviour for the eager path.
                        let scores_dtype = attention_scores.dtype();
                        if scores_dtype == DType::BF16 || scores_dtype == DType::F16 {
                            attention_scores = attention_scores.to_dtype(DType::F32)?;
                        }
                        if causal && mask_chunk.is_none() {
                            crate::ops::cuda_apply_causal_mask_f32(
                                &attention_scores,
                                q_offset,
                                prefix_len,
                            )?;
                        }
                        attention_scores = candle_nn::ops::softmax_last_dim(&attention_scores)?;
                        if attention_scores.dtype() != scores_dtype {
                            attention_scores = attention_scores.to_dtype(scores_dtype)?;
                        }

                        let context_layer = cublaslt.batch_matmul(
                            &v_flat.t()?.contiguous()?,
                            &attention_scores,
                            // We save one allocation
                            Some(&q_flat),
                            None,
                            None,
                            None,
                            None,
                        )?;

                        // Reshape to dims4
                        context_layer.reshape((
                            chunk_b_sz,
                            chunk_n_heads,
                            chunk_seq_len,
                            v_head_dim,
                        ))
                    },
                )
            }
            #[cfg(not(feature = "cuda"))]
            {
                candle_core::bail!("`cuda` feature is not enabled")
            }
        } else {
            naive_sdpa(q, &k, &v, mask, sdpa_params)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Result as CandleResult, D};

    const EPS: f32 = 1e-4;

    fn assert_close(lhs: &Tensor, rhs: &Tensor) -> CandleResult<()> {
        let lhs = lhs.flatten_all()?.to_vec1::<f32>()?;
        let rhs = rhs.flatten_all()?.to_vec1::<f32>()?;
        for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
            assert!((lhs - rhs).abs() < EPS, "{lhs} != {rhs}");
        }
        Ok(())
    }

    #[test]
    fn causal_flash_has_attention_intent_without_a_custom_tensor() {
        let mask = AttentionMask::CausalFlash;

        assert!(!mask.is_none());
        assert!(!mask.is_custom());
    }

    #[test]
    fn test_custom_cpu_mask_uses_attention_dispatch() -> CandleResult<()> {
        let (b, h, q_len, kv_len, d) = (1, 2, 3, 3, 4);
        let q = Tensor::from_vec(
            (0..b * h * q_len * d)
                .map(|x| x as f32 / 31.0)
                .collect::<Vec<_>>(),
            (b, h, q_len, d),
            &Device::Cpu,
        )?;
        let k = Tensor::from_vec(
            (0..b * h * kv_len * d)
                .map(|x| x as f32 / 37.0)
                .collect::<Vec<_>>(),
            (b, h, kv_len, d),
            &Device::Cpu,
        )?;
        let v = Tensor::from_vec(
            (0..b * h * kv_len * d)
                .map(|x| x as f32 / 41.0)
                .collect::<Vec<_>>(),
            (b, h, kv_len, d),
            &Device::Cpu,
        )?;
        let mask = Tensor::from_vec(
            vec![
                0.0,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0,
                0.0,
                f32::NEG_INFINITY,
                0.0,
                0.0,
                0.0,
            ],
            (q_len, kv_len),
            &Device::Cpu,
        )?;
        let sdpa_params = SdpaParams {
            n_kv_groups: 1,
            softcap: None,
            softmax_scale: 1.0,
            sliding_window: None,
            sinks: None,
        };

        let out = Sdpa.run_attention(
            &q,
            &k,
            &v,
            &AttentionMask::Custom(mask.clone()),
            Some(&FlashParams::empty(true)),
            &sdpa_params,
        )?;
        let logits = q.matmul(&k.transpose(2, 3)?)?.broadcast_add(&mask)?;
        let expected = candle_nn::ops::softmax(&logits, D::Minus1)?.matmul(&v)?;

        assert_eq!(out.shape().dims(), &[b, h, q_len, d]);
        assert_close(&out, &expected)
    }

    #[cfg(any(
        feature = "flash-attn",
        feature = "flash-attn-v3",
        all(feature = "cuda", target_family = "unix")
    ))]
    #[test]
    fn sliding_window_capacity_converts_to_left_distance() {
        assert_eq!(sliding_window_left(None), None);
        assert_eq!(sliding_window_left(Some(1)), Some(0));
        assert_eq!(sliding_window_left(Some(4096)), Some(4095));
    }

    #[test]
    fn eager_mask_combines_suffix_causality_and_sliding_capacity() -> CandleResult<()> {
        let mask = eager_attention_mask(3, 8, true, Some(4), DType::F32, &Device::Cpu)?
            .expect("causal sliding mask");
        let mask = mask.to_vec2::<f32>()?;

        for (row, values) in mask.iter().enumerate() {
            let query_pos = row + 5;
            for (key_pos, &value) in values.iter().enumerate() {
                let visible = key_pos <= query_pos && key_pos + 4 > query_pos;
                assert_eq!(value == 0.0, visible);
            }
        }
        Ok(())
    }

    #[test]
    fn eager_mask_unit_window_keeps_only_current_token() -> CandleResult<()> {
        let mask = eager_attention_mask(1, 3, true, Some(1), DType::F32, &Device::Cpu)?
            .expect("unit sliding mask");

        assert_eq!(
            mask.flatten_all()?.to_vec1::<f32>()?,
            vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0]
        );
        Ok(())
    }
}
