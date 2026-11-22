//! qwen4exp (Qwen3.8-Flash-Next) pieces on top of the qwen35moe trunk: the hyper-connection streams that replace the
//! layer norms, and the PLE n-gram hash embedding block. After llama.cpp `src/models/qwen4exp.cpp` and transformers
//! `modeling_qwen4_exp.py`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, Result, Tensor, D};
use mistralrs_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

use crate::gguf::Content;

pub(crate) const ARCH: &str = "qwen4exp";
/// Floor of |gate| before the signed square root of the PLE gate.
const PLE_GATE_FLOOR: f64 = 1e-6;
/// kvalues_iq4nl (ggml-common.h).
const KVALUES_IQ4NL: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];
const IQ4NL_BLOCK: usize = 32;
const IQ4NL_BYTES: usize = 18;

fn linear(q: candle_core::quantized::QTensor) -> Result<Arc<dyn QuantMethod>> {
    Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf { q_weight: Arc::new(q), b: None })?))
}

fn f32_tensor<R: std::io::Seek + std::io::Read>(ct: &mut Content<'_, R>, name: &str, device: &Device) -> Result<Tensor> {
    ct.tensor(name, device)?.dequantize(device)?.to_dtype(DType::F32)
}

/// RMS over each of the `hc` groups of the last dim, times `w` (`hc * e`, the converter's folded 1 + w).
fn grouped_rms(x: &Tensor, w: &Tensor, hc: usize, eps: f64) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let e = dims[dims.len() - 1] / hc;
    let mut g = dims[..dims.len() - 1].to_vec();
    g.extend([hc, e]);
    let xg = x.reshape(g)?;
    let inv = (xg.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?.recip()?;
    xg.broadcast_mul(&inv)?.reshape(dims)?.broadcast_mul(w)
}

/// One hyper-connection module: the sub-block input mixed from the `hc` streams, and (unless it is the final
/// mixer) the weights that scatter the sub-block output back into the streams.
pub(crate) struct HcMix {
    norm: Tensor,
    down: Arc<dyn QuantMethod>,
    up: Arc<dyn QuantMethod>,
    hc: usize,
    low_rank: usize,
    inject: bool,
    eps: f64,
}

impl HcMix {
    /// `prefix` `blk.N.hc_attn`, `blk.N.hc_ffn` (down and inject merged as `<prefix>_down_inject`) or `output_hc`.
    pub(crate) fn load<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        prefix: &str,
        hc: usize,
        low_rank: usize,
        eps: f64,
        device: &Device,
    ) -> Result<Self> {
        let norm = f32_tensor(ct, &format!("{prefix}_norm.weight"), device)?.flatten_all()?;
        let merged = format!("{prefix}_down_inject.weight");
        let (down, inject) = if ct.has_tensor(&merged) {
            (linear(ct.tensor(&merged, device)?)?, true)
        } else {
            if ct.has_tensor(&format!("{prefix}_inject.weight")) {
                candle_core::bail!("{prefix}: separate hc down / inject weights are not supported, re-convert with merges");
            }
            (linear(ct.tensor(&format!("{prefix}_down.weight"), device)?)?, false)
        };
        let up = linear(ct.tensor(&format!("{prefix}_up.weight"), device)?)?;
        Ok(Self { norm, down, up, hc, low_rank, inject, eps })
    }

    /// `x` `(b, t, hc * e)` f32 -> the mixed input `(b, t, e)` and the scatter weights `(b, t, hc)`.
    pub(crate) fn mix(&self, x: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let (b, t, hd) = x.dims3()?;
        let e = hd / self.hc;
        let xn = grouped_rms(x, &self.norm, self.hc, self.eps)?;
        let li = (self.down.forward(&xn)?.to_dtype(DType::F32)? / self.hc as f64)?;
        let lo = candle_nn::ops::silu(&li.narrow(2, 0, self.low_rank)?.contiguous()?)?;
        let gate = candle_nn::ops::sigmoid(&self.up.forward(&lo)?.to_dtype(DType::F32)?)?;
        let mixed = (xn * gate)?.reshape((b, t, self.hc, e))?.mean(2)?;
        let inject = if self.inject {
            let logits = li.narrow(2, self.low_rank, self.hc)?.contiguous()?;
            Some((candle_nn::ops::sigmoid(&logits)? * 2.0)?)
        } else {
            None
        };
        Ok((mixed, inject))
    }
}

/// `x + out * w` per stream: `x` `(b, t, hc * e)`, `out` `(b, t, e)`, `w` `(b, t, hc)`.
pub(crate) fn hc_combine(x: &Tensor, out: &Tensor, w: &Tensor) -> Result<Tensor> {
    let (b, t, hd) = x.dims3()?;
    let add = out.to_dtype(DType::F32)?.unsqueeze(2)?.broadcast_mul(&w.unsqueeze(3)?)?;
    x + add.reshape((b, t, hd))?
}

/// The `hc` copies of the embedding the streams start from: `(b, t, e)` -> `(b, t, hc * e)`.
pub(crate) fn hc_init(x: &Tensor, hc: usize) -> Result<Tensor> {
    x.to_dtype(DType::F32)?.repeat((1, 1, hc))
}

/// Per sequence slot: the last `ngram - 1` tokens and the conv history, both valid for the next position `pos`.
struct PleState {
    pos: usize,
    tokens: Vec<u32>,
    hist: Tensor,
}

/// PLE: each token gathers `n_heads` rows of the n-gram hash table (in the GGUF mmap), projects them to a value and
/// one key per stream, gates the value per stream, and adds it plus a dilated causal conv of its normed copy.
pub(crate) struct Ple {
    table: Arc<memmap2::Mmap>,
    table_off: usize,
    table_rows: usize,
    head_dim: usize,
    n_heads: usize,
    ngram: usize,
    per_gram: usize,
    eos: u32,
    multipliers: Vec<u64>,
    offsets: Vec<u64>,
    vocab: Vec<u64>,
    key: Arc<dyn QuantMethod>,
    value: Arc<dyn QuantMethod>,
    norm_key: Tensor,
    norm_query: Tensor,
    norm_conv: Tensor,
    /// `(hc * e, kernel)` f32: tap k reads `(kernel - 1 - k) * dilation` positions back
    conv: Tensor,
    kernel: usize,
    dilation: usize,
    hc: usize,
    eps: f64,
    state: Mutex<HashMap<usize, PleState>>,
}

/// Any GGUF integer value as u64 (the converter writes these arrays as I32, U32, I64 or U64).
pub(crate) fn int_u64(x: &candle_core::quantized::gguf_file::Value) -> Result<u64> {
    x.to_u64()
        .or_else(|_| x.to_u32().map(u64::from))
        .or_else(|_| x.to_i64().map(|v| v as u64))
        .or_else(|_| x.to_i32().map(|v| v as u64))
}

fn u64s(v: &candle_core::quantized::gguf_file::Value) -> Result<Vec<u64>> {
    v.to_vec()?.iter().map(int_u64).collect()
}

impl Ple {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        layer: usize,
        hc: usize,
        eps: f64,
        device: &Device,
    ) -> Result<Self> {
        let md = ct.get_metadata();
        let get = |k: &str| md.get(&format!("{ARCH}.{k}")).ok_or_else(|| candle_core::Error::msg(format!("{ARCH}: missing {k}")));
        let ngram = get("ple.ngram_size")?.to_u32()? as usize;
        let per_gram = get("ple.heads_per_ngram")?.to_u32()? as usize;
        let kernel = get("ple.conv_kernel")?.to_u32()? as usize;
        let eos = get("ple.eos_token_id")?.to_u32()?;
        let head_dim = get("embedding_length_per_layer_input")?.to_u32()? as usize;
        let multipliers = u64s(get("ple.layer_multipliers")?)?;
        let offsets = u64s(get("ple.head_offsets")?)?;
        let vocab = u64s(get("ple.head_vocab_sizes")?)?;
        let n_heads = (ngram - 1) * per_gram;
        if multipliers.len() < ngram || offsets.len() < n_heads || vocab.len() < n_heads {
            candle_core::bail!("{ARCH}: PLE hash constants shorter than {ngram} / {n_heads}");
        }
        let name = "per_layer_token_embd.weight";
        let info = ct.tensor_info(name)?;
        if info.ggml_dtype != GgmlDType::IQ4NL || head_dim % IQ4NL_BLOCK != 0 {
            candle_core::bail!("{ARCH}: PLE table {:?} rows of {head_dim}: only IQ4_NL is read in place", info.ggml_dtype);
        }
        let table_rows = info.shape.dims()[0];
        let (table, table_off) = ct
            .tensor_mapped(name)
            .ok_or_else(|| candle_core::Error::msg(format!("{ARCH}: the PLE table needs the GGUF memory-mapped")))?;
        let p = |s: &str| format!("blk.{layer}.{s}");
        let conv = f32_tensor(ct, &p("ple_conv1d.weight"), device)?; // GGUF {kernel, hc*e}: (hc*e, kernel)
        Ok(Self {
            table,
            table_off,
            table_rows,
            head_dim,
            n_heads,
            ngram,
            per_gram,
            eos,
            multipliers,
            offsets,
            vocab,
            key: linear(ct.tensor(&p("ple_key.weight"), device)?)?,
            value: linear(ct.tensor(&p("ple_value.weight"), device)?)?,
            norm_key: f32_tensor(ct, &p("ple_norm_key.weight"), device)?.flatten_all()?,
            norm_query: f32_tensor(ct, &p("ple_norm_query.weight"), device)?.flatten_all()?,
            norm_conv: f32_tensor(ct, &p("ple_norm_conv.weight"), device)?.flatten_all()?,
            conv,
            kernel,
            dilation: ngram,
            hc,
            eps,
            state: Mutex::new(HashMap::new()),
        })
    }

    fn history_len(&self) -> usize {
        (self.kernel - 1) * self.dilation
    }

    /// The `n_heads` table rows of each token: `toks` is the prefix context (oldest first) followed by the new tokens.
    fn rows(&self, ctx: &[u32], new: usize) -> Vec<u32> {
        let n_prev = self.ngram - 1;
        let base = ctx.len() - new;
        let mut idx = Vec::with_capacity(new * self.n_heads);
        for i in 0..new {
            let p = base + i;
            let mut gram = vec![self.eos as u64; self.ngram];
            gram[0] = ctx[p] as u64;
            let mut cut = false;
            for s in 1..=n_prev {
                let t = if p >= s && !cut { Some(ctx[p - s]) } else { None };
                cut = cut || t.is_none() || t == Some(self.eos);
                gram[s] = if cut { self.eos as u64 } else { t.unwrap_or(self.eos) as u64 };
            }
            for n in 2..=self.ngram {
                let mut mixed = gram[0].wrapping_mul(self.multipliers[0]);
                for j in 1..n {
                    mixed ^= gram[j].wrapping_mul(self.multipliers[j]);
                }
                let h0 = (n - 2) * self.per_gram;
                for h in h0..h0 + self.per_gram {
                    idx.push((mixed % self.vocab[h] + self.offsets[h]) as u32);
                }
            }
        }
        idx
    }

    /// IQ4_NL rows of the mapped table, dequantized: `(rows.len(), head_dim)` flattened.
    fn gather(&self, rows: &[u32]) -> Result<Vec<f32>> {
        let row_bytes = self.head_dim / IQ4NL_BLOCK * IQ4NL_BYTES;
        let mut out = vec![0f32; rows.len() * self.head_dim];
        for (r, o) in rows.iter().zip(out.chunks_exact_mut(self.head_dim)) {
            if *r as usize >= self.table_rows {
                candle_core::bail!("{ARCH}: PLE row {r} past the table ({})", self.table_rows);
            }
            let src = &self.table[self.table_off + *r as usize * row_bytes..][..row_bytes];
            for (blk, ob) in src.chunks_exact(IQ4NL_BYTES).zip(o.chunks_exact_mut(IQ4NL_BLOCK)) {
                let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                for j in 0..16 {
                    let q = blk[2 + j];
                    ob[j] = d * KVALUES_IQ4NL[(q & 0xF) as usize] as f32;
                    ob[j + 16] = d * KVALUES_IQ4NL[(q >> 4) as usize] as f32;
                }
            }
        }
        Ok(out)
    }

    /// `x + ple(x)` for one sequence: `x` `(1, t, hc * e)`, `tokens` its `t` new ids at positions `pos..pos + t` of
    /// recurrent slot `slot`.
    pub(crate) fn forward(&self, x: &Tensor, tokens: &[u32], slot: usize, pos: usize) -> Result<Tensor> {
        let (b, t, hd) = x.dims3()?;
        if b != 1 || tokens.len() != t {
            candle_core::bail!("{ARCH}: PLE runs one sequence at a time, got {:?} for {} tokens", x.dims(), tokens.len());
        }
        let e = hd / self.hc;
        let n_prev = self.ngram - 1;
        let hist_len = self.history_len();
        let mut states = self.state.lock().expect("ple state poisoned");
        let prior = states.remove(&slot).filter(|s| s.pos == pos && pos > 0);
        if prior.is_none() && pos > 0 {
            tracing::warn!("{ARCH}: PLE has no history for slot {slot} at position {pos}; the n-gram window and conv restart");
        }
        let (prev_tokens, hist) = match prior {
            Some(s) => (s.tokens, s.hist),
            None => (Vec::new(), Tensor::zeros((hist_len, hd), DType::F32, x.device())?),
        };
        let mut ctx_tokens = prev_tokens;
        ctx_tokens.extend_from_slice(tokens);
        let rows = self.rows(&ctx_tokens, t);
        let emb = Tensor::from_vec(self.gather(&rows)?, (1, t, self.n_heads * self.head_dim), x.device())?;

        let key = grouped_rms(&self.key.forward(&emb)?.to_dtype(DType::F32)?, &self.norm_key, self.hc, self.eps)?;
        let value = self.value.forward(&emb)?.to_dtype(DType::F32)?; // (1, t, e)
        let query = grouped_rms(x, &self.norm_query, self.hc, self.eps)?;
        let s = ((key * query)?.reshape((1, t, self.hc, e))?.sum(D::Minus1)? / (e as f64).sqrt())?; // (1, t, hc)
        let mag = s.abs()?.maximum(PLE_GATE_FLOOR)?.sqrt()?;
        let sign = ((s.ge(0f64)?.to_dtype(DType::F32)? * 2.0)? - 1.0)?;
        let gate = candle_nn::ops::sigmoid(&(sign * mag)?)?;
        let gated = value.unsqueeze(2)?.broadcast_mul(&gate.unsqueeze(3)?)?.reshape((1, t, hd))?;
        let normed = grouped_rms(&gated, &self.norm_conv, self.hc, self.eps)?.squeeze(0)?; // (t, hd)

        let padded = Tensor::cat(&[&hist, &normed], 0)?; // (hist + t, hd)
        let mut conv: Option<Tensor> = None;
        for k in 0..self.kernel {
            let start = hist_len - (self.kernel - 1 - k) * self.dilation;
            let term = padded.narrow(0, start, t)?.broadcast_mul(&self.conv.narrow(1, k, 1)?.t()?)?;
            conv = Some(match conv {
                Some(c) => (c + term)?,
                None => term,
            });
        }
        let conv = candle_nn::ops::silu(&conv.expect("kernel > 0"))?.unsqueeze(0)?;

        let keep = ctx_tokens.len().min(n_prev);
        let new_tokens = ctx_tokens[ctx_tokens.len() - keep..].to_vec();
        let new_hist = padded.narrow(0, padded.dim(0)? - hist_len, hist_len)?.contiguous()?;
        states.insert(slot, PleState { pos: pos + t, tokens: new_tokens, hist: new_hist });
        (x + gated)? + conv
    }
}
