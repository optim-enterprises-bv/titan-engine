#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! GGUF `qwen35` (dense), `qwen35moe` (Qwen3.5/3.6) and `qwen3next` (Qwen3-Next), after llama.cpp
//! `src/models/{qwen35,qwen35moe,qwen3next}.cpp` and `conversion/qwen.py`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor, D};
use mistralrs_quant::{GgufMatMul, QuantMethod, QuantMethodConfig, QuantizedConfig};

use super::quantized_qwen3_moe::{load_experts, Experts};
use crate::attention::{AttentionMask, SdpaParams};
use crate::device_map::{DeviceMappedMask, DeviceMapper};
use crate::titan_gdn::{GatedDeltaNet, GdnConfig, GdnLayerCache};
use crate::gguf::Content;
use crate::kv_cache::{
    HybridCache, HybridCacheConfig, HybridLayerCache, HybridLayerType, HybridPrefixEntry,
    HybridPrefixPoint, RecurrentLayerConfig, RecurrentStateSnapshot, RecurrentStateSpec,
    SingleCache,
};
use crate::layers::{CausalMaskConfig, CausalMasker, QRmsNorm, RotaryEmbedding, Sdpa};
use crate::layers_masker::PastKvLenCache;
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::{
    EitherCache, ForwardMaskCache, KvCache, ModelForwardContext, RecurrentBatchKind,
};
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};

const ARCH_DENSE: &str = "qwen35";
const ARCH_MOE: &str = "qwen35moe";
/// Same block as `qwen35moe`; its converter keeps the GDN V heads grouped and fuses in_proj_ba (`ssm_ba`).
const ARCH_NEXT: &str = "qwen3next";
const DEFAULT_MAX_SEQ_LEN: u64 = 4096;

pub(crate) struct PropsGGUF {
    pub head_count: usize,
    pub head_count_kv: usize,
    pub block_count: usize,
    pub embedding_length: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    pub rope_freq_base: f32,
    pub rope_dim: usize,
    pub head_dim: usize,
    /// None for the dense `qwen35`
    pub expert_used_count: Option<usize>,
    pub ssm_conv_kernel: usize,
    pub ssm_state_size: usize,
    pub ssm_group_count: usize,
    pub ssm_time_step_rank: usize,
    pub ssm_inner_size: usize,
    pub full_attention_interval: usize,
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> std::result::Result<Self, Self::Error> {
        c.verify_arch_any(&[ARCH_DENSE, ARCH_MOE, ARCH_NEXT])?;
        let is_moe = c.path_prefix != ARCH_DENSE;
        if is_moe {
            c.has_required_keys(&["expert_count", "expert_used_count"])?;
        }
        let required = [
            "attention.head_count",
            "attention.head_count_kv",
            "block_count",
            "embedding_length",
            "attention.layer_norm_rms_epsilon",
            "ssm.conv_kernel",
            "ssm.state_size",
            "ssm.group_count",
            "ssm.time_step_rank",
            "ssm.inner_size",
        ];
        c.has_required_keys(&required)?;

        let u = |k: &str| c.get_value::<u32>(k).map(|v| v as usize);
        let head_count = u("attention.head_count")?;
        let embedding_length = u("embedding_length")?;
        let head_dim = c
            .get_value::<u32>("attention.key_length")
            .map(|v| v as usize)
            .unwrap_or(embedding_length / head_count);
        let value_length = c
            .get_value::<u32>("attention.value_length")
            .map(|v| v as usize)
            .unwrap_or(head_dim);
        anyhow::ensure!(
            value_length == head_dim,
            "qwen35: expected key_length == value_length, got {head_dim} != {value_length}"
        );

        Ok(Self {
            head_count,
            head_count_kv: u("attention.head_count_kv")?,
            block_count: u("block_count")?,
            embedding_length,
            rms_norm_eps: c.get_value("attention.layer_norm_rms_epsilon")?,
            max_seq_len: c
                .get_value::<u64>("context_length")
                .unwrap_or(DEFAULT_MAX_SEQ_LEN) as usize,
            rope_freq_base: c.get_value("rope.freq_base").unwrap_or(10_000_000_f32),
            rope_dim: c
                .get_value::<u32>("rope.dimension_count")
                .map(|v| v as usize)
                .unwrap_or(head_dim / 4),
            head_dim,
            expert_used_count: is_moe.then(|| u("expert_used_count")).transpose()?,
            ssm_conv_kernel: u("ssm.conv_kernel")?,
            ssm_state_size: u("ssm.state_size")?,
            ssm_group_count: u("ssm.group_count")?,
            ssm_time_step_rank: u("ssm.time_step_rank")?,
            ssm_inner_size: u("ssm.inner_size")?,
            full_attention_interval: u("full_attention_interval").unwrap_or(4),
        })
    }
}

// llama.cpp: head_k_dim = ssm.state_size, n_k = ssm.group_count, n_v = ssm.time_step_rank
struct GdnCfg {
    hidden_size: usize,
    rms_norm_eps: f64,
    conv_kernel: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    quant: Option<QuantizedConfig>,
}

impl GdnConfig for GdnCfg {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn rms_norm_eps(&self) -> f64 {
        self.rms_norm_eps
    }
    fn linear_conv_kernel_dim(&self) -> usize {
        self.conv_kernel
    }
    fn linear_key_head_dim(&self) -> usize {
        self.head_k_dim
    }
    fn linear_value_head_dim(&self) -> usize {
        self.head_v_dim
    }
    fn linear_num_key_heads(&self) -> usize {
        self.num_k_heads
    }
    fn linear_num_value_heads(&self) -> usize {
        self.num_v_heads
    }
    fn quantization_config(&self) -> &Option<QuantizedConfig> {
        &self.quant
    }
}

/// The converter reorders GDN V heads grouped (kh * r + vi, what our kernels use) -> tiled (vi * n_k + kh).
fn v_head_perm(num_k_heads: usize, num_v_heads: usize) -> Vec<usize> {
    let r = num_v_heads / num_k_heads;
    (0..num_v_heads)
        .map(|h| (h % r) * num_k_heads + h / r)
        .collect()
}

/// Gather index turning a grouped-order V activation into the converter's tiled order:
/// `tiled[perm[h] * d + j] = grouped[h * d + j]`.
fn v_gather_index(perm: &[usize], d: usize) -> Vec<u32> {
    let mut src = vec![0u32; perm.len() * d];
    for (h, &slot) in perm.iter().enumerate() {
        for j in 0..d {
            src[slot * d + j] = (h * d + j) as u32;
        }
    }
    src
}

/// Unit `h` becomes unit `perm[h]`, `outer` times at `outer_stride`; raw bytes, so quant blocks move intact.
fn permute_units(
    data: &mut [u8],
    start: usize,
    outer: usize,
    outer_stride: usize,
    unit_bytes: usize,
    perm: &[usize],
) {
    let span = unit_bytes * perm.len();
    let mut tmp = vec![0u8; span];
    for o in 0..outer {
        let base = start + o * outer_stride;
        let region = &mut data[base..base + span];
        for (h, &g) in perm.iter().enumerate() {
            tmp[h * unit_bytes..(h + 1) * unit_bytes]
                .copy_from_slice(&region[g * unit_bytes..(g + 1) * unit_bytes]);
        }
        region.copy_from_slice(&tmp);
    }
}

/// llama.cpp qwen3next `mixed_ba`: per K head, `r` beta rows then `r` alpha rows. Returns the beta
/// and alpha row blocks, each in grouped V-head order (`kh * r + vi`).
fn split_ba(data: &[u8], row_bytes: usize, num_k_heads: usize, r: usize) -> (Vec<u8>, Vec<u8>) {
    let group = 2 * r * row_bytes;
    let (mut b, mut a) = (Vec::with_capacity(data.len() / 2), Vec::with_capacity(data.len() / 2));
    for kh in 0..num_k_heads {
        let g = &data[kh * group..(kh + 1) * group];
        b.extend_from_slice(&g[..r * row_bytes]);
        a.extend_from_slice(&g[r * row_bytes..]);
    }
    (b, a)
}

fn row_bytes(q: &QTensor) -> Result<usize> {
    let cols = q.shape().dims().last().copied().unwrap_or(1);
    let dt = q.dtype();
    if cols % dt.block_size() != 0 {
        candle_core::bail!("row of {cols} not a multiple of {dt:?} block size");
    }
    Ok(cols / dt.block_size() * dt.type_size())
}

enum VReorder {
    Rows { row0: usize, rows_per_head: usize },
    Cols { cols_per_head: usize },
}

fn is_identity(perm: &[usize]) -> bool {
    perm.iter().enumerate().all(|(i, &p)| i == p)
}

fn load_v_reordered<R: std::io::Seek + std::io::Read>(
    ct: &mut Content<'_, R>,
    name: &str,
    perm: &[usize],
    how: VReorder,
    device: &Device,
) -> Result<QTensor> {
    if is_identity(perm) {
        return ct.tensor(name, device);
    }
    let q = ct.tensor(name, &Device::Cpu)?;
    let dtype = q.dtype();
    let shape = q.shape().clone();
    // A 1-D tensor (ssm_a, ssm_dt.bias) has one head per element: each element is a row.
    let (rb, n_rows) = if shape.rank() == 1 && dtype.block_size() == 1 {
        (dtype.type_size(), shape.elem_count())
    } else {
        (row_bytes(&q)?, shape.elem_count() / shape.dims().last().copied().unwrap_or(1))
    };
    let mut data = q.data()?.into_owned();
    match how {
        VReorder::Rows {
            row0,
            rows_per_head,
        } => {
            if row0 + perm.len() * rows_per_head > n_rows {
                candle_core::bail!("{name}: V rows out of range");
            }
            permute_units(&mut data, row0 * rb, 1, 0, rows_per_head * rb, perm);
        }
        VReorder::Cols { cols_per_head } => {
            if cols_per_head % dtype.block_size() != 0 {
                candle_core::bail!(
                    "{name}: {dtype:?} blocks ({}) do not tile a V head of {cols_per_head} columns",
                    dtype.block_size()
                );
            }
            let unit = cols_per_head / dtype.block_size() * dtype.type_size();
            if unit * perm.len() != rb {
                candle_core::bail!("{name}: V columns do not span the row");
            }
            permute_units(&mut data, 0, n_rows, rb, unit, perm);
        }
    }
    QTensor::new(
        QStorage::from_data(std::borrow::Cow::Owned(data), device, dtype)?,
        shape,
    )
}

fn gguf_linear(q: QTensor) -> Result<Arc<dyn QuantMethod>> {
    Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
        q_weight: Arc::new(q),
        b: None,
    })?))
}

/// Quantized, in host memory: a dense F32 table would be 2 GB (248320 x 2048) of VRAM.
pub(crate) struct QEmbedding {
    data: Vec<u8>,
    dtype: GgmlDType,
    row_bytes: usize,
    vocab: usize,
    hidden: usize,
}

impl QEmbedding {
    pub(crate) fn new(q: QTensor) -> Result<Self> {
        let (vocab, hidden) = q.shape().dims2()?;
        let row_bytes = row_bytes(&q)?;
        Ok(Self {
            data: q.data()?.into_owned(),
            dtype: q.dtype(),
            row_bytes,
            vocab,
            hidden,
        })
    }

    pub(crate) fn forward(&self, ids: &Tensor, device: &Device) -> Result<Tensor> {
        let dims = ids.dims().to_vec();
        let flat: Vec<u32> = ids.flatten_all()?.to_dtype(DType::U32)?.to_vec1()?;
        let mut rows = Vec::with_capacity(flat.len() * self.row_bytes);
        for &id in &flat {
            let id = id as usize;
            if id >= self.vocab {
                candle_core::bail!("token id {id} out of range for vocab {}", self.vocab);
            }
            rows.extend_from_slice(&self.data[id * self.row_bytes..(id + 1) * self.row_bytes]);
        }
        let q = QTensor::new(
            QStorage::from_data(std::borrow::Cow::Owned(rows), &Device::Cpu, self.dtype)?,
            (flat.len(), self.hidden),
        )?;
        let mut out_dims = dims;
        out_dims.push(self.hidden);
        q.dequantize(&Device::Cpu)?
            .reshape(out_dims)?
            .to_device(device)
    }
}

struct FullAttention {
    wq: Arc<dyn QuantMethod>,
    wk: Arc<dyn QuantMethod>,
    wv: Arc<dyn QuantMethod>,
    wo: Arc<dyn QuantMethod>,
    q_norm: QRmsNorm,
    k_norm: QRmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rotary: Arc<RotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    dtype: DType,
}

impl FullAttention {
    fn forward(
        &self,
        x: &Tensor,
        mask: &AttentionMask,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        // IMRoPE with text-only positions (t == h == w) reduces to partial NEOX RoPE
        let positions = ctx
            .text_positions(x.device(), x.dim(1)?)?
            .ok_or_else(|| candle_core::Error::msg("missing RoPE positions"))?
            .clone();
        let ctx = &*ctx;
        self.forward_with(x, mask, &positions, |q, k, v| match &self.paged_attn {
            Some(paged_attn) => match ctx.paged_layer(layer_idx) {
                Some(((key_cache, value_cache), input_metadata)) => paged_attn.forward(
                    q,
                    k,
                    v,
                    mask,
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    Some(ctx.flash_params()),
                ),
                None => {
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    assert!(!matches!(mask, AttentionMask::None));
                    paged_attn.forward(
                        q,
                        k,
                        v,
                        mask,
                        None,
                        None,
                        &input_metadata,
                        &self.sdpa_params,
                        Some(ctx.flash_params()),
                    )
                }
            },
            None => {
                let (k, v) = kv_cache.append(k, v)?;
                self.attend(q, &k, &v, mask, Some(ctx.flash_params()))
            }
        })
    }

    /// Attention over the whole cache `k`, `v` after the append: a masked prompt chunk long enough for the
    /// flash-prefill kernel takes it, everything else `Sdpa`.
    fn attend(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: &AttentionMask,
        flash_params: Option<&crate::pipeline::text_models_inputs_processor::FlashParams>,
    ) -> Result<Tensor> {
        let rows = q.dim(2)?;
        if matches!(mask, AttentionMask::Custom(_))
            && rows > MTP_CATCHUP_MAX_ROWS
            && crate::attention::flash_prefill_wanted(rows, k.dim(2)?)
            && crate::attention::flash_prefill_supported(q, k, v, &self.sdpa_params)
        {
            return crate::attention::flash_prefill(q, k, v, &self.sdpa_params);
        }
        if let AttentionMask::Custom(t) = mask {
            if rows > 1 && t.elem_count() == 1 {
                candle_core::bail!("qwen35: placeholder mask for {rows} rows but flash-prefill did not take them");
            }
        }
        Sdpa.run_attention(q, k, v, mask, flash_params, &self.sdpa_params)
    }

    /// Eager attention over `kv_cache` at explicit RoPE `positions` (the MTP block, and one row of a
    /// speculative verification, which must match the decode step bit for bit).
    fn forward_eager(
        &self,
        x: &Tensor,
        mask: &AttentionMask,
        kv_cache: &mut KvCache,
        positions: &Tensor,
        flash_params: Option<&crate::pipeline::text_models_inputs_processor::FlashParams>,
    ) -> Result<Tensor> {
        self.forward_with(x, mask, positions, |q, k, v| {
            let (k, v) = kv_cache.append(k, v)?;
            self.attend(q, &k, &v, mask, flash_params)
        })
    }

    /// The rows `x` `(1, n, hidden)` of a verification at positions `start..start + n`, each row
    /// bit-identical to its decode step: batched row-exact projections and RoPE, then one decode-shaped
    /// attention per row over the cache prefix it would have seen.
    fn forward_decode_rows(
        &self,
        x: &Tensor,
        kv_cache: &mut KvCache,
        start: usize,
        flash_params: Option<&crate::pipeline::text_models_inputs_processor::FlashParams>,
    ) -> Result<Tensor> {
        let n = x.dim(1)?;
        let positions = crate::pipeline::text_positions_tensor(&[start], n, x.device())?;
        let (q, k, v, gate) = self.decode_rows_qkv(x, &positions)?;
        let y = self.decode_rows_attend(&q, &k, &v, kv_cache, flash_params)?;
        self.decode_rows_out(&y, &gate, x.dtype())
    }

    /// `forward_decode_rows` up to the attention: `(q, k, v, gate)` at RoPE `positions`.
    fn decode_rows_qkv(&self, x: &Tensor, positions: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let (b_sz, n, _) = x.dims3()?;
        let q_gate = lin_rows(&self.wq, x)?;
        let k = lin_rows(&self.wk, x)?;
        let v = lin_rows(&self.wv, x)?;
        let q_gate = q_gate.reshape((b_sz, n, self.n_head, self.head_dim * 2))?;
        let q = q_gate.narrow(D::Minus1, 0, self.head_dim)?;
        let gate = q_gate
            .narrow(D::Minus1, self.head_dim, self.head_dim)?
            .reshape((b_sz, n, self.n_head * self.head_dim))?;
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k
            .reshape((b_sz, n, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, n, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let (q, k) = self.rotary.forward_qk_norm(
            &q,
            &k,
            self.q_norm.weight(),
            self.k_norm.weight(),
            self.q_norm.eps(),
            self.k_norm.eps(),
            positions,
        )?;
        Ok((q.to_dtype(self.dtype)?, k.to_dtype(self.dtype)?, v.to_dtype(self.dtype)?, gate))
    }

    /// `forward_decode_rows`' attention: append the rows to the cache, one decode-shaped attention
    /// per row over the prefix it would have seen, `(b, n, heads * head_dim)`.
    fn decode_rows_attend(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        kv_cache: &mut KvCache,
        flash_params: Option<&crate::pipeline::text_models_inputs_processor::FlashParams>,
    ) -> Result<Tensor> {
        let (b_sz, _, n, _) = q.dims4()?;
        let (k, v) = kv_cache.append(k, v)?;
        let past = k.dim(2)? - n;
        // rows from `first` on (their key counts at or above TITAN_ATTN_FLASH_MIN) share one
        // flash-decode call; each still equals its decode step (the kernel is row-exact)
        let first = crate::attention::flash_decode_first_row(q, &k, &v, past, n, &self.sdpa_params);
        let mut ys = (0..first)
            .map(|r| {
                let q = q.narrow(2, r, 1)?.contiguous()?;
                let k = k.narrow(2, 0, past + r + 1)?;
                let v = v.narrow(2, 0, past + r + 1)?;
                Sdpa.run_attention(&q, &k, &v, &AttentionMask::None, flash_params, &self.sdpa_params)?
                    .reshape((b_sz, 1, ()))
            })
            .collect::<Result<Vec<_>>>()?;
        if first < n {
            ys.push(crate::attention::flash_decode_rows(
                &q.narrow(2, first, n - first)?,
                &k,
                &v,
                past + first,
                &self.sdpa_params,
            )?);
        }
        Tensor::cat(&ys, 1)
    }

    /// `forward_decode_rows` after the attention `y`: the output gate and `wo`.
    fn decode_rows_out(&self, y: &Tensor, gate: &Tensor, x_dtype: DType) -> Result<Tensor> {
        let gate = candle_nn::ops::sigmoid(&gate.to_dtype(y.dtype())?)?;
        let y = y.broadcast_mul(&gate)?;
        lin_rows(&self.wo, &y.to_dtype(x_dtype)?)
    }

    fn forward_with(
        &self,
        x: &Tensor,
        mask: &AttentionMask,
        positions: &Tensor,
        attend: impl FnOnce(&Tensor, &Tensor, &Tensor) -> Result<Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        // attn_q is [q | gate] per head, like Qwen3-Next
        let q_gate = self.wq.forward(x)?;
        let k = self.wk.forward(x)?;
        let v = self.wv.forward(x)?;

        let q_gate = q_gate.reshape((b_sz, seq_len, self.n_head, self.head_dim * 2))?;
        let q = q_gate.narrow(D::Minus1, 0, self.head_dim)?;
        let gate = q_gate
            .narrow(D::Minus1, self.head_dim, self.head_dim)?
            .reshape((b_sz, seq_len, self.n_head * self.head_dim))?;

        let (q, k, v) = if seq_len != 1 {
            let q = q.transpose(1, 2)?.contiguous()?;
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

        let (q, k) = self.rotary.forward_qk_norm(
            &q,
            &k,
            self.q_norm.weight(),
            self.k_norm.weight(),
            self.q_norm.eps(),
            self.k_norm.eps(),
            positions,
        )?;
        let (q, k, v) = (
            q.to_dtype(self.dtype)?,
            k.to_dtype(self.dtype)?,
            v.to_dtype(self.dtype)?,
        );

        let y = attend(&q, &k, &v)?;

        let y = if !matches!(mask, AttentionMask::None) {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };
        let gate = candle_nn::ops::sigmoid(&gate.to_dtype(y.dtype())?)?;
        let y = y.broadcast_mul(&gate)?;
        self.wo.forward(&y.to_dtype(x.dtype())?)
    }

    /// `forward_with` of one decode row, up to the attention: `(q, k, v, gate)`, the same ops.
    #[cfg(feature = "cuda")]
    fn decode_qkv(&self, x: &Tensor, positions: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let (b_sz, seq_len, _) = x.dims3()?;
        if seq_len != 1 {
            candle_core::bail!("qwen35 decode_qkv: one row, got {seq_len}");
        }
        let q_gate = self.wq.forward(x)?;
        let k = self.wk.forward(x)?;
        let v = self.wv.forward(x)?;
        let q_gate = q_gate.reshape((b_sz, seq_len, self.n_head, self.head_dim * 2))?;
        let q = q_gate.narrow(D::Minus1, 0, self.head_dim)?;
        let gate = q_gate
            .narrow(D::Minus1, self.head_dim, self.head_dim)?
            .reshape((b_sz, seq_len, self.n_head * self.head_dim))?;
        let q = q.reshape((b_sz, self.n_head, seq_len, self.head_dim))?;
        let k = k.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;
        let v = v.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;
        let (q, k) = self.rotary.forward_qk_norm(
            &q,
            &k,
            self.q_norm.weight(),
            self.k_norm.weight(),
            self.q_norm.eps(),
            self.k_norm.eps(),
            positions,
        )?;
        Ok((q.to_dtype(self.dtype)?, k.to_dtype(self.dtype)?, v.to_dtype(self.dtype)?, gate))
    }

    /// `forward_with` of one decode row after the attention `y`: the output gate and `wo`.
    #[cfg(feature = "cuda")]
    fn decode_out(&self, y: &Tensor, gate: &Tensor, masked: bool, x_dtype: DType) -> Result<Tensor> {
        let (b_sz, seq_len, _) = gate.dims3()?;
        let y = if masked {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };
        let gate = candle_nn::ops::sigmoid(&gate.to_dtype(y.dtype())?)?;
        let y = y.broadcast_mul(&gate)?;
        self.wo.forward(&y.to_dtype(x_dtype)?)
    }
}

struct SharedExpert {
    gate: Arc<dyn QuantMethod>,
    up: Arc<dyn QuantMethod>,
    down: Arc<dyn QuantMethod>,
    /// `(hidden, 1)`; the shared expert output is scaled by sigmoid(x . w)
    gate_inp: Tensor,
}

struct MoeBlock {
    router: QMatMul,
    experts: Experts,
    shared: SharedExpert,
    top_k: usize,
}

impl MoeBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_lookahead(xs, &[])
    }

    /// `forward`, passing later layers' predicted routing to the experts (see `Experts::forward_lookahead`).
    fn forward_lookahead(&self, xs: &Tensor, preds: &[(&Experts, usize, Tensor)]) -> Result<Tensor> {
        let (b, s, hidden) = xs.dims3()?;
        let dtype = xs.dtype();
        let flat = xs.reshape(((), hidden))?.to_dtype(DType::F32)?;
        let num_tokens = flat.dim(0)?;

        let topk = self.route(&flat)?;
        let piece = prefill_moe_piece();
        if preds.is_empty() && piece > 0 && num_tokens > piece {
            // a big prompt chunk: the routed outputs are weighted and summed piece by piece
            let w = &topk.values;
            let routed = self.experts.forward_reduced(
                &flat.reshape((num_tokens, 1, hidden))?,
                &topk.indices,
                piece,
                |r0, y| self.routed_sum(y, &w.narrow(0, r0, y.dim(0)?)?),
            )?;
            return (routed + self.shared_out(&flat)?)?.reshape((b, s, hidden))?.to_dtype(dtype);
        }
        let routed = self.experts.forward_lookahead(
            &flat.reshape((num_tokens, 1, hidden))?,
            &topk.indices,
            preds,
        )?;
        self.combine(&flat, &routed, &topk.values)?
            .reshape((b, s, hidden))?
            .to_dtype(dtype)
    }

    /// `forward` for the rows of a speculative verification `(1, n, hidden)`: everything is computed
    /// one row at a time exactly as a decode step would, except the routed experts, whose kernels
    /// (candle indexed MoE, titan tiered GPU slots and CPU twin) compute each (token, expert) pair
    /// independently of the batch, so all rows share one call (and one tiered miss sync).
    fn forward_rows(&self, xs: &Tensor) -> Result<Tensor> {
        if mtp_verify_rowwise() {
            return self.forward_rows_single(xs);
        }
        let (_, n, hidden) = xs.dims3()?;
        let dev = xs.device().clone();
        let mut t = std::time::Instant::now();
        let dtype = xs.dtype();
        let (flat, topk, gates) = self.route_rows(xs)?;
        prof("m.route", &dev, &mut t);
        let routed = self.experts.forward(&flat.reshape((n, 1, hidden))?, &topk.indices)?;
        prof("m.experts", &dev, &mut t);
        self.finish_rows(&flat, &routed, &topk.values, &gates, dtype)
    }

    /// `forward_rows` before the routed experts: `(flat, top-k, shared-expert gates)`.
    fn route_rows(&self, xs: &Tensor) -> Result<(Tensor, crate::ops::TopKOutput, Tensor)> {
        let (_, n, hidden) = xs.dims3()?;
        let flat = xs.reshape((n, hidden))?.to_dtype(DType::F32)?;
        // cuBLAS picks its F32 kernel by shape (a batch of m = 1 problems is not the decode kernel
        // either): the router and shared-expert gate matmuls run per row; top-k is one warp per row.
        let per_row = |f: &dyn Fn(&Tensor) -> Result<Tensor>| -> Result<Tensor> {
            Tensor::cat(&(0..n).map(|r| f(&flat.narrow(0, r, 1)?)).collect::<Result<Vec<_>>>()?, 0)
        };
        let topk = self.route_logits(&per_row(&|row| self.router.forward(row))?)?;
        let gates = candle_nn::ops::sigmoid(&per_row(&|row| row.matmul(&self.shared.gate_inp))?)?;
        Ok((flat, topk, gates))
    }

    /// `forward_rows` after the routed experts: weighted sum plus the gated shared expert, `(1, n, hidden)`.
    fn finish_rows(&self, flat: &Tensor, routed: &Tensor, weights: &Tensor, gates: &Tensor, dtype: DType) -> Result<Tensor> {
        let (n, hidden) = flat.dims2()?;
        // fast_sum reduces each output element in one block sized by the reduced length only
        let routed = routed
            .broadcast_mul(&weights.to_dtype(routed.dtype())?.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?
            .to_dtype(DType::F32)?;
        let sh = &self.shared;
        let g = lin_rows(&sh.gate, flat)?;
        let u = lin_rows(&sh.up, flat)?;
        let shared = lin_rows(
            &sh.down,
            &crate::ops::mul_and_act(&g, &u, crate::layers::Activation::Silu)?,
        )?;
        let shared = shared.to_dtype(DType::F32)?.broadcast_mul(gates)?;
        (routed + shared)?.reshape((1, n, hidden))?.to_dtype(dtype)
    }

    /// `forward_rows` with every op but the routed experts on one row at a time (debug reference).
    fn forward_rows_single(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, n, hidden) = xs.dims3()?;
        let dtype = xs.dtype();
        let mut flats = Vec::with_capacity(n);
        let mut topks = Vec::with_capacity(n);
        for r in 0..n {
            let flat = xs.narrow(1, r, 1)?.reshape(((), hidden))?.to_dtype(DType::F32)?;
            topks.push(self.route(&flat)?);
            flats.push(flat);
        }
        let routed_rows: Vec<Tensor> = if mtp_rowwise_experts() {
            flats
                .iter()
                .zip(&topks)
                .map(|(f, t)| self.experts.forward(&f.reshape((1, 1, hidden))?, &t.indices))
                .collect::<Result<_>>()?
        } else {
            let indices = Tensor::cat(&topks.iter().map(|t| &t.indices).collect::<Vec<_>>(), 0)?;
            let all = Tensor::cat(&flats, 0)?.reshape((n, 1, hidden))?;
            let routed = self.experts.forward(&all, &indices)?;
            (0..n).map(|r| routed.narrow(0, r, 1)).collect::<Result<_>>()?
        };
        let rows = (0..n)
            .map(|r| {
                self.combine(&flats[r], &routed_rows[r], &topks[r].values)?
                    .reshape((1, 1, hidden))?
                    .to_dtype(dtype)
            })
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&rows, 1)
    }

    fn route(&self, flat: &Tensor) -> Result<crate::ops::TopKOutput> {
        self.route_logits(&self.router.forward(flat)?)
    }

    /// `forward` of a prompt chunk through fully resident (stock) experts with llama.cpp's grouped MMQ and a
    /// device-side dispatch, instead of one indexed matvec per (token, expert); `None` for other expert sets.
    #[cfg(feature = "cuda")]
    fn forward_grouped_stock(&self, xs: &Tensor) -> Result<Option<Tensor>> {
        let Experts::Stock { gate: QMatMul::QTensor(g), up: QMatMul::QTensor(u), down: QMatMul::QTensor(d) } = &self.experts
        else {
            return Ok(None);
        };
        if !xs.device().is_cuda() || ![g, u, d].iter().all(|w| mistralrs_quant::supports_mmq(w.dtype())) {
            return Ok(None);
        }
        let (b, s, hidden) = xs.dims3()?;
        let dtype = xs.dtype();
        let flat = xs.reshape(((), hidden))?.to_dtype(DType::F32)?;
        let tokens = flat.dim(0)?;
        let topk = self.route(&flat)?;
        let k = topk.indices.dim(1)?;
        let total = tokens * k;
        let experts = g.shape().dims3()?.0;
        let dev = xs.device().as_cuda_device()?;
        let ids = topk.indices.flatten_all()?.contiguous()?;
        let (bounds, by_task, by_token) = {
            let (st, layout) = ids.storage_and_layout();
            let candle_core::Storage::Cuda(c) = &*st else {
                candle_core::bail!("qwen35 grouped experts: ids not on CUDA");
            };
            if layout.start_offset() != 0 {
                candle_core::bail!("qwen35 grouped experts: ids at an offset");
            }
            mistralrs_quant::moe_dispatch_build(c.as_cuda_slice::<u32>()?, total, experts, k, dev)?
        };
        use candle_core::cuda::cudarc::driver::CudaSlice;
        let proj = |w: &QTensor, x: &Tensor, src: &CudaSlice<u32>| {
            mistralrs_quant::grouped_moe_mmq(w, x, src, &by_task, &bounds, total, tokens, experts, dev)
        };
        let act = crate::ops::mul_and_act(
            &proj(&**g, &flat, &by_token)?,
            &proj(&**u, &flat, &by_token)?,
            crate::layers::Activation::Silu,
        )?;
        let routed = proj(&**d, &act, &by_task)?.reshape((tokens, k, hidden))?;
        Ok(Some(self.combine(&flat, &routed, &topk.values)?.reshape((b, s, hidden))?.to_dtype(dtype)?))
    }

    fn route_logits(&self, router_logits: &Tensor) -> Result<crate::ops::TopKOutput> {
        // llama.cpp build_moe_ffn: softmax gating, norm_w = true (HF norm_topk_prob)
        crate::ops::moe_router_topk(
            router_logits,
            crate::ops::MoeRouterTopKConfig {
                top_k: self.top_k,
                score_function: crate::ops::MoeRouterScoreFunction::Softmax,
                selected_weight: crate::ops::MoeRouterSelectedWeight::Score,
                renormalize: true,
                norm_min: 0.0,
                output_scale: 1.0,
                logit_clip: None,
            },
            None,
            None,
        )
    }

    /// Weighted sum of the routed experts plus the gated shared expert, `(tokens, hidden)` F32.
    fn combine(&self, flat: &Tensor, routed: &Tensor, weights: &Tensor) -> Result<Tensor> {
        let routed = self.routed_sum(routed, weights)?;
        routed + self.shared_out(flat)?
    }

    /// The routed experts' outputs `(tokens, k, hidden)` weighted and summed over k, `(tokens, hidden)` F32.
    fn routed_sum(&self, routed: &Tensor, weights: &Tensor) -> Result<Tensor> {
        expert_sum(&routed.broadcast_mul(&weights.to_dtype(routed.dtype())?.unsqueeze(D::Minus1)?)?)?.to_dtype(DType::F32)
    }

    /// The gated shared expert, `(tokens, hidden)` F32.
    fn shared_out(&self, flat: &Tensor) -> Result<Tensor> {
        let sh = &self.shared;
        let g = sh.gate.forward(flat)?;
        let u = sh.up.forward(flat)?;
        let shared = sh.down.forward(&crate::ops::mul_and_act(
            &g,
            &u,
            crate::layers::Activation::Silu,
        )?)?;
        let shared_gate = candle_nn::ops::sigmoid(&flat.matmul(&sh.gate_inp)?)?;
        shared.to_dtype(DType::F32)?.broadcast_mul(&shared_gate)
    }
}

struct DenseFfn {
    gate: Arc<dyn QuantMethod>,
    up: Arc<dyn QuantMethod>,
    down: Arc<dyn QuantMethod>,
}

enum Ffn {
    Moe(MoeBlock),
    Dense(DenseFfn),
}

impl Ffn {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Moe(moe) => moe.forward(xs),
            Self::Dense(ffn) => {
                let g = ffn.gate.forward(xs)?;
                let u = ffn.up.forward(xs)?;
                ffn.down.forward(&crate::ops::mul_and_act(
                    &g,
                    &u,
                    crate::layers::Activation::Silu,
                )?)
            }
        }
    }
}

impl Ffn {
    /// Rows of a speculative verification, each bit-identical to its decode step (see `MoeBlock::forward_rows`).
    fn forward_rows(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Moe(moe) => moe.forward_rows(xs),
            Self::Dense(_) => {
                let rows = (0..xs.dim(1)?)
                    .map(|r| self.forward(&xs.narrow(1, r, 1)?))
                    .collect::<Result<Vec<_>>>()?;
                Tensor::cat(&rows, 1)
            }
        }
    }
}

enum Mixer {
    Attention(FullAttention),
    Linear(GatedDeltaNet),
}

struct DecoderLayer {
    mixer: Mixer,
    attn_norm: QRmsNorm,
    post_attention_norm: QRmsNorm,
    ffn: Ffn,
}

pub struct ModelWeights {
    tok_embeddings: QEmbedding,
    layers: Vec<DecoderLayer>,
    norm: QRmsNorm,
    output: Arc<dyn QuantMethod>,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
    /// CUDA GDN kernels only take F16/BF16
    gdn_dtype: DType,
    /// The file's nextn (MTP) block, loaded when `TITAN_MTP` asks for drafts.
    mtp: Option<MtpBlock>,
    spec: Arc<Mutex<SpecState>>,
    /// Prefix cache: recurrent-state resume points of the running sequence of each recurrent slot,
    /// saved during its prompt passes (and carried over from the entry it resumed), collected with
    /// its caches when it finishes (`prefix_export`).
    prefix_points: Mutex<HashMap<usize, Vec<Arc<HybridPrefixPoint>>>>,
    /// `TITAN_CUDA_GRAPHS=1`: the captured decode segments (see `forward_trunk_graphed`).
    #[cfg(feature = "cuda")]
    graphs: super::titan_graph::SegmentGraphs,
}

/// `TITAN_MTP=<n>`: draft `n` tokens per step with the GGUF's nextn (multi-token prediction) block.
/// Read at load time: the block's VRAM is planned (gguf_metadata) and it is loaded only when set.
pub(crate) fn mtp_draft_len() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("TITAN_MTP")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
            .min(8)
    })
}

/// `TITAN_PREFILL_CHUNK=<tokens>`: unpaged prompt passes longer than this run as consecutive chunks
/// of this many tokens (0: never chunk). Bounds the prompt pass's activations, which otherwise grow
/// with the prompt. At least `MIN_PREFILL_CHUNK`.
pub(crate) fn prefill_chunk_size() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        match std::env::var("TITAN_PREFILL_CHUNK").ok().and_then(|v| v.parse::<usize>().ok()) {
            Some(0) => 0,
            Some(n) => n.max(MIN_PREFILL_CHUNK),
            None => DEFAULT_PREFILL_CHUNK,
        }
    })
}

/// 512: the prompt pass's activations (MoE rows, attention scores, GDN buffers) stay small next to
/// the 1 GiB tiered reserve; 1024 was ~3% faster to prefill but failed 13k-token prompts at 16k context
/// and 30k-token prompts at 32k context once decode steps had fragmented the pool.
const DEFAULT_PREFILL_CHUNK: usize = 512;
/// Granularity (tokens) of the KV room a chunked prompt reserves up front (also `titan_admit`'s).
pub(crate) const KV_RESERVE_STEP: usize = 8192;
/// The GDN prefill switches recurrence kernels below 64 rows (gdn::backend
/// RECURRENCE_CHUNK_THRESHOLD); every chunk stays at or above it so a chunk runs the same
/// sequential per-token warp kernel as the whole prompt.
const MIN_PREFILL_CHUNK: usize = 64;

/// `(start, len)` chunks of `c` tokens covering `0..l`; a tail shorter than `MIN_PREFILL_CHUNK` joins
/// the chunk before it.
fn prefill_chunk_bounds(l: usize, c: usize) -> Vec<(usize, usize)> {
    let mut bounds = (0..l).step_by(c).map(|s| (s, c.min(l - s))).collect::<Vec<_>>();
    if bounds.len() > 1 && bounds.last().is_some_and(|b| b.1 < MIN_PREFILL_CHUNK) {
        let (_, tail) = bounds.pop().expect("tail");
        bounds.last_mut().expect("chunk").1 += tail;
    }
    bounds
}

/// Rows from which `expert_sum` adds the top-k slices as a tree instead of calling `sum`: fast_sum gives
/// every output element its own block of top-k threads, ~2.5 ms per 512-token MoE layer.
const TREE_SUM_MIN_ROWS: usize = 64;

/// `p.sum(D::Minus2)` for weighted expert outputs `p` `(tokens, k, hidden)`: at `TREE_SUM_MIN_ROWS` rows and a
/// power-of-two k, the pairwise halving adds fast_sum's shared-memory tree does (`shr[t] += shr[t + s]` for
/// s = k/2, ..., 1), as whole-tensor adds; the same f32 sums in the same order.
fn expert_sum(p: &Tensor) -> Result<Tensor> {
    let (rows, k) = match p.dims() {
        &[rows, k, _] => (rows, k),
        _ => return p.sum(D::Minus2),
    };
    if rows < TREE_SUM_MIN_ROWS || !k.is_power_of_two() {
        return p.sum(D::Minus2);
    }
    let mut x = p.clone();
    let mut w = k;
    while w > 1 {
        w /= 2;
        x = (x.narrow(1, 0, w)? + x.narrow(1, w, w)?)?;
    }
    x.squeeze(1)
}

/// `TITAN_PREFILL_MOE_PIECE=<rows>` (default 512, 0 = off): a prompt chunk of more rows (a big chunk) runs its
/// MoE layers' down projections and expert sums in pieces of this many rows against one staged copy of the
/// experts (and the MTP block's experts piece by piece), so its MoE temporaries stay at the size of a
/// `TITAN_PREFILL_CHUNK` chunk while the expert copies are still amortised over the whole chunk.
fn prefill_moe_piece() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("TITAN_PREFILL_MOE_PIECE").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(512))
}

/// Rows from which an MTP pass (a prompt chunk) runs its experts as grouped MMQ (`forward_grouped_stock`); drafts
/// only steer acceptance, so the MMQ rounding is free to differ from the indexed matvecs.
const MTP_GROUPED_MIN_ROWS: usize = 128;

/// Most rows an MTP catch-up pass takes through the flash-decode rows path (the row-exact matvec limit).
const MTP_CATCHUP_MAX_ROWS: usize = 8;

/// `TITAN_PREFILL_BIG_CHUNK=<tokens>` (a multiple of `TITAN_PREFILL_CHUNK`, default 2048; 0 = off): the bulk of a
/// chunked prompt pass runs in chunks this large, the last stretch still in `TITAN_PREFILL_CHUNK` chunks, so every
/// boundary, and the resume point saved at the last one, stays on the `TITAN_PREFILL_CHUNK` grid. Big chunks copy
/// the streamed experts once per 2048 tokens instead of per 512; `prefill_moe_piece` keeps their MoE temporaries
/// at the 512-row size. Passes ending past `TITAN_PREFILL_BIG_MAX_CTX` tokens (0 = no limit) keep small chunks.
fn prefill_big_chunk() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        let c = prefill_chunk_size();
        let b = std::env::var("TITAN_PREFILL_BIG_CHUNK").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(DEFAULT_BIG_CHUNK);
        if c == 0 || b <= c {
            0
        } else {
            b / c * c
        }
    })
}

const DEFAULT_BIG_CHUNK: usize = 2048;
const DEFAULT_BIG_MAX_CTX: usize = 0;

/// `TITAN_PREFILL_BIG_MAX_CTX`: see `prefill_big_chunk`.
fn prefill_big_max_ctx() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("TITAN_PREFILL_BIG_MAX_CTX").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(DEFAULT_BIG_MAX_CTX)
    })
}

/// `prefill_chunk_bounds(l, c)` with its chunks before the last one merged into `big`-token chunks as far as
/// whole ones reach (`prefill_big_chunk`); the rest keeps `c`-token chunks.
fn prefill_grid_bounds(l: usize, c: usize, big: usize) -> Vec<(usize, usize)> {
    let small = prefill_chunk_bounds(l, c);
    if big <= c {
        return small;
    }
    let tail = small.last().map_or(0, |b| b.0);
    let mut bounds = Vec::new();
    let mut s = 0;
    while s + big <= tail {
        bounds.push((s, big));
        s += big;
    }
    bounds.extend(small.into_iter().filter(|b| b.0 >= s));
    bounds
}

/// `TITAN_PREFILL_SPLIT_AT=<pos>[,<pos>..]` (debug): a prompt pass from position 0 also splits at
/// each `pos` and restarts the chunk grid there, as a chain of prefix-cache resumes at those
/// positions runs it (the reference for comparing a resumed request with a cold one).
fn prefill_split_at() -> &'static [usize] {
    static S: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();
    S.get_or_init(|| {
        let mut v: Vec<usize> = std::env::var("TITAN_PREFILL_SPLIT_AT")
            .unwrap_or_default()
            .split(',')
            .filter_map(|x| x.trim().parse().ok())
            .filter(|&x| x > 0)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    })
}

/// The recurrent slot of a batch-1, unpaged prompt pass whose resume points the prefix cache keeps.
fn prefix_slot(ctx: &ModelForwardContext<'_>) -> Option<usize> {
    if ctx.is_paged()
        || !crate::prefix_cacher::hybrid_prefix_cache_on()
        || !matches!(ctx.recurrent_batch_kind(), Some(RecurrentBatchKind::Prefill))
    {
        return None;
    }
    match ctx.recurrent_metadata()?.state_indices_host()? {
        [slot] => Some(*slot as usize),
        _ => None,
    }
}

/// Host copy of the first `len` positions of a KV cache (`None` if it holds fewer).
fn kv_to_host(kv: &KvCache, len: usize) -> Result<Option<(Tensor, Tensor)>> {
    let KvCache::Normal { k, v } = kv else {
        return Ok(None);
    };
    let (Some(kd), Some(vd)) = (k.all_data(), v.all_data()) else {
        return Ok(None);
    };
    if len == 0 || k.current_seq_len() < len || v.current_seq_len() < len {
        return Ok(None);
    }
    let host = |t: &Tensor| t.narrow(2, 0, len)?.contiguous()?.to_device(&Device::Cpu);
    Ok(Some((host(kd)?, host(vd)?)))
}

/// A KV cache on `dev` holding the first `len` positions of a host copy, with room for `cap`.
fn kv_from_host(k: &Tensor, v: &Tensor, len: usize, cap: usize, max_seq_len: usize, dev: &Device) -> Result<KvCache> {
    let single = |t: &Tensor| -> Result<SingleCache> {
        let mut shape = t.dims().to_vec();
        shape[2] = cap;
        let buf = Tensor::zeros(shape, t.dtype(), dev)?;
        buf.slice_set(&t.narrow(2, 0, len)?.contiguous()?.to_device(dev)?, 2, 0)?;
        Ok(SingleCache {
            all_data: Some(buf),
            dim: 2,
            current_seq_len: len,
            capacity_seq_len: cap,
            max_seq_len,
        })
    };
    Ok(KvCache::Normal { k: single(k)?, v: single(v)? })
}

/// The device's stream-ordered allocation pool around prompt chunks.
mod cuda_pool {
    use candle_core::{Device, Result};

    /// `TITAN_PREFILL_MEMLOG=1`: before every prompt chunk (enqueued, not synchronised), log the
    /// pool's used / reserved / peak bytes.
    #[allow(unused_variables)]
    pub(super) fn between_chunks(dev: &Device, chunk: usize, start: usize) -> Result<()> {
        #[cfg(feature = "cuda")]
        if let Device::Cuda(d) = dev {
            use candle_core::cuda_backend::cudarc::driver::sys;
            if !std::env::var("TITAN_PREFILL_MEMLOG").is_ok_and(|v| v == "1") {
                return Ok(());
            }
            let stream = d.cuda_stream();
            let ctx = stream.context();
            if !ctx.has_async_alloc() {
                return Ok(());
            }
            ctx.bind_to_thread().map_err(candle_core::Error::wrap)?;
            let mut pool = std::ptr::null_mut();
            if unsafe { sys::cuDeviceGetMemPool(&mut pool, ctx.cu_device()) } != sys::CUresult::CUDA_SUCCESS {
                return Ok(());
            }
            let get = |attr| {
                let mut v = 0u64;
                unsafe { sys::cuMemPoolGetAttribute(pool, attr, (&mut v as *mut u64).cast()) };
                v >> 20
            };
            use sys::CUmemPool_attribute::*;
            tracing::info!(
                "titan prefill chunk {chunk} at {start}: pool used {} MiB, reserved {} MiB, used peak {} MiB, reserved peak {} MiB",
                get(CU_MEMPOOL_ATTR_USED_MEM_CURRENT),
                get(CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT),
                get(CU_MEMPOOL_ATTR_USED_MEM_HIGH),
                get(CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH)
            );
        }
        Ok(())
    }
}

/// `TITAN_TIERED_LOOKAHEAD=N`: predict the next N layers' routing during decode (P1 prefetch), 0 = off.
fn lookahead_depth() -> usize {
    static D: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *D.get_or_init(|| {
        std::env::var("TITAN_TIERED_LOOKAHEAD").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0).min(4)
    })
}

/// `TITAN_MTP_PROF=1`: device-synchronised phase timings of decode / verification forwards, logged
/// with the MTP stats (the syncs slow everything down; for relative costs only).
fn mtp_prof_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TITAN_MTP_PROF").is_ok_and(|v| v == "1"))
}

/// `TITAN_MTP_PROF=wall`: no syncs; wall time inside forward (`in.*`) and between forwards (`gap.*`,
/// sampling, verification, drafting and the engine loop), keyed by the forward's row count.
fn mtp_prof_wall() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TITAN_MTP_PROF").is_ok_and(|v| v == "wall"))
}

static LAST_EXIT: Mutex<Option<std::time::Instant>> = Mutex::new(None);

fn prof_add(name: &'static str, d: std::time::Duration) {
    let mut p = PROF.lock().unwrap();
    match p.iter_mut().find(|e| e.0 == name) {
        Some(e) => {
            e.1 += d;
            e.2 += 1;
        }
        None => p.push((name, d, 1)),
    }
}

fn prof_wall_enter(rows: usize) -> Option<std::time::Instant> {
    if !mtp_prof_wall() {
        return None;
    }
    let now = std::time::Instant::now();
    if let Some(last) = LAST_EXIT.lock().unwrap().take() {
        let gap = now - last;
        if gap < std::time::Duration::from_millis(500) {
            prof_add(["gap.0", "gap.1", "gap.2", "gap.3", "gap.4", "gap.5", "gap.6", "gap.7", "gap.8"][rows.min(8)], gap);
        }
    }
    Some(now)
}

fn prof_wall_exit(rows: usize, t: Option<std::time::Instant>) {
    if let Some(t) = t {
        prof_add(["in.0", "in.1", "in.2", "in.3", "in.4", "in.5", "in.6", "in.7", "in.8"][rows.min(8)], t.elapsed());
        *LAST_EXIT.lock().unwrap() = Some(std::time::Instant::now());
    }
}

static PROF: Mutex<Vec<(&'static str, std::time::Duration, u64)>> = Mutex::new(Vec::new());

/// Add the time since `*t` to phase `name` (after a device sync) and restart `*t`.
fn prof(name: &'static str, dev: &Device, t: &mut std::time::Instant) {
    if !mtp_prof_on() {
        return;
    }
    let _ = dev.synchronize();
    let d = t.elapsed();
    let mut p = PROF.lock().unwrap();
    match p.iter_mut().find(|e| e.0 == name) {
        Some(e) => {
            e.1 += d;
            e.2 += 1;
        }
        None => p.push((name, d, 1)),
    }
    *t = std::time::Instant::now();
}

fn prof_report() -> String {
    let p = PROF.lock().unwrap();
    p.iter()
        .map(|(n, d, c)| format!("{n} {:.2}ms x{c}", d.as_secs_f64() * 1e3 / *c as f64))
        .collect::<Vec<_>>()
        .join(", ")
}

const VERIFY_NAMES: [&str; 9] = ["v0", "verify1", "verify2", "verify3", "verify4", "verify5", "verify6", "verify7", "verify8"];

/// `TITAN_MTP_VERIFY=rowwise`: debug, run every verification op (but the routed experts) per row
/// through the plain decode path instead of the row-exact batched kernels.
fn mtp_verify_rowwise() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TITAN_MTP_VERIFY").is_ok_and(|v| v == "rowwise"))
}

/// `w @ x` for the rows of a verification, each row bit-identical to the decode step's matmul:
/// the batch-1 MMVQ reduction for all rows in one launch (`mistralrs_quant::mmvq_rows`), or one
/// plain forward per row where that kernel does not apply (e.g. F32 weights through cuBLAS).
fn lin_rows(w: &Arc<dyn QuantMethod>, x: &Tensor) -> Result<Tensor> {
    let k = x.dim(D::Minus1)?;
    let rows = x.elem_count() / k;
    if rows == 1 {
        return w.forward(x);
    }
    #[cfg(feature = "cuda")]
    if rows <= mistralrs_quant::mmvq_rows::MAX_ROWS && !w.has_bias() {
        if let Some(q) = w.get_qtensor() {
            if matches!(
                q.dtype(),
                GgmlDType::Q4_0
                    | GgmlDType::Q4_1
                    | GgmlDType::Q5_0
                    | GgmlDType::Q5_1
                    | GgmlDType::Q8_0
                    | GgmlDType::Q2K
                    | GgmlDType::Q3K
                    | GgmlDType::Q4K
                    | GgmlDType::Q5K
                    | GgmlDType::Q6K
            ) && mistralrs_quant::mmvq_rows::supports(&q, x)
            {
                return mistralrs_quant::mmvq_rows::plain_rows(&q, x);
            }
        }
    }
    let mut out_dims = x.dims().to_vec();
    let x = x.reshape((1, rows, k))?;
    let outs = (0..rows)
        .map(|r| w.forward(&x.narrow(1, r, 1)?))
        .collect::<Result<Vec<_>>>()?;
    let out = Tensor::cat(&outs, 1)?;
    *out_dims.last_mut().unwrap() = out.dim(D::Minus1)?;
    out.reshape(out_dims)
}

/// `TITAN_MTP_ROWWISE_EXPERTS=1`: debug, run the routed experts of a verification one row at a time.
fn mtp_rowwise_experts() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("TITAN_MTP_ROWWISE_EXPERTS").is_ok_and(|v| !v.is_empty() && v != "0")
    })
}

/// llama.cpp `graph_mtp` (qwen35moe.cpp): `eh_proj([enorm(embed(x_p)); hnorm(h_{p-1})])` feeds one
/// full-attention MoE block with its own KV cache at the token positions; `head_norm` then the
/// trunk's LM head give the draft logits, and the head-normed hidden seeds the next draft step.
struct MtpBlock {
    eh_proj: Arc<dyn QuantMethod>,
    enorm: QRmsNorm,
    hnorm: QRmsNorm,
    attn_norm: QRmsNorm,
    attn: FullAttention,
    post_attention_norm: QRmsNorm,
    ffn: Ffn,
    head_norm: QRmsNorm,
    n_draft: usize,
}

/// Speculation state shared with `speculative::unpaged` (the verifier rolls the trunk back through it).
pub(crate) struct SpecState {
    /// Final-normed trunk hidden rows of the last forward: (first position, `(rows, hidden)` F32).
    last: Option<(usize, Tensor)>,
    /// Trunk hidden at `pending.0`, needed as `h_{p-1}` for the next MTP row (after a prefill).
    pending: Option<(usize, Tensor)>,
    /// MTP KV cache and how many leading positions of it hold verified rows.
    mtp_kv: Option<KvCache>,
    mtp_len: usize,
    mtp_valid: bool,
    /// Recurrent states after each row but the last of the running verification.
    verify: Option<VerifyRecord>,
    stats: MtpStats,
}

struct VerifyRecord {
    start: usize,
    rows: usize,
    slot: usize,
    /// per GDN layer: (layer, [(conv_state, recurrent_state) after row r] for r < rows - 1)
    gdn: Vec<(usize, Vec<(Tensor, Tensor)>)>,
}

#[derive(Default)]
struct MtpStats {
    steps: u64,
    drafted: u64,
    accepted: u64,
    /// accepted[i]: steps whose draft i+1 was accepted
    by_pos: [u64; 8],
    draft_time: std::time::Duration,
}

/// A small dense tensor (norms, BF16 routers) dequantized to F32 on the host, then moved to `device`.
fn host_f32<R: std::io::Seek + std::io::Read>(
    ct: &mut Content<'_, R>,
    name: &str,
    device: &Device,
) -> Result<Tensor> {
    ct.tensor(name, &Device::Cpu)?
        .dequantize(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .to_device(device)
}

#[allow(clippy::too_many_arguments)]
fn load_full_attention<R: std::io::Seek + std::io::Read>(
    ct: &mut Content<'_, R>,
    prefix: &str,
    device: &Device,
    rotary: Arc<RotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    (head_count, head_count_kv, head_dim): (usize, usize, usize),
    rms_norm_eps: f32,
    dtype: DType,
) -> Result<FullAttention> {
    Ok(FullAttention {
        wq: gguf_linear(ct.tensor(&format!("{prefix}.attn_q.weight"), device)?)?,
        wk: gguf_linear(ct.tensor(&format!("{prefix}.attn_k.weight"), device)?)?,
        wv: gguf_linear(ct.tensor(&format!("{prefix}.attn_v.weight"), device)?)?,
        wo: gguf_linear(ct.tensor(&format!("{prefix}.attn_output.weight"), device)?)?,
        q_norm: QRmsNorm::new(
            ct.tensor(&format!("{prefix}.attn_q_norm.weight"), device)?,
            rms_norm_eps,
        )?,
        k_norm: QRmsNorm::new(
            ct.tensor(&format!("{prefix}.attn_k_norm.weight"), device)?,
            rms_norm_eps,
        )?,
        n_head: head_count,
        n_kv_head: head_count_kv,
        head_dim,
        rotary,
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

fn load_shared_expert<R: std::io::Seek + std::io::Read>(
    ct: &mut Content<'_, R>,
    prefix: &str,
    device: &Device,
) -> Result<SharedExpert> {
    let name = format!("{prefix}.ffn_gate_inp_shexp.weight");
    let gate_inp = if ct.tensor_info(&name)?.ggml_dtype == GgmlDType::BF16 {
        // the MTP block stores its routers in BF16; dequantize on the host
        host_f32(ct, &name, device)?
    } else {
        ct.tensor(&name, device)?
            .dequantize(device)?
            .to_dtype(DType::F32)?
    };
    let hidden = gate_inp.elem_count();
    Ok(SharedExpert {
        gate: gguf_linear(ct.tensor(&format!("{prefix}.ffn_gate_shexp.weight"), device)?)?,
        up: gguf_linear(ct.tensor(&format!("{prefix}.ffn_up_shexp.weight"), device)?)?,
        down: gguf_linear(ct.tensor(&format!("{prefix}.ffn_down_shexp.weight"), device)?)?,
        gate_inp: gate_inp.reshape((hidden, 1))?,
    })
}

#[allow(clippy::too_many_arguments)]
fn load_gdn<R: std::io::Seek + std::io::Read>(
    ct: &mut Content<'_, R>,
    prefix: &str,
    cfg: &GdnCfg,
    perm: &[usize],
    device: &Device,
    gdn_dtype: DType,
) -> Result<GatedDeltaNet> {
    let key_dim = cfg.num_k_heads * cfg.head_k_dim;
    let value_dim = cfg.num_v_heads * cfg.head_v_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let rows = |row0, rows_per_head| VReorder::Rows {
        row0,
        rows_per_head,
    };

    let in_proj_qkv = load_v_reordered(
        ct,
        &format!("{prefix}.attn_qkv.weight"),
        perm,
        rows(2 * key_dim, cfg.head_v_dim),
        device,
    )?;
    let in_proj_z = load_v_reordered(
        ct,
        &format!("{prefix}.attn_gate.weight"),
        perm,
        rows(0, cfg.head_v_dim),
        device,
    )?;
    if ct.has_tensor(&format!("{prefix}.ssm_in.weight")) {
        candle_core::bail!("{prefix}: legacy fused ssm_in (qkvz) GGUF; re-convert with a llama.cpp that writes attn_qkv / attn_gate");
    }
    // ssm_beta = in_proj_b, ssm_alpha = in_proj_a; qwen3next fuses them as ssm_ba
    let ba_name = format!("{prefix}.ssm_ba.weight");
    let (in_proj_b, in_proj_a) = if ct.has_tensor(&ba_name) {
        if !is_identity(perm) {
            candle_core::bail!("{prefix}: ssm_ba with reordered V heads is not a layout any converter writes");
        }
        let ba = ct.tensor(&ba_name, &Device::Cpu)?;
        let (b, a) = split_ba(&ba.data()?, row_bytes(&ba)?, cfg.num_k_heads, cfg.num_v_heads / cfg.num_k_heads);
        let hidden = ba.shape().dims2()?.1;
        let qt = |bytes: Vec<u8>| {
            QTensor::new(
                QStorage::from_data(std::borrow::Cow::Owned(bytes), device, ba.dtype())?,
                (cfg.num_v_heads, hidden),
            )
        };
        (qt(b)?, qt(a)?)
    } else {
        (
            load_v_reordered(ct, &format!("{prefix}.ssm_beta.weight"), perm, rows(0, 1), device)?,
            load_v_reordered(ct, &format!("{prefix}.ssm_alpha.weight"), perm, rows(0, 1), device)?,
        )
    };
    // ssm_out's input columns are V heads in the converter's tiled order. When its quant blocks tile a
    // head, move the column blocks back to grouped order (as for the other V tensors); otherwise (e.g. a
    // 256-wide Q4_K block spans two 128-wide heads) keep the weight as stored and gather the GDN output
    // into the tiled order at run time: tiled slot perm[h] holds grouped head h.
    let ssm_out_name = format!("{prefix}.ssm_out.weight");
    let ssm_out_block = ct.tensor_info(&ssm_out_name)?.ggml_dtype.block_size();
    let (out_proj, out_perm) = if is_identity(perm) {
        (ct.tensor(&ssm_out_name, device)?, None)
    } else if cfg.head_v_dim % ssm_out_block == 0 {
        let w = load_v_reordered(ct, &ssm_out_name, perm, VReorder::Cols { cols_per_head: cfg.head_v_dim }, device)?;
        (w, None)
    } else {
        let src = v_gather_index(perm, cfg.head_v_dim);
        let n = src.len();
        let index = Tensor::from_vec(src, n, device)?;
        (ct.tensor(&ssm_out_name, device)?, Some(index))
    };

    let conv1d = load_v_reordered(
        ct,
        &format!("{prefix}.ssm_conv1d.weight"),
        perm,
        rows(2 * key_dim, cfg.head_v_dim),
        device,
    )?
    .dequantize(device)?
    .reshape((conv_dim, 1, cfg.conv_kernel))?
    .to_dtype(gdn_dtype)?;
    let dt_bias = load_v_reordered(
        ct,
        &format!("{prefix}.ssm_dt.bias"),
        perm,
        rows(0, 1),
        device,
    )?
    .dequantize(device)?
    .to_dtype(DType::F32)?;
    // converter stores ssm_a = -exp(A_log); the backend wants A_log
    let a_log = load_v_reordered(ct, &format!("{prefix}.ssm_a"), perm, rows(0, 1), device)?
        .dequantize(device)?
        .to_dtype(DType::F32)?
        .neg()?
        .log()?;
    // the only norm the converter does not bake +1 into
    let norm_weight = ct
        .tensor(&format!("{prefix}.ssm_norm.weight"), device)?
        .dequantize(device)?
        .to_dtype(gdn_dtype)?;

    let gdn = GatedDeltaNet::from_parts(
        cfg,
        gguf_linear(in_proj_qkv)?,
        gguf_linear(in_proj_z)?,
        gguf_linear(in_proj_b)?,
        gguf_linear(in_proj_a)?,
        conv1d,
        dt_bias,
        a_log,
        norm_weight,
        gguf_linear(out_proj)?,
    );
    Ok(match out_perm {
        Some(index) => gdn.with_out_proj_input_perm(index),
        None => gdn,
    })
}

/// `{arch}.nextn_predict_layers`: MTP blocks appended after the trunk (0 when absent).
pub(crate) fn nextn_predict_layers(
    metadata: &HashMap<String, candle_core::quantized::gguf_file::Value>,
    arch: &str,
) -> usize {
    metadata
        .get(&format!("{arch}.nextn_predict_layers"))
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(0) as usize
}

#[allow(clippy::too_many_arguments)]
fn load_mtp_block<R: std::io::Seek + std::io::Read>(
    ct: &mut Content<'_, R>,
    prefix: &str,
    device: &Device,
    rotary: Arc<RotaryEmbedding>,
    heads: (usize, usize, usize),
    rms_norm_eps: f32,
    top_k: usize,
    dtype: DType,
) -> Result<MtpBlock> {
    let norm = |ct: &mut Content<'_, R>, name: &str| -> Result<QRmsNorm> {
        QRmsNorm::new(ct.tensor(&format!("{prefix}.{name}"), device)?, rms_norm_eps)
    };
    if ct.has_tensor(&format!("{prefix}.nextn.embed_tokens.weight"))
        || ct.has_tensor(&format!("{prefix}.nextn.shared_head_head.weight"))
    {
        candle_core::bail!("{prefix}: MTP blocks with their own embed_tokens / shared_head_head are not supported");
    }
    let attn = load_full_attention(ct, prefix, device, rotary, None, heads, rms_norm_eps, dtype)?;
    // The MTP experts stay stock (fully on the GPU): drafting never waits for a tiered miss.
    let exps = |ct: &mut Content<'_, R>, n: &str| -> Result<QMatMul> {
        QMatMul::from_qtensor(ct.tensor(&format!("{prefix}.{n}.weight"), device)?)
    };
    let experts = Experts::Stock {
        gate: exps(ct, "ffn_gate_exps")?,
        up: exps(ct, "ffn_up_exps")?,
        down: exps(ct, "ffn_down_exps")?,
    };
    let ffn = Ffn::Moe(MoeBlock {
        router: QMatMul::Tensor(host_f32(ct, &format!("{prefix}.ffn_gate_inp.weight"), device)?),
        experts,
        shared: load_shared_expert(ct, prefix, device)?,
        top_k,
    });
    let head_norm = if ct.has_tensor(&format!("{prefix}.nextn.shared_head_norm.weight")) {
        norm(ct, "nextn.shared_head_norm.weight")?
    } else {
        QRmsNorm::new(ct.tensor("output_norm.weight", device)?, rms_norm_eps)?
    };
    let block = MtpBlock {
        eh_proj: gguf_linear(ct.tensor(&format!("{prefix}.nextn.eh_proj.weight"), device)?)?,
        enorm: norm(ct, "nextn.enorm.weight")?,
        hnorm: norm(ct, "nextn.hnorm.weight")?,
        attn_norm: norm(ct, "attn_norm.weight")?,
        attn,
        post_attention_norm: norm(ct, "post_attention_norm.weight")?,
        ffn,
        head_norm,
        n_draft: mtp_draft_len(),
    };
    tracing::info!(
        "titan mtp: loaded {prefix} as the MTP draft head (stock experts on the GPU), {} draft tokens per step",
        block.n_draft
    );
    Ok(block)
}

impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>,
        device: &Device,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        attention_mechanism: AttentionImplementation,
        dtype: DType,
    ) -> Result<Self> {
        let arch = ct.arch().to_string();
        let metadata = ContentMetadata {
            path_prefix: &arch,
            metadata: ct.get_metadata(),
        };
        let PropsGGUF {
            head_count,
            head_count_kv,
            block_count,
            embedding_length,
            rms_norm_eps,
            max_seq_len,
            rope_freq_base,
            rope_dim,
            head_dim,
            expert_used_count,
            ssm_conv_kernel,
            ssm_state_size,
            ssm_group_count,
            ssm_time_step_rank,
            ssm_inner_size,
            full_attention_interval,
        } = PropsGGUF::try_from(metadata).or_else(|err| candle_core::bail!("{err}"))?;
        // llama.cpp appends the nextn (MTP) blocks after the trunk and counts them in block_count
        let nextn = nextn_predict_layers(ct.get_metadata(), &arch);
        let block_count = block_count - nextn;

        let gdn_cfg = GdnCfg {
            hidden_size: embedding_length,
            rms_norm_eps: rms_norm_eps as f64,
            conv_kernel: ssm_conv_kernel,
            head_k_dim: ssm_state_size,
            head_v_dim: ssm_inner_size / ssm_time_step_rank,
            num_k_heads: ssm_group_count,
            num_v_heads: ssm_time_step_rank,
            quant: None,
        };
        let perm = if arch == ARCH_NEXT {
            (0..gdn_cfg.num_v_heads).collect()
        } else {
            v_head_perm(gdn_cfg.num_k_heads, gdn_cfg.num_v_heads)
        };
        let gdn_dtype = match (device, dtype) {
            (Device::Cuda(_) | Device::Metal(_), DType::F32) => DType::BF16,
            (_, d) => d,
        };

        let is_attn: Vec<bool> = match ct
            .get_metadata()
            .get(&format!("{arch}.attention.recurrent_layers"))
        {
            Some(v) => {
                use crate::utils::gguf_metadata::TryValueInto;
                let recr: Vec<bool> = v.clone().try_value_into()?;
                (0..block_count)
                    .map(|i| !recr.get(i).copied().unwrap_or(false))
                    .collect()
            }
            None => (0..block_count)
                .map(|i| (i + 1) % full_attention_interval == 0)
                .collect(),
        };

        let tok_embeddings = QEmbedding::new(ct.tensor("token_embd.weight", &Device::Cpu)?)?;
        let norm = QRmsNorm::new(ct.tensor("output_norm.weight", device)?, rms_norm_eps)?;
        let output = if ct.has_tensor("output.weight") {
            ct.tensor("output.weight", device)?
        } else {
            ct.tensor("token_embd.weight", device)?
        };

        let mut ropes = HashMap::new();
        for layer_idx in (0..block_count).filter(|&i| is_attn[i]) {
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            if mistralrs_quant::TieredExperts::enabled() && !device.is_cuda() {
                candle_core::bail!(
                    "titan tiered experts: layer {layer_idx} was mapped to {device:?}; tiering needs every layer on the GPU. \
                     Lower TITAN_TIERED_GPU_FRACTION (or use `auto`) instead of letting the device map spill layers."
                );
            }
            if let std::collections::hash_map::Entry::Vacant(e) = ropes.entry(device.location()) {
                e.insert(Arc::new(RotaryEmbedding::new_partial(
                    rope_freq_base,
                    rope_dim,
                    max_seq_len,
                    device,
                    true,
                    DType::F32,
                )?));
            }
        }

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in NiceProgressBar::<_, 'b'>(
            0..block_count,
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let prefix = format!("blk.{layer_idx}");
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            if mistralrs_quant::TieredExperts::enabled() && !device.is_cuda() {
                candle_core::bail!(
                    "titan tiered experts: layer {layer_idx} was mapped to {device:?}; tiering needs every layer on the GPU. \
                     Lower TITAN_TIERED_GPU_FRACTION (or use `auto`) instead of letting the device map spill layers."
                );
            }

            let mixer = if is_attn[layer_idx] {
                let rotary = ropes
                    .get(&device.location())
                    .expect("No RoPE for device location!")
                    .clone();
                let paged_attn = match &attention_mechanism {
                    AttentionImplementation::Eager => None,
                    AttentionImplementation::PagedAttention => {
                        Some(PagedAttention::new(head_dim, device, None)?)
                    }
                };
                Mixer::Attention(load_full_attention(
                    &mut ct,
                    &prefix,
                    device,
                    rotary,
                    paged_attn,
                    (head_count, head_count_kv, head_dim),
                    rms_norm_eps,
                    dtype,
                )?)
            } else {
                Mixer::Linear(load_gdn(
                    &mut ct, &prefix, &gdn_cfg, &perm, device, gdn_dtype,
                )?)
            };

            let ffn = if ct.has_tensor(&format!("{prefix}.ffn_gate_exps.weight")) {
                let Some(top_k) = expert_used_count else {
                    candle_core::bail!("{arch}: {prefix} has experts but no expert_used_count");
                };
                Ffn::Moe(MoeBlock {
                    router: QMatMul::from_qtensor(
                        ct.tensor(&format!("{prefix}.ffn_gate_inp.weight"), device)?,
                    )?,
                    experts: load_experts(&mut ct, &prefix, device)?,
                    shared: load_shared_expert(&mut ct, &prefix, device)?,
                    top_k,
                })
            } else {
                Ffn::Dense(DenseFfn {
                    gate: gguf_linear(ct.tensor(&format!("{prefix}.ffn_gate.weight"), device)?)?,
                    up: gguf_linear(ct.tensor(&format!("{prefix}.ffn_up.weight"), device)?)?,
                    down: gguf_linear(ct.tensor(&format!("{prefix}.ffn_down.weight"), device)?)?,
                })
            };

            layers.push(DecoderLayer {
                mixer,
                attn_norm: QRmsNorm::new(
                    ct.tensor(&format!("{prefix}.attn_norm.weight"), device)?,
                    rms_norm_eps,
                )?,
                post_attention_norm: QRmsNorm::new(
                    ct.tensor(&format!("{prefix}.post_attention_norm.weight"), device)?,
                    rms_norm_eps,
                )?,
                ffn,
            });
        }

        let mtp = if nextn > 0 && mtp_draft_len() > 0 {
            if nextn > 1 {
                tracing::warn!("{arch}: {nextn} nextn blocks, using the first only");
            }
            let rotary = ropes
                .entry(device.location())
                .or_insert_with(|| {
                    Arc::new(
                        RotaryEmbedding::new_partial(
                            rope_freq_base,
                            rope_dim,
                            max_seq_len,
                            device,
                            true,
                            DType::F32,
                        )
                        .expect("MTP RoPE"),
                    )
                })
                .clone();
            let Some(top_k) = expert_used_count else {
                candle_core::bail!("{arch}: MTP needs an MoE (qwen35moe / qwen3next) arch");
            };
            #[cfg(feature = "cuda")]
            if let Device::Cuda(dev) = device {
                mistralrs_quant::mmvq_rows::warm_up(dev)?;
            }
            Some(load_mtp_block(
                &mut ct,
                &format!("blk.{block_count}"),
                device,
                rotary,
                (head_count, head_count_kv, head_dim),
                rms_norm_eps,
                top_k,
                dtype,
            )?)
        } else {
            if nextn > 0 {
                tracing::info!("{arch}: {nextn} nextn (MTP) block(s) present, not loaded (set TITAN_MTP=<draft tokens> to use)");
            }
            None
        };

        let num_attn = is_attn.iter().filter(|a| **a).count();
        tracing::info!(
            "{arch}: {num_attn} full-attention layers, {} gated-deltanet layers, GDN dtype {gdn_dtype:?}",
            block_count - num_attn
        );

        let hybrid_cache = HybridCache::new(
            HybridCacheConfig {
                layer_types: is_attn
                    .iter()
                    .map(|&a| {
                        if a {
                            HybridLayerType::Attention
                        } else {
                            HybridLayerType::Recurrent
                        }
                    })
                    .collect(),
                max_seq_len,
                recurrent: RecurrentLayerConfig {
                    conv_dim: gdn_cfg.linear_conv_dim(),
                    conv_width: gdn_cfg.conv_kernel,
                    // key-major [heads, k, v] state as the titan GDN kernels (84b53bf / oxide) keep
                    // it; v0.9.4's `Gdn` spec would pick the value-major layout on CUDA
                    state: RecurrentStateSpec::Opaque {
                        dims: vec![gdn_cfg.num_v_heads, gdn_cfg.head_k_dim, gdn_cfg.head_v_dim],
                    },
                    recurrent_dtype: Some(DType::F32),
                },
            },
            gdn_dtype,
            // every layer's caches on the main device, as before (tiering keeps all layers on it)
            &vec![device.clone(); is_attn.len()],
        )?;

        Ok(Self {
            tok_embeddings,
            layers,
            norm,
            output: gguf_linear(output)?,
            device: device.clone(),
            cache: EitherCache::Hybrid(Arc::new(Mutex::new(hybrid_cache))),
            max_seq_len,
            mapper: Some(mapper),
            dtype,
            gdn_dtype,
            mtp,
            spec: Arc::new(Mutex::new(SpecState {
                last: None,
                pending: None,
                mtp_kv: None,
                mtp_len: 0,
                mtp_valid: false,
                verify: None,
                stats: MtpStats::default(),
            })),
            prefix_points: Mutex::new(HashMap::new()),
            #[cfg(feature = "cuda")]
            graphs: Default::default(),
        })
    }
}

impl ModelWeights {
    pub fn forward(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        let rows = input_ids.dim(1)?;
        crate::titan_faults::forward_start(rows);
        let t = prof_wall_enter(rows);
        let r = self.forward_inner(input_ids, ctx);
        prof_wall_exit(rows, t);
        r
    }

    fn forward_inner(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        if input_ids.dim(1)? > 1
            && matches!(ctx.recurrent_batch_kind(), Some(RecurrentBatchKind::Decode))
        {
            // a decode step carrying staged speculative tokens
            let mut t = std::time::Instant::now();
            #[cfg(feature = "cuda")]
            let r = if super::titan_graph::enabled()
                && input_ids.dim(0)? == 1
                && !ctx.is_paged()
                && !mtp_verify_rowwise()
                && self.device.is_cuda()
                && self.graph_model_ok()
            {
                self.forward_verify_graphed(input_ids, ctx)
            } else {
                self.forward_verify(input_ids, ctx)
            };
            #[cfg(not(feature = "cuda"))]
            let r = self.forward_verify(input_ids, ctx);
            prof(VERIFY_NAMES[input_ids.dim(1)?.min(8)], &self.device, &mut t);
            return r;
        }
        if let Some(bounds) = self.prefill_chunks(input_ids, ctx)? {
            return self.forward_chunked(input_ids, ctx, &bounds);
        }
        // a prompt pass continuing a resumed prefix (prefix cache) starts from its conv inputs
        let prefill = matches!(ctx.recurrent_batch_kind(), Some(RecurrentBatchKind::Prefill));
        let base = ctx.seqlen_offsets().first().copied().unwrap_or(0);
        let slot = if prefill { prefix_slot(ctx) } else { None };
        if base == 0 {
            if let Some(slot) = slot {
                self.prefix_points.lock().expect("prefix points poisoned").remove(&slot);
            }
        }
        let mut t_fwd = std::time::Instant::now();
        #[cfg(feature = "cuda")]
        let x = if self.graphable_decode(input_ids, ctx)? {
            self.forward_trunk_graphed(input_ids, ctx)?
        } else {
            self.forward_trunk(input_ids, ctx, prefill && base > 0)?
        };
        #[cfg(not(feature = "cuda"))]
        let x = self.forward_trunk(input_ids, ctx, prefill && base > 0)?;
        if let Some(slot) = slot {
            self.save_prefix_point(slot, base + input_ids.dim(1)?, false)?;
        }
        let x = ctx.logits(&x)?;
        let out = self.output.forward(&x.contiguous()?);
        if input_ids.dim(1)? == 1 {
            prof("decode", &self.device, &mut t_fwd);
        }
        out
    }

    /// The `(start, len)` chunks a prompt pass is split into, or `None` to run it whole: only an
    /// unpaged prefill longer than `TITAN_PREFILL_CHUNK` whose requested logits all lie in the last
    /// chunk (not raw logits over the whole prompt).
    fn prefill_chunks(
        &self,
        input_ids: &Tensor,
        ctx: &ModelForwardContext<'_>,
    ) -> Result<Option<Vec<(usize, usize)>>> {
        let c = prefill_chunk_size();
        let l = input_ids.dim(1)?;
        let split: Vec<usize> = if ctx.seqlen_offsets().first().copied().unwrap_or(0) == 0 {
            prefill_split_at().iter().copied().filter(|&s| s < l).collect()
        } else {
            Vec::new()
        };
        if c == 0
            || (l <= c && split.is_empty())
            || ctx.is_paged()
            || crate::using_flash_attn()
            || !matches!(ctx.recurrent_batch_kind(), Some(RecurrentBatchKind::Prefill))
        {
            return Ok(None);
        }
        let base0 = ctx.seqlen_offsets().first().copied().unwrap_or(0);
        let cap = prefill_big_max_ctx();
        let big = if cap > 0 && base0 + l > cap { 0 } else { prefill_big_chunk() };
        let bounds = if split.is_empty() {
            prefill_grid_bounds(l, c, big)
        } else {
            // debug reference for prefix-cache resumes at the split points: every segment runs the
            // chunks a prompt pass resumed at its start runs (whole up to `c` tokens)
            let mut b = Vec::new();
            let mut from = 0;
            for end in split.iter().copied().chain(std::iter::once(l)) {
                let n = end - from;
                let seg = if n > c { prefill_grid_bounds(n, c, big) } else { vec![(0, n)] };
                b.extend(seg.into_iter().map(|(st, k)| (st + from, k)));
                from = end;
            }
            b
        };
        let last = bounds.last().map_or(0, |b| b.0);
        if ctx.context_lens().iter().any(|(start, _)| *start < last) {
            return Ok(None);
        }
        Ok(Some(bounds))
    }

    /// A prompt pass as consecutive chunks, each a prompt pass continuing the caches of the one before:
    /// attention appends to the KV cache (the causal mask and RoPE positions start at the chunk's
    /// offset), the GDN layers start from the recurrent state and the conv inputs the previous chunk
    /// left, and the MTP block catches up chunk by chunk. Only the last chunk's logits are computed.
    fn forward_chunked(
        &self,
        input_ids: &Tensor,
        ctx: &ModelForwardContext<'_>,
        bounds: &[(usize, usize)],
    ) -> Result<Tensor> {
        let b = input_ids.dim(0)?;
        let base = ctx.seqlen_offsets().to_vec();
        let position_ids = ctx.position_ids_vec();
        let (last_start, _) = *bounds.last().expect("at least one chunk");
        let last_lens = ctx
            .context_lens()
            .iter()
            .map(|(start, len)| (start - last_start, *len))
            .collect::<Vec<_>>();
        let recurrent_metadata = ctx.recurrent_metadata().cloned();
        let kind = ctx.recurrent_batch_kind().expect("prefill");
        let base0 = base.first().copied().unwrap_or(0);
        let slot = prefix_slot(ctx);
        if base0 == 0 {
            if let Some(slot) = slot {
                self.prefix_points.lock().expect("prefix points poisoned").remove(&slot);
            }
        }
        // KV room for the whole prompt and the start of its reply, in KV_RESERVE_STEP buckets: the
        // buffers are allocated once, not regrown per chunk, and bucket-sized buffers freed by one
        // sequence are reused whole by the next instead of fragmenting the device pool
        let end = (base.iter().max().copied().unwrap_or(0) + input_ids.dim(1)? + 1)
            .next_multiple_of(KV_RESERVE_STEP)
            .min(self.max_seq_len);
        self.cache.hybrid().reserve_attention(end)?;
        #[cfg(feature = "cuda")]
        mistralrs_quant::TieredExperts::set_prefill_context(end);
        let mut out = None;
        let tf_tokens: Option<Vec<u32>> = if std::env::var_os("TITAN_TF_DUMP").is_some() && b == 1 {
            Some(input_ids.to_dtype(DType::U32)?.flatten_all()?.to_vec1::<u32>()?)
        } else {
            None
        };
        for (i, &(start, len)) in bounds.iter().enumerate() {
            cuda_pool::between_chunks(&self.device, i, start)?;
            if i == 1 {
                // the MTP block's KV cache starts with the first chunk
                if let Some(kv) = self.spec.lock().expect("spec state poisoned").mtp_kv.as_mut() {
                    kv.reserve(end)?;
                }
            }
            let is_last = i + 1 == bounds.len();
            if is_last && i > 0 {
                // the last chunk boundary: the chunk grid a later prompt with this prefix runs too
                if let Some(slot) = slot {
                    self.save_prefix_point(slot, base0 + start, true)?;
                }
            }
            crate::titan_monitor::prefill_progress(base0 + start);
            let ids = input_ids.narrow(1, start, len)?;
            let offsets = base.iter().map(|o| o + start).collect::<Vec<_>>();
            let lens = if is_last { last_lens.clone() } else { vec![(len - 1, 1); b] };
            let mut cctx = ModelForwardContext::new(&offsets, &lens, &position_ids, None, ctx.flash_params())
                .with_recurrent_batch_kind(kind)
                .with_recurrent_metadata(recurrent_metadata.clone());
            let x = self.forward_trunk(&ids, &mut cctx, i > 0 || base0 > 0)?;
            if let Some(toks) = &tf_tokens {
                tf_dump(&self.output, &x, toks, start)?;
            }
            if is_last {
                let x = cctx.logits(&x)?;
                out = Some(self.output.forward(&x.contiguous()?)?);
            }
        }
        if let Some(slot) = slot {
            self.save_prefix_point(slot, base0 + input_ids.dim(1)?, false)?;
        }
        Ok(out.expect("last chunk"))
    }

    /// Embedding, the decoder layers and the final norm over `input_ids` (the MTP block runs on the
    /// result too); `conv_carry` marks a prompt chunk continuing the previous one.
    fn forward_trunk(
        &self,
        input_ids: &Tensor,
        ctx: &mut ModelForwardContext<'_>,
        conv_carry: bool,
    ) -> Result<Tensor> {
        let mut x = self
            .tok_embeddings
            .forward(input_ids, &self.device)?
            .to_dtype(DType::F32)?;

        let mut hybrid_cache = self.cache.hybrid();
        let recurrent_metadata = ctx.recurrent_metadata().cloned().ok_or_else(|| {
            candle_core::Error::msg(
                "qwen35: hybrid recurrent metadata is required for the linear-attention layers",
            )
        })?;

        let mask = if ctx.is_paged() {
            let cache = ForwardMaskCache::Paged(ctx.seqlen_offsets());
            CausalMasker.make_causal_mask(
                input_ids,
                &cache,
                self.dtype,
                &CausalMaskConfig::default(),
            )?
        } else {
            CausalMasker.make_causal_mask(
                input_ids,
                &*hybrid_cache as &dyn PastKvLenCache,
                self.dtype,
                &CausalMaskConfig::default(),
            )?
        };
        let mask = if ctx.is_first_prompt_chunk() {
            mask
        } else {
            AttentionMask::None
        };
        let mask = match &self.mapper {
            Some(mapper) => DeviceMappedMask::new(mask, &**mapper)?,
            None => DeviceMappedMask::from_single(mask),
        };

        let look = lookahead_depth();
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(mapper) = &self.mapper {
                x = mapper.map(x, i)?;
            }
            let mut t = std::time::Instant::now();
            let residual = &x;
            let h = layer.attn_norm.forward(&x)?;
            let h = match (&layer.mixer, hybrid_cache.get_mut(i)) {
                (Mixer::Attention(attn), Some(HybridLayerCache::Attention(kv_cache))) => {
                    attn.forward(&h, &mask.get(h.device()), kv_cache, ctx, i)?
                }
                (Mixer::Linear(gdn), Some(HybridLayerCache::Recurrent(pool))) => {
                    let indices = recurrent_metadata.state_indices();
                    let mut gdn_cache = GdnLayerCache {
                        conv_state: pool.gather_conv_state(indices)?,
                        recurrent_state: pool.gather_recurrent_state(indices)?,
                    };
                    let out = gdn.forward_with_conv_carry(
                        &h.to_dtype(self.gdn_dtype)?,
                        &mut gdn_cache,
                        recurrent_metadata.batch_kind(),
                        conv_carry,
                    )?;
                    pool.scatter_conv_state_with_host_indices(
                        indices,
                        recurrent_metadata.state_indices_host(),
                        &gdn_cache.conv_state,
                    )?;
                    pool.scatter_recurrent_state_with_host_indices(
                        indices,
                        recurrent_metadata.state_indices_host(),
                        &gdn_cache.recurrent_state,
                    )?;
                    out.to_dtype(DType::F32)?
                }
                _ => candle_core::bail!("qwen35: layer {i} cache kind does not match its mixer"),
            };
            if h.dim(1)? == 1 {
                prof(if matches!(layer.mixer, Mixer::Attention(_)) { "d.attn" } else { "d.gdn" }, &self.device, &mut t);
            }
            let x_attn = (h + residual)?;
            if x_attn.dim(1)? > 1 {
                // mixer (attention / GDN) output, before the FFN: record 1000 + layer
                layer_dump_bin(1000 + i, ctx.seqlen_offsets().first().copied().unwrap_or(0), &x_attn)?;
            }
            let h = layer.post_attention_norm.forward(&x_attn)?;
            let decode_row = x_attn.dim(1)? == 1;
            let h = match &layer.ffn {
                Ffn::Moe(moe) if look > 0 && decode_row => {
                    let preds = self.lookahead_preds(i, &x_attn, look)?;
                    prof("d.look", &self.device, &mut t);
                    moe.forward_lookahead(&h, &preds)?
                }
                _ => layer.ffn.forward(&h)?,
            };
            x = (h + x_attn)?;
            if x.dim(1)? == 1 {
                prof("d.moe", &self.device, &mut t);
            }
            if x.dim(1)? > 1 {
                layer_dump(i, &x)?;
                layer_dump_bin(i, ctx.seqlen_offsets().first().copied().unwrap_or(0), &x)?;
            }
        }

        drop(hybrid_cache);
        if mtp_prof_on() && self.mtp.is_none() && x.dim(1)? == 1 {
            static STEPS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            if STEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 256 == 255 {
                tracing::info!("titan prof: {}", prof_report());
            }
        }
        let x = x.to_device(&self.device)?;
        let x = self.norm.forward(&x)?;
        if self.mtp.is_some() {
            self.mtp_after_trunk(input_ids, &x, ctx)?;
        }
        Ok(x)
    }

    /// P1: the routing of layers `i + 1 ..= i + depth` predicted from layer `i`'s residual (after its
    /// mixer), each through that layer's own norm and router.
    fn lookahead_preds(&self, i: usize, x_attn: &Tensor, depth: usize) -> Result<Vec<(&Experts, usize, Tensor)>> {
        let mut out = Vec::new();
        for d in 1..=depth {
            let Some(next) = self.layers.get(i + d) else {
                break;
            };
            let Ffn::Moe(moe) = &next.ffn else {
                continue;
            };
            let h = next.post_attention_norm.forward(x_attn)?;
            let hidden = h.dim(D::Minus1)?;
            let flat = h.reshape(((), hidden))?.to_dtype(DType::F32)?;
            out.push((&moe.experts, d, moe.route(&flat)?.indices));
        }
        Ok(out)
    }

    /// Target forward of a decode step plus its staged drafts, `(1, n)` tokens at positions
    /// `start..start + n`. Every row reproduces the plain decode step bit for bit: each op runs on one
    /// row with the decode kernels (so the mmvq/cuBLAS/GDN kernel choice is the one-token choice),
    /// except the routed experts, which are row-independent and run batched. Leaves every row appended
    /// to the caches and records the GDN states after each row so that `SpecRollback` can keep any
    /// accepted prefix.
    fn forward_verify(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        let (b, n) = input_ids.dims2()?;
        if b != 1 {
            candle_core::bail!("qwen35: speculative verification runs one sequence per step, got {b}");
        }
        let start = ctx
            .seqlen_offsets()
            .first()
            .copied()
            .ok_or_else(|| candle_core::Error::msg("qwen35 verify: missing seqlen offset"))?;
        let recurrent_metadata = ctx.recurrent_metadata().cloned().ok_or_else(|| {
            candle_core::Error::msg(
                "qwen35: hybrid recurrent metadata is required for the linear-attention layers",
            )
        })?;
        let slot = recurrent_metadata
            .state_indices_host()
            .and_then(|h| h.first().copied())
            .ok_or_else(|| candle_core::Error::msg("qwen35 verify: missing recurrent slot"))?
            as usize;
        let flash_params = ctx.flash_params();
        let positions = (0..n)
            .map(|r| crate::pipeline::text_positions_tensor(&[start + r], 1, &self.device))
            .collect::<Result<Vec<_>>>()?;

        let mut x = self
            .tok_embeddings
            .forward(input_ids, &self.device)?
            .to_dtype(DType::F32)?;
        let mut hybrid_cache = self.cache.hybrid();
        let mut gdn_record = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(mapper) = &self.mapper {
                x = mapper.map(x, i)?;
            }
            if !mtp_verify_rowwise() {
                let mut t = std::time::Instant::now();
                let h = layer.attn_norm.forward(&x)?;
                let h = match (&layer.mixer, hybrid_cache.get_mut(i)) {
                    (Mixer::Attention(attn), Some(HybridLayerCache::Attention(kv_cache))) => {
                        attn.forward_decode_rows(&h, kv_cache, start, Some(flash_params))?
                    }
                    (Mixer::Linear(gdn), Some(HybridLayerCache::Recurrent(pool))) => {
                        let indices = recurrent_metadata.state_indices();
                        let mut gdn_cache = GdnLayerCache {
                            conv_state: pool.gather_conv_state(indices)?,
                            recurrent_state: pool.gather_recurrent_state(indices)?,
                        };
                        let mut snaps = Vec::with_capacity(n - 1);
                        let out = gdn.forward_decode_rows(
                            &h.to_dtype(self.gdn_dtype)?,
                            &mut gdn_cache,
                            &|w, x| lin_rows(w, x),
                            &mut |r, c| {
                                if r + 1 < n {
                                    // the recurrent state is updated in place by the next row: copy it
                                    snaps.push((c.conv_state.copy()?, c.recurrent_state.copy()?));
                                }
                                Ok(())
                            },
                        )?;
                        pool.scatter_conv_state_with_host_indices(
                            indices,
                            recurrent_metadata.state_indices_host(),
                            &gdn_cache.conv_state,
                        )?;
                        pool.scatter_recurrent_state_with_host_indices(
                            indices,
                            recurrent_metadata.state_indices_host(),
                            &gdn_cache.recurrent_state,
                        )?;
                        gdn_record.push((i, snaps));
                        out.to_dtype(DType::F32)?
                    }
                    _ => candle_core::bail!("qwen35: layer {i} cache kind does not match its mixer"),
                };
                prof(
                    if matches!(layer.mixer, Mixer::Attention(_)) { "v.attn" } else { "v.gdn" },
                    &self.device,
                    &mut t,
                );
                let x_attn = (h + &x)?;
                let h = layer.post_attention_norm.forward(&x_attn)?;
                let h = layer.ffn.forward_rows(&h)?;
                x = (h + x_attn)?;
                prof("v.moe", &self.device, &mut t);
                continue;
            }
            let mut x_attn_rows = Vec::with_capacity(n);
            match (&layer.mixer, hybrid_cache.get_mut(i)) {
                (Mixer::Attention(attn), Some(HybridLayerCache::Attention(kv_cache))) => {
                    for (r, pos) in positions.iter().enumerate() {
                        let residual = x.narrow(1, r, 1)?;
                        let h = layer.attn_norm.forward(&residual)?;
                        let pos = pos.to_device(h.device())?;
                        let h = attn.forward_eager(
                            &h,
                            &AttentionMask::None,
                            kv_cache,
                            &pos,
                            Some(flash_params),
                        )?;
                        x_attn_rows.push((h + &residual)?);
                    }
                }
                (Mixer::Linear(gdn), Some(HybridLayerCache::Recurrent(pool))) => {
                    let indices = recurrent_metadata.state_indices();
                    let mut gdn_cache = GdnLayerCache {
                        conv_state: pool.gather_conv_state(indices)?,
                        recurrent_state: pool.gather_recurrent_state(indices)?,
                    };
                    let mut snaps = Vec::with_capacity(n - 1);
                    for r in 0..n {
                        let residual = x.narrow(1, r, 1)?;
                        let h = layer.attn_norm.forward(&residual)?;
                        let out = gdn.forward(
                            &h.to_dtype(self.gdn_dtype)?,
                            &mut gdn_cache,
                            RecurrentBatchKind::Decode,
                        )?;
                        if r + 1 < n {
                            // the recurrent state is updated in place by the next row: copy it
                            snaps.push((
                                gdn_cache.conv_state.copy()?,
                                gdn_cache.recurrent_state.copy()?,
                            ));
                        }
                        x_attn_rows.push((out.to_dtype(DType::F32)? + &residual)?);
                    }
                    pool.scatter_conv_state_with_host_indices(
                        indices,
                        recurrent_metadata.state_indices_host(),
                        &gdn_cache.conv_state,
                    )?;
                    pool.scatter_recurrent_state_with_host_indices(
                        indices,
                        recurrent_metadata.state_indices_host(),
                        &gdn_cache.recurrent_state,
                    )?;
                    gdn_record.push((i, snaps));
                }
                _ => candle_core::bail!("qwen35: layer {i} cache kind does not match its mixer"),
            }
            let h = x_attn_rows
                .iter()
                .map(|row| layer.post_attention_norm.forward(row))
                .collect::<Result<Vec<_>>>()?;
            let h = layer.ffn.forward_rows(&Tensor::cat(&h, 1)?)?;
            x = (h + Tensor::cat(&x_attn_rows, 1)?)?;
        }
        drop(hybrid_cache);

        let x = x.to_device(&self.device)?;
        let (normed, logits) = if mtp_verify_rowwise() {
            let mut normed = Vec::with_capacity(n);
            let mut logits = Vec::with_capacity(n);
            for r in 0..n {
                let row = self.norm.forward(&x.narrow(1, r, 1)?)?;
                logits.push(self.output.forward(&row.contiguous()?)?);
                normed.push(row);
            }
            (Tensor::cat(&normed, 1)?, Tensor::cat(&logits, 1)?)
        } else {
            let normed = self.norm.forward(&x)?;
            let logits = lin_rows(&self.output, &normed.contiguous()?)?;
            (normed, logits)
        };
        {
            let mut st = self.spec.lock().expect("spec state poisoned");
            st.last = Some((start, normed.squeeze(0)?));
            st.verify = Some(VerifyRecord {
                start,
                rows: n,
                slot,
                gdn: gdn_record,
            });
        }
        Ok(logits)
    }

    /// After a trunk forward: keep the final-normed hidden rows for the MTP head and, for a prompt
    /// chunk, run the MTP block over it so its KV cache covers every prompt position (llama.cpp's
    /// draft-mtp `process()` hook: MTP position p pairs token x_p with trunk hidden h_{p-1}).
    fn mtp_after_trunk(
        &self,
        input_ids: &Tensor,
        normed: &Tensor,
        ctx: &ModelForwardContext<'_>,
    ) -> Result<()> {
        let Some(mtp) = &self.mtp else {
            return Ok(());
        };
        let mut st = self.spec.lock().expect("spec state poisoned");
        let (b, n) = input_ids.dims2()?;
        let start = ctx.seqlen_offsets().first().copied().unwrap_or(0);
        if b != 1 {
            st.last = None;
            st.mtp_valid = false;
            return Ok(());
        }
        let hidden = normed.squeeze(0)?.to_dtype(DType::F32)?;
        st.last = Some((start, hidden.clone()));
        if matches!(ctx.recurrent_batch_kind(), Some(RecurrentBatchKind::Decode)) {
            return Ok(()); // the draft step catches the MTP cache up with the accepted rows
        }
        if start == 0 {
            if st.stats.steps > 0 || mtp_prof_wall() || mtp_prof_on() {
                log_mtp_stats(&st.stats);
            }
            st.mtp_kv = Some(KvCache::new_normal(
                2,
                self.max_seq_len,
                crate::kv_cache::NormalCache::CACHE_GROW_SIZE,
            ));
            st.mtp_len = 0;
            st.pending = None;
            st.mtp_valid = true;
        }
        let h_prev0 = match &st.pending {
            _ if start == 0 => Tensor::zeros((1, hidden.dim(1)?), DType::F32, hidden.device())?,
            Some((pos, h)) if *pos + 1 == start => h.clone(),
            _ => {
                st.mtp_valid = false;
                return Ok(());
            }
        };
        if !st.mtp_valid || st.mtp_len != start {
            st.mtp_valid = false;
            return Ok(());
        }
        let h_prev = if n > 1 {
            Tensor::cat(&[h_prev0, hidden.narrow(0, 0, n - 1)?], 0)?
        } else {
            h_prev0
        };
        let mut kv = st.mtp_kv.take().expect("mtp kv");
        let res = self.mtp_forward(mtp, &mut kv, input_ids, &h_prev, start, false);
        st.mtp_kv = Some(kv);
        res?;
        st.mtp_len = start + n;
        st.pending = Some((start + n - 1, hidden.narrow(0, n - 1, 1)?));
        Ok(())
    }

    /// One MTP block pass over `tokens` `(1, m)` at positions `start..start + m`, `h_prev` `(m, hidden)`
    /// the trunk (or previous MTP step) hidden paired with each token. Returns the last row's draft
    /// logits (when asked) and its head-normed hidden `(1, hidden)`.
    fn mtp_forward(
        &self,
        mtp: &MtpBlock,
        kv: &mut KvCache,
        tokens: &Tensor,
        h_prev: &Tensor,
        start: usize,
        want_logits: bool,
    ) -> Result<(Option<Tensor>, Tensor)> {
        let m = tokens.dim(1)?;
        let e = self
            .tok_embeddings
            .forward(tokens, &self.device)?
            .to_dtype(DType::F32)?;
        let e = mtp.enorm.forward(&e)?;
        let h = mtp.hnorm.forward(&h_prev.unsqueeze(0)?.to_dtype(DType::F32)?)?;
        let x = mtp.eh_proj.forward(&Tensor::cat(&[e, h], D::Minus1)?.contiguous()?)?;

        let past = [kv.current_seq_len()];
        let past: &[usize] = &past;
        // a prompt chunk that flash-prefill takes needs no mask tensor: at 2048 rows and 37k keys the
        // (rows, keys) mask would hold 150 MiB through the MTP layer; `attend` checks the placeholder is unused
        let mask = if m > MTP_CATCHUP_MAX_ROWS && crate::attention::flash_prefill_wanted(m, past[0] + m) {
            AttentionMask::Custom(Tensor::zeros((1, 1), self.dtype, &self.device)?)
        } else {
            CausalMasker.make_causal_mask(
                tokens,
                &past as &dyn PastKvLenCache,
                self.dtype,
                &CausalMaskConfig::default(),
            )?
        };
        let positions = crate::pipeline::text_positions_tensor(&[start], m, &self.device)?;
        let h = mtp.attn_norm.forward(&x)?;
        // catch-up over accepted tokens (2..=MTP_CATCHUP_MAX_ROWS rows, not a prompt chunk) at or above the
        // flash-decode threshold: the rows path instead of the masked eager attention; a draft only steers acceptance
        let h = if (2..=MTP_CATCHUP_MAX_ROWS).contains(&m)
            && crate::attention::flash_decode_min_kv().is_some_and(|t| past[0] + m >= t)
        {
            mtp.attn.forward_decode_rows(&h, kv, start, None)?
        } else {
            mtp.attn.forward_eager(&h, &mask, kv, &positions, None)?
        };
        let x_attn = (h + x)?;
        let h = mtp.post_attention_norm.forward(&x_attn)?;
        #[cfg(feature = "cuda")]
        let grouped = match &mtp.ffn {
            Ffn::Moe(moe) if m >= MTP_GROUPED_MIN_ROWS => {
                let piece = prefill_moe_piece();
                if piece > 0 && m > piece {
                    // a big prompt chunk: pieces keep the (tokens, k, hidden) temporaries at the piece size
                    let mut parts = Vec::with_capacity(m.div_ceil(piece));
                    let mut r0 = 0;
                    while r0 < m {
                        let n = piece.min(m - r0);
                        match moe.forward_grouped_stock(&h.narrow(1, r0, n)?)? {
                            Some(p) => parts.push(p),
                            None => break,
                        }
                        r0 += n;
                    }
                    if r0 == m {
                        Some(Tensor::cat(&parts, 1)?)
                    } else {
                        None
                    }
                } else {
                    moe.forward_grouped_stock(&h)?
                }
            }
            _ => None,
        };
        #[cfg(not(feature = "cuda"))]
        let grouped = None;
        let h = match grouped {
            Some(h) => h,
            None => mtp.ffn.forward(&h)?,
        };
        let x = (h + x_attn)?;
        let last = mtp.head_norm.forward(&x.narrow(1, m - 1, 1)?)?;
        let logits = if want_logits {
            Some(self.output.forward(&last.contiguous()?)?.flatten_all()?)
        } else {
            None
        };
        Ok((logits, last.squeeze(0)?))
    }

    /// Whether this forward runs as piecewise CUDA graphs: `TITAN_CUDA_GRAPHS=1`, one decode row of one
    /// sequence, unpaged attention, one device, MoE in every layer, no router lookahead.
    #[cfg(feature = "cuda")]
    fn graphable_decode(&self, input_ids: &Tensor, ctx: &ModelForwardContext<'_>) -> Result<bool> {
        if !super::titan_graph::enabled() || input_ids.dims2()? != (1, 1) || !self.device.is_cuda() {
            return Ok(false);
        }
        if !matches!(ctx.recurrent_batch_kind(), Some(RecurrentBatchKind::Decode)) || ctx.is_paged() {
            return Ok(false);
        }
        Ok(self.graph_model_ok())
    }

    /// The step-invariant part of `graphable_decode` (checked once).
    #[cfg(feature = "cuda")]
    fn graph_model_ok(&self) -> bool {
        static STATIC_OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *STATIC_OK.get_or_init(|| {
            let one_device = self.mapper.as_ref().is_none_or(|m| m.get_unique_devices().len() == 1);
            let layers_ok = self.layers.iter().all(|l| {
                matches!(l.ffn, Ffn::Moe(_))
                    && match &l.mixer {
                        Mixer::Attention(a) => a.paged_attn.is_none(),
                        Mixer::Linear(_) => true,
                    }
            });
            let ok = one_device && layers_ok && lookahead_depth() == 0;
            tracing::info!("titan graphs: piecewise CUDA graphs for batch-1 decode {}", if ok { "on" } else { "off (unsupported model setup)" });
            ok
        })
    }

    /// The previous layer's MoE output added to its residual (`MoeBlock::forward_lookahead` after the
    /// routed experts, then the trunk's residual add): `carry` = `[x_attn, flat, routed, weights]`, or
    /// `[x]` before the first layer.
    #[cfg(feature = "cuda")]
    fn graph_layer_input(&self, i: usize, carry: &[Tensor]) -> Result<Tensor> {
        if i == 0 {
            return Ok(carry[0].clone());
        }
        let Ffn::Moe(moe) = &self.layers[i - 1].ffn else {
            candle_core::bail!("qwen35 graphs: layer {} is not MoE", i - 1);
        };
        let (x_attn, flat, routed, weights) = (&carry[0], &carry[1], &carry[2], &carry[3]);
        let (b, s, hidden) = x_attn.dims3()?;
        let h = moe
            .combine(flat, routed, weights)?
            .reshape((b, s, hidden))?
            .to_dtype(x_attn.dtype())?;
        h + x_attn
    }

    /// After a mixer: the residual add, the FFN norm and the router, `[x_attn, flat, ids, weights]`.
    #[cfg(feature = "cuda")]
    fn graph_route(&self, i: usize, mixer_out: Tensor, residual: &Tensor) -> Result<Vec<Tensor>> {
        let layer = &self.layers[i];
        let Ffn::Moe(moe) = &layer.ffn else {
            candle_core::bail!("qwen35 graphs: layer {i} is not MoE");
        };
        let x_attn = (mixer_out + residual)?;
        let h = layer.post_attention_norm.forward(&x_attn)?;
        let hidden = h.dim(D::Minus1)?;
        let flat = h.reshape(((), hidden))?.to_dtype(DType::F32)?;
        let topk = moe.route(&flat)?;
        Ok(vec![x_attn, flat, topk.indices, topk.values])
    }

    /// `forward_trunk` of one decode row as piecewise CUDA graphs (`titan_graph`): per layer, one
    /// graph from the previous layer's combine through this layer's router top-k (full-attention
    /// layers: one up to q/k/v, the KV append and attention eager, one after), the routed experts
    /// eager in between (their routing-id sync and CPU misses cannot be captured), and a last graph for
    /// the final combine and norm. The same ops as `forward_trunk`, so the output is bit-identical.
    #[cfg(feature = "cuda")]
    fn forward_trunk_graphed(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        use super::titan_graph::{dptr, SegKey};
        let g = &self.graphs;
        let x = self
            .tok_embeddings
            .forward(input_ids, &self.device)?
            .to_dtype(DType::F32)?;
        let mut hybrid_cache = self.cache.hybrid();
        let recurrent_metadata = ctx.recurrent_metadata().cloned().ok_or_else(|| {
            candle_core::Error::msg("qwen35: hybrid recurrent metadata is required for the linear-attention layers")
        })?;
        let mask = CausalMasker.make_causal_mask(
            input_ids,
            &*hybrid_cache as &dyn PastKvLenCache,
            self.dtype,
            &CausalMaskConfig::default(),
        )?;
        let mask = if ctx.is_first_prompt_chunk() { mask } else { AttentionMask::None };
        let masked = !matches!(mask, AttentionMask::None);
        let positions = ctx
            .text_positions(&self.device, 1)?
            .ok_or_else(|| candle_core::Error::msg("missing RoPE positions"))?
            .clone();
        let flash_params = ctx.flash_params();
        let indices = recurrent_metadata.state_indices().clone();
        let slot = recurrent_metadata
            .state_indices_host()
            .and_then(|h| h.first().copied())
            .ok_or_else(|| candle_core::Error::msg("qwen35 graphs: missing recurrent slot"))? as u64;

        let mut carry = vec![x];
        for (i, layer) in self.layers.iter().enumerate() {
            let Ffn::Moe(moe) = &layer.ffn else {
                candle_core::bail!("qwen35 graphs: layer {i} is not MoE");
            };
            let routed_in = match (&layer.mixer, hybrid_cache.get_mut(i)) {
                (Mixer::Linear(gdn), Some(HybridLayerCache::Recurrent(pool))) => {
                    let key = SegKey {
                        tag: "gdn",
                        layer: i,
                        extra: vec![slot, dptr(&pool.conv_state)?, dptr(&pool.recurrent_state)?],
                    };
                    let mut inputs: Vec<&Tensor> = carry.iter().collect();
                    inputs.push(&indices);
                    let batch_kind = recurrent_metadata.batch_kind();
                    let host = recurrent_metadata.state_indices_host();
                    g.run(key, &inputs, |ins| {
                        let (carry_in, idx) = ins.split_at(ins.len() - 1);
                        let x = self.graph_layer_input(i, carry_in)?;
                        let h = layer.attn_norm.forward(&x)?;
                        let mut gdn_cache = GdnLayerCache {
                            conv_state: pool.gather_conv_state(&idx[0])?,
                            recurrent_state: pool.gather_recurrent_state(&idx[0])?,
                        };
                        let out = gdn.forward_with_conv_carry(&h.to_dtype(self.gdn_dtype)?, &mut gdn_cache, batch_kind, false)?;
                        pool.scatter_conv_state_with_host_indices(&idx[0], host, &gdn_cache.conv_state)?;
                        pool.scatter_recurrent_state_with_host_indices(&idx[0], host, &gdn_cache.recurrent_state)?;
                        self.graph_route(i, out.to_dtype(DType::F32)?, &x)
                    })?
                }
                (Mixer::Attention(attn), Some(HybridLayerCache::Attention(kv_cache))) => {
                    let mut inputs: Vec<&Tensor> = carry.iter().collect();
                    inputs.push(&positions);
                    let pre = g.run(SegKey { tag: "attn.pre", layer: i, extra: vec![] }, &inputs, |ins| {
                        let (carry_in, pos) = ins.split_at(ins.len() - 1);
                        let x = self.graph_layer_input(i, carry_in)?;
                        let h = layer.attn_norm.forward(&x)?;
                        let (q, k, v, gate) = attn.decode_qkv(&h, &pos[0])?;
                        Ok(vec![x, q, k, v, gate])
                    })?;
                    let (x, q, k, v, gate) = (&pre[0], &pre[1], &pre[2], &pre[3], &pre[4]);
                    let (k, v) = kv_cache.append(k, v)?;
                    let y = Sdpa.run_attention(q, &k, &v, &mask, Some(flash_params), &attn.sdpa_params)?;
                    let y = y.contiguous()?;
                    g.run(
                        SegKey { tag: "attn.post", layer: i, extra: vec![masked as u64] },
                        &[&y, gate, x],
                        |ins| {
                            let out = attn.decode_out(&ins[0], &ins[1], masked, ins[2].dtype())?;
                            self.graph_route(i, out, &ins[2])
                        },
                    )?
                }
                _ => candle_core::bail!("qwen35: layer {i} cache kind does not match its mixer"),
            };
            let (x_attn, flat, ids, weights) = (&routed_in[0], &routed_in[1], &routed_in[2], &routed_in[3]);
            let hidden = flat.dim(D::Minus1)?;
            let num_tokens = flat.dim(0)?;
            let routed = moe.experts.forward_lookahead(&flat.reshape((num_tokens, 1, hidden))?, ids, &[])?;
            carry = vec![x_attn.clone(), flat.clone(), routed, weights.clone()];
        }
        drop(hybrid_cache);

        let n = self.layers.len();
        let inputs: Vec<&Tensor> = carry.iter().collect();
        let out = g.run(SegKey { tag: "final", layer: n, extra: vec![] }, &inputs, |ins| {
            let x = self.graph_layer_input(n, ins)?;
            Ok(vec![self.norm.forward(&x)?])
        })?;
        // detach from the graph's output buffer: the MTP state keeps the trunk hidden past this step
        let x = out[0].copy()?;
        static STEPS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        if STEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 1024 == 2 {
            tracing::info!("titan graphs: {}", g.report(&self.device));
        }
        if self.mtp.is_some() {
            self.mtp_after_trunk(input_ids, &x, ctx)?;
        }
        Ok(x)
    }

    /// Verify-path twin of `graph_layer_input`: `carry` = `[x_attn, flat, routed, weights, gates]` or `[x]`.
    #[cfg(feature = "cuda")]
    fn graph_rows_input(&self, i: usize, carry: &[Tensor]) -> Result<Tensor> {
        if i == 0 {
            return Ok(carry[0].clone());
        }
        let Ffn::Moe(moe) = &self.layers[i - 1].ffn else {
            candle_core::bail!("qwen35 graphs: layer {} is not MoE", i - 1);
        };
        let x_attn = &carry[0];
        let h = moe.finish_rows(&carry[1], &carry[2], &carry[3], &carry[4], x_attn.dtype())?;
        h + x_attn
    }

    /// Verify-path twin of `graph_route`: `[x_attn, flat, ids, weights, gates]`.
    #[cfg(feature = "cuda")]
    fn graph_route_rows(&self, i: usize, mixer_out: Tensor, residual: &Tensor) -> Result<Vec<Tensor>> {
        let layer = &self.layers[i];
        let Ffn::Moe(moe) = &layer.ffn else {
            candle_core::bail!("qwen35 graphs: layer {i} is not MoE");
        };
        let x_attn = (mixer_out + residual)?;
        let h = layer.post_attention_norm.forward(&x_attn)?;
        let (flat, topk, gates) = moe.route_rows(&h)?;
        Ok(vec![x_attn, flat, topk.indices, topk.values, gates])
    }

    /// `forward_verify` (batched rows, not `TITAN_MTP_VERIFY=rowwise`) as piecewise CUDA graphs, cut
    /// like `forward_trunk_graphed`; segments are keyed by the row count. The GDN state snapshots taken
    /// for rollback are graph outputs: `SpecRollback::keep` consumes them before the next verification.
    #[cfg(feature = "cuda")]
    fn forward_verify_graphed(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        use super::titan_graph::{dptr, SegKey};
        let g = &self.graphs;
        let n = input_ids.dim(1)?;
        let start = ctx
            .seqlen_offsets()
            .first()
            .copied()
            .ok_or_else(|| candle_core::Error::msg("qwen35 verify: missing seqlen offset"))?;
        let recurrent_metadata = ctx.recurrent_metadata().cloned().ok_or_else(|| {
            candle_core::Error::msg("qwen35: hybrid recurrent metadata is required for the linear-attention layers")
        })?;
        let slot = recurrent_metadata
            .state_indices_host()
            .and_then(|h| h.first().copied())
            .ok_or_else(|| candle_core::Error::msg("qwen35 verify: missing recurrent slot"))? as usize;
        let flash_params = ctx.flash_params();
        let positions = crate::pipeline::text_positions_tensor(&[start], n, &self.device)?;
        let indices = recurrent_metadata.state_indices().clone();
        let host = recurrent_metadata.state_indices_host();

        let x = self
            .tok_embeddings
            .forward(input_ids, &self.device)?
            .to_dtype(DType::F32)?;
        let mut hybrid_cache = self.cache.hybrid();
        let mut gdn_record = Vec::new();
        let mut carry = vec![x];
        for (i, layer) in self.layers.iter().enumerate() {
            let Ffn::Moe(moe) = &layer.ffn else {
                candle_core::bail!("qwen35 graphs: layer {i} is not MoE");
            };
            let routed_in = match (&layer.mixer, hybrid_cache.get_mut(i)) {
                (Mixer::Linear(gdn), Some(HybridLayerCache::Recurrent(pool))) => {
                    let key = SegKey {
                        tag: "v.gdn",
                        layer: i,
                        extra: vec![n as u64, slot as u64, dptr(&pool.conv_state)?, dptr(&pool.recurrent_state)?],
                    };
                    let mut inputs: Vec<&Tensor> = carry.iter().collect();
                    inputs.push(&indices);
                    let mut outs = g.run(key, &inputs, |ins| {
                        let (carry_in, idx) = ins.split_at(ins.len() - 1);
                        let x = self.graph_rows_input(i, carry_in)?;
                        let h = layer.attn_norm.forward(&x)?;
                        let mut gdn_cache = GdnLayerCache {
                            conv_state: pool.gather_conv_state(&idx[0])?,
                            recurrent_state: pool.gather_recurrent_state(&idx[0])?,
                        };
                        let mut snaps = Vec::with_capacity(2 * (n - 1));
                        let out = gdn.forward_decode_rows(
                            &h.to_dtype(self.gdn_dtype)?,
                            &mut gdn_cache,
                            &|w, x| lin_rows(w, x),
                            &mut |r, c| {
                                if r + 1 < n {
                                    snaps.push(c.conv_state.copy()?);
                                    snaps.push(c.recurrent_state.copy()?);
                                }
                                Ok(())
                            },
                        )?;
                        pool.scatter_conv_state_with_host_indices(&idx[0], host, &gdn_cache.conv_state)?;
                        pool.scatter_recurrent_state_with_host_indices(&idx[0], host, &gdn_cache.recurrent_state)?;
                        let mut o = self.graph_route_rows(i, out.to_dtype(DType::F32)?, &x)?;
                        o.extend(snaps);
                        Ok(o)
                    })?;
                    let snaps = outs.split_off(5);
                    gdn_record.push((i, snaps.chunks(2).map(|c| (c[0].clone(), c[1].clone())).collect::<Vec<_>>()));
                    outs
                }
                (Mixer::Attention(attn), Some(HybridLayerCache::Attention(kv_cache))) => {
                    let mut inputs: Vec<&Tensor> = carry.iter().collect();
                    inputs.push(&positions);
                    let pre = g.run(SegKey { tag: "v.attn.pre", layer: i, extra: vec![n as u64] }, &inputs, |ins| {
                        let (carry_in, pos) = ins.split_at(ins.len() - 1);
                        let x = self.graph_rows_input(i, carry_in)?;
                        let h = layer.attn_norm.forward(&x)?;
                        let (q, k, v, gate) = attn.decode_rows_qkv(&h, &pos[0])?;
                        Ok(vec![x, q, k, v, gate])
                    })?;
                    let (x, q, k, v, gate) = (&pre[0], &pre[1], &pre[2], &pre[3], &pre[4]);
                    let y = attn.decode_rows_attend(q, k, v, kv_cache, Some(flash_params))?.contiguous()?;
                    g.run(SegKey { tag: "v.attn.post", layer: i, extra: vec![n as u64] }, &[&y, gate, x], |ins| {
                        let out = attn.decode_rows_out(&ins[0], &ins[1], ins[2].dtype())?;
                        self.graph_route_rows(i, out, &ins[2])
                    })?
                }
                _ => candle_core::bail!("qwen35: layer {i} cache kind does not match its mixer"),
            };
            let (flat, ids) = (&routed_in[1], &routed_in[2]);
            let (rows, hidden) = flat.dims2()?;
            let routed = moe.experts.forward(&flat.reshape((rows, 1, hidden))?, ids)?;
            carry = vec![routed_in[0].clone(), flat.clone(), routed, routed_in[3].clone(), routed_in[4].clone()];
        }
        drop(hybrid_cache);

        let nl = self.layers.len();
        let inputs: Vec<&Tensor> = carry.iter().collect();
        let out = g.run(SegKey { tag: "v.final", layer: nl, extra: vec![n as u64] }, &inputs, |ins| {
            let x = self.graph_rows_input(nl, ins)?;
            Ok(vec![self.norm.forward(&x)?])
        })?;
        // detach from the graph's output buffer: `st.last` keeps the normed rows past this step
        let normed = out[0].copy()?;
        static STEPS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        if STEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 256 == 2 {
            tracing::info!("titan graphs: {}", g.report(&self.device));
        }
        let logits = lin_rows(&self.output, &normed.contiguous()?)?;
        {
            let mut st = self.spec.lock().expect("spec state poisoned");
            st.last = Some((start, normed.squeeze(0)?));
            st.verify = Some(VerifyRecord {
                start,
                rows: n,
                slot,
                gdn: gdn_record,
            });
        }
        Ok(logits)
    }

    pub(crate) fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    pub(crate) fn mtp_draft_len(&self) -> Option<usize> {
        self.mtp.as_ref().map(|m| m.n_draft)
    }

    /// Final-normed trunk hidden of `(batch index, row)` of the last forward; one sequence only.
    pub(crate) fn mtp_target_hiddens(&self, rows: &[(usize, usize)]) -> Result<Option<Tensor>> {
        if self.mtp.is_none() || rows.len() != 1 || rows[0].0 != 0 {
            return Ok(None);
        }
        let st = self.spec.lock().expect("spec state poisoned");
        let Some((_, last)) = &st.last else {
            return Ok(None);
        };
        if rows[0].1 >= last.dim(0)? {
            return Ok(None);
        }
        Ok(Some(last.narrow(0, rows[0].1, 1)?))
    }

    /// Draft `n_draft` tokens after `toks` (whose last token, at position `pos_c`, was just sampled
    /// by the trunk): catch the MTP KV cache up with the verified rows `mtp_len..=pos_c`, then run
    /// the block recurrently on its own greedy drafts.
    pub(crate) fn mtp_propose(&self, toks: &[u32], pos_c: usize) -> Result<Option<Vec<u32>>> {
        let Some(mtp) = &self.mtp else {
            return Ok(None);
        };
        if std::env::var("TITAN_MTP_NODRAFT").is_ok_and(|v| v == "1") {
            return Ok(None); // A/B: MTP block loaded (same VRAM plan), plain decoding
        }
        let t0 = std::time::Instant::now();
        let mut st = self.spec.lock().expect("spec state poisoned");
        if !st.mtp_valid
            || toks.len() != pos_c + 1
            || st.mtp_len > pos_c
            || pos_c + mtp.n_draft + 1 >= self.max_seq_len
        {
            return Ok(None);
        }
        let Some((start, last)) = st.last.clone() else {
            return Ok(None);
        };
        // h_{p-1} for every MTP row p in mtp_len..=pos_c
        let mut h_rows = Vec::with_capacity(pos_c + 1 - st.mtp_len);
        for p in st.mtp_len..=pos_c {
            let q = p.checked_sub(1);
            let h = match q {
                Some(q) if q >= start && q < start + last.dim(0)? => last.narrow(0, q - start, 1)?,
                Some(q) if st.pending.as_ref().is_some_and(|(pp, _)| *pp == q) => {
                    st.pending.as_ref().unwrap().1.clone()
                }
                None => Tensor::zeros((1, last.dim(1)?), DType::F32, last.device())?,
                _ => {
                    tracing::warn!("titan mtp: no trunk hidden for position {p}, drafting stops for this sequence");
                    st.mtp_valid = false;
                    return Ok(None);
                }
            };
            h_rows.push(h);
        }
        let mut kv = st.mtp_kv.take().expect("mtp kv");
        let res = (|| -> Result<Vec<u32>> {
            kv.set_len(st.mtp_len)?;
            let tokens = Tensor::new(&toks[st.mtp_len..=pos_c], &self.device)?.unsqueeze(0)?;
            let (logits, mut h) =
                self.mtp_forward(mtp, &mut kv, &tokens, &Tensor::cat(&h_rows, 0)?, st.mtp_len, true)?;
            let mut draft = logits.expect("logits").argmax(D::Minus1)?.to_scalar::<u32>()?;
            let mut drafts = vec![draft];
            for i in 1..mtp.n_draft {
                let tok = Tensor::new(&[draft], &self.device)?.unsqueeze(0)?;
                let (logits, h_next) = self.mtp_forward(mtp, &mut kv, &tok, &h, pos_c + i, true)?;
                draft = logits.expect("logits").argmax(D::Minus1)?.to_scalar::<u32>()?;
                drafts.push(draft);
                h = h_next;
            }
            Ok(drafts)
        })();
        st.mtp_kv = Some(kv);
        let drafts = res?;
        let mut tp = t0;
        prof("propose", &self.device, &mut tp);
        st.mtp_len = pos_c + 1;
        st.stats.draft_time += t0.elapsed();
        Ok(Some(drafts))
    }

    /// Rollback handle for the speculative verifier (see `speculative::unpaged`).
    pub(crate) fn spec_rollback(&self) -> Arc<dyn crate::speculative::unpaged::SpeculativeRollback> {
        let EitherCache::Hybrid(hybrid) = &self.cache else {
            unreachable!("qwen35 always has a hybrid cache")
        };
        Arc::new(SpecRollback {
            spec: self.spec.clone(),
            hybrid: hybrid.clone(),
        })
    }
}

/// Prefix cache (see `PrefixCacheManagerV2::search_hybrid`). A resume point is the recurrent state
/// after `len` tokens: the GDN layers cannot rewind, so it is saved where a later request can pick
/// it up (the last chunk boundary of a prompt pass, the prompt's end, the finished sequence's end).
/// With an MTP block the point also keeps the trunk hidden at `len - 1` and the MTP KV holds
/// `len` rows, so drafting continues after a resume as after an uninterrupted prompt pass.
impl ModelWeights {
    /// The trunk hidden at `pos` from the last forward's rows or the pending prompt row.
    fn hidden_at(st: &SpecState, pos: usize) -> Result<Option<Tensor>> {
        if let Some((start, last)) = &st.last {
            if pos >= *start && pos < start + last.dim(0)? {
                return Ok(Some(last.narrow(0, pos - start, 1)?));
            }
        }
        Ok(match &st.pending {
            Some((p, h)) if *p == pos => Some(h.clone()),
            _ => None,
        })
    }

    fn gdn_to_host(snaps: Vec<RecurrentStateSnapshot>) -> Result<Vec<RecurrentStateSnapshot>> {
        snaps
            .into_iter()
            .map(|s| {
                Ok(RecurrentStateSnapshot {
                    conv_state: s.conv_state.to_device(&Device::Cpu)?,
                    recurrent_state: s.recurrent_state.to_device(&Device::Cpu)?,
                    state_layout: s.state_layout,
                })
            })
            .collect()
    }

    /// Save the recurrent state of `slot` (just after its first `len` tokens) as a resume point.
    fn save_prefix_point(&self, slot: usize, len: usize, boundary: bool) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let mtp_hidden = if self.mtp.is_some() {
            let st = self.spec.lock().expect("spec state poisoned");
            match &st.pending {
                Some((pos, h)) if st.mtp_valid && st.mtp_len == len && pos + 1 == len => {
                    Some(h.to_device(&Device::Cpu)?)
                }
                // drafting could not continue from here
                _ => return Ok(()),
            }
        } else {
            None
        };
        let gdn = Self::gdn_to_host(self.cache.hybrid().titan_snapshot_physical(slot)?)?;
        let mut points = self.prefix_points.lock().expect("prefix points poisoned");
        let v = points.entry(slot).or_default();
        v.retain(|p| p.len != len);
        v.push(Arc::new(HybridPrefixPoint { len, boundary, gdn, mtp_hidden }));
        Ok(())
    }

    /// Bring the MTP block's KV up to `p` verified rows (a finished sequence's last accepted rows
    /// were not drafted from yet) and return its host copy and the trunk hidden at `p - 1`.
    fn mtp_export(&self, toks: &[u32], p: usize) -> Result<Option<((Tensor, Tensor), Tensor, usize)>> {
        let Some(mtp) = &self.mtp else {
            return Ok(None);
        };
        let mut st = self.spec.lock().expect("spec state poisoned");
        if !st.mtp_valid || st.mtp_kv.is_none() {
            return Ok(None);
        }
        let m = st.mtp_len.min(p);
        let mut h_rows = Vec::with_capacity(p - m);
        for q in m..p {
            let h = match q.checked_sub(1) {
                Some(q1) => Self::hidden_at(&st, q1)?,
                None => None,
            };
            match h {
                Some(h) => h_rows.push(h),
                None => break,
            }
        }
        let reach = m + h_rows.len();
        let mut kv = st.mtp_kv.take().expect("mtp kv");
        let res = (|| -> Result<()> {
            kv.set_len(m)?;
            if reach > m {
                let tokens = Tensor::new(&toks[m..reach], &self.device)?.unsqueeze(0)?;
                self.mtp_forward(mtp, &mut kv, &tokens, &Tensor::cat(&h_rows, 0)?, m, false)?;
            }
            Ok(())
        })();
        st.mtp_kv = Some(kv);
        res?;
        st.mtp_len = reach;
        let Some(h_last) = reach.checked_sub(1).map(|q| Self::hidden_at(&st, q)).transpose()?.flatten() else {
            return Ok(None);
        };
        let Some(host) = kv_to_host(st.mtp_kv.as_ref().expect("mtp kv"), reach)? else {
            return Ok(None);
        };
        Ok(Some((host, h_last.to_device(&Device::Cpu)?, reach)))
    }

    /// Host copies of the caches of the sequence the last step ran alone (it just finished): its
    /// attention KV, the recurrent state at its end and the points saved during its prompt passes.
    pub(crate) fn prefix_export(&self, toks: &[u32]) -> Result<Option<HybridPrefixEntry>> {
        if !crate::prefix_cacher::hybrid_prefix_cache_on() {
            return Ok(None);
        }
        let hybrid = self.cache.hybrid();
        let slot = match hybrid.state_indices_host() {
            Some([slot]) => *slot as usize,
            _ => return Ok(None),
        };
        let p = hybrid.get_past_kv_len()?;
        if p == 0 || p > toks.len() {
            return Ok(None);
        }
        let mut kv = Vec::with_capacity(hybrid.num_layers());
        for c in &hybrid.caches {
            kv.push(match c {
                HybridLayerCache::Attention(layer) => match kv_to_host(layer, p)? {
                    Some(host) => Some(host),
                    None => return Ok(None),
                },
                HybridLayerCache::Recurrent(_) => None,
            });
        }
        let gdn_end = Self::gdn_to_host(hybrid.titan_snapshot_physical(slot)?)?;
        drop(hybrid);
        let mut points = self
            .prefix_points
            .lock()
            .expect("prefix points poisoned")
            .remove(&slot)
            .unwrap_or_default();
        let mut mtp_kv = None;
        if self.mtp.is_some() {
            // points need the MTP rows before them; the end point also the hidden at p - 1
            match self.mtp_export(toks, p)? {
                Some((host, h_last, reach)) => {
                    points.retain(|pt| pt.len <= reach);
                    if reach == p {
                        points.push(Arc::new(HybridPrefixPoint { len: p, boundary: false, gdn: gdn_end, mtp_hidden: Some(h_last) }));
                    }
                    mtp_kv = Some(host);
                }
                None => points.clear(),
            }
        } else {
            points.push(Arc::new(HybridPrefixPoint { len: p, boundary: false, gdn: gdn_end, mtp_hidden: None }));
        }
        points.retain(|pt| pt.len <= p);
        if points.is_empty() {
            return Ok(None);
        }
        Ok(Some(HybridPrefixEntry { tokens: toks[..p].to_vec(), kv, mtp_kv, points }))
    }

    /// Put a prefix-cache hit into place before the sequence's prompt step: its attention KV (cut
    /// to the resume point, with room for the whole prompt), the recurrent state of its slot, the
    /// MTP KV and pending hidden, and the entry's points up to the resume point.
    pub(crate) fn prefix_restore(&self, seq: &mut crate::sequence::Sequence) -> Result<()> {
        let Some(resume) = seq.take_hybrid_resume() else {
            return Ok(());
        };
        let q = resume.len;
        let entry = &resume.entry;
        let slot = seq
            .recurrent_state_idx()
            .ok_or_else(|| candle_core::Error::msg("prefix restore: sequence has no recurrent slot"))?;
        let point = entry
            .points
            .iter()
            .find(|p| p.len == q)
            .ok_or_else(|| candle_core::Error::msg("prefix restore: no resume point"))?;
        // the sequence holds the prompt tokens after the resume point
        let total = q + seq.get_toks().len();
        let cap = (total + 1).next_multiple_of(KV_RESERVE_STEP).min(self.max_seq_len).max(q);
        let mut hybrid = self.cache.hybrid();
        let max_seq_len = hybrid.config().max_seq_len;
        let mut caches = Vec::with_capacity(entry.kv.len());
        for kv in &entry.kv {
            caches.push(match kv {
                Some((k, v)) => Some(kv_from_host(k, v, q, cap, max_seq_len, &self.device)?),
                None => None,
            });
        }
        hybrid.titan_restore_slot(slot, &point.gdn)?;
        drop(hybrid);
        *seq.normal_cache() = caches;
        if self.mtp.is_some() {
            let mut st = self.spec.lock().expect("spec state poisoned");
            if st.stats.steps > 0 || mtp_prof_wall() || mtp_prof_on() {
                log_mtp_stats(&st.stats);
            }
            st.last = None;
            st.verify = None;
            match (&entry.mtp_kv, &point.mtp_hidden) {
                (Some((k, v)), Some(h)) if k.dim(2)? >= q => {
                    st.mtp_kv = Some(kv_from_host(k, v, q, cap, self.max_seq_len, &self.device)?);
                    st.mtp_len = q;
                    st.pending = Some((q - 1, h.to_device(&self.device)?));
                    st.mtp_valid = true;
                }
                _ => {
                    st.mtp_kv = None;
                    st.pending = None;
                    st.mtp_valid = false;
                }
            }
        }
        self.prefix_points
            .lock()
            .expect("prefix points poisoned")
            .insert(slot, entry.points.iter().filter(|p| p.len <= q).cloned().collect());
        Ok(())
    }
}

struct SpecRollback {
    spec: Arc<Mutex<SpecState>>,
    hybrid: Arc<Mutex<HybridCache>>,
}

impl crate::speculative::unpaged::SpeculativeRollback for SpecRollback {
    /// Keep the first `keep_len` positions: the verification's rows up to and including the last
    /// accepted draft. Attention layers truncate; GDN layers get the state saved after that row.
    fn keep(&self, keep_len: usize) -> Result<()> {
        let mut st = self.spec.lock().expect("spec state poisoned");
        let Some(rec) = st.verify.take() else {
            return Ok(());
        };
        let kept = keep_len.saturating_sub(rec.start).clamp(1, rec.rows);
        let stats = &mut st.stats;
        stats.steps += 1;
        stats.drafted += (rec.rows - 1) as u64;
        stats.accepted += (kept - 1) as u64;
        crate::titan_monitor::mtp_step((rec.rows - 1) as u64, (kept - 1) as u64);
        for i in 0..kept - 1 {
            stats.by_pos[i.min(7)] += 1;
        }
        if stats.steps.is_multiple_of(200) {
            log_mtp_stats(stats);
        }
        if kept < rec.rows {
            let mut hybrid = self.hybrid.lock().expect("hybrid cache poisoned");
            for (layer, snaps) in &rec.gdn {
                let (conv, recurrent) = &snaps[kept - 1];
                let Some(HybridLayerCache::Recurrent(pool)) = hybrid.get_mut(*layer) else {
                    candle_core::bail!("qwen35 rollback: layer {layer} is not recurrent");
                };
                pool.scatter_conv_state_for_indices(&[rec.slot as u32], conv)?;
                pool.scatter_recurrent_state_for_indices(&[rec.slot as u32], recurrent)?;
            }
            hybrid.truncate_attention_to(rec.start + kept)?;
        }
        Ok(())
    }
}

fn log_mtp_stats(s: &MtpStats) {
    let rate = if s.drafted > 0 { s.accepted as f64 / s.drafted as f64 } else { 0.0 };
    let by_pos = s
        .by_pos
        .iter()
        .take_while(|&&c| c > 0)
        .map(|c| format!("{:.2}", *c as f64 / s.steps.max(1) as f64))
        .collect::<Vec<_>>()
        .join(" ");
    if mtp_prof_on() || mtp_prof_wall() {
        tracing::info!("titan mtp prof: {}", prof_report());
    }
    tracing::info!(
        "titan mtp stats: {} verify steps, {} drafted, {} accepted ({:.1}%), {:.2} tokens/step, per-position acceptance [{by_pos}], draft time {:.0} ms",
        s.steps,
        s.drafted,
        s.accepted,
        100.0 * rate,
        1.0 + s.accepted as f64 / s.steps.max(1) as f64,
        s.draft_time.as_secs_f64() * 1e3
    );
}

/// Log-probabilities kept per position by `TITAN_TF_DUMP`.
const TF_TOPK: usize = 64;

/// `TITAN_TF_DUMP=<file>` (evaluation): every chunk of a batch-1 chunked prompt pass also runs the LM head over all
/// its positions and appends `u32 first position, u32 rows`, then per position `u32 next token (u32::MAX past the
/// end), f32 its log-probability, TF_TOPK x (u32 id, f32 log-probability)` (teacher-forced KL / perplexity).
fn tf_dump(output: &Arc<dyn QuantMethod>, x: &Tensor, tokens: &[u32], start: usize) -> Result<()> {
    static FILE: std::sync::OnceLock<Option<std::sync::Mutex<std::fs::File>>> = std::sync::OnceLock::new();
    let Some(f) = FILE.get_or_init(|| {
        let path = std::env::var("TITAN_TF_DUMP").ok()?;
        std::fs::OpenOptions::new().create(true).append(true).open(path).ok().map(std::sync::Mutex::new)
    }) else {
        return Ok(());
    };
    if x.dim(0)? != 1 {
        return Ok(());
    }
    use std::io::Write;
    let logits: Vec<Vec<f32>> = output.forward(&x.contiguous()?)?.to_dtype(DType::F32)?.squeeze(0)?.to_vec2()?;
    let mut buf = Vec::with_capacity(8 + logits.len() * (8 + TF_TOPK * 8));
    buf.extend_from_slice(&(start as u32).to_le_bytes());
    buf.extend_from_slice(&(logits.len() as u32).to_le_bytes());
    let mut idx: Vec<u32> = Vec::new();
    for (i, row) in logits.iter().enumerate() {
        let max = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
        let lse = max + row.iter().map(|&v| (v - max).exp() as f64).sum::<f64>().ln() as f32;
        let next = tokens.get(start + i + 1).copied().unwrap_or(u32::MAX);
        let lp = if (next as usize) < row.len() { row[next as usize] - lse } else { 0.0 };
        buf.extend_from_slice(&next.to_le_bytes());
        buf.extend_from_slice(&lp.to_le_bytes());
        idx.clear();
        idx.extend(0..row.len() as u32);
        idx.select_nth_unstable_by(TF_TOPK, |&a, &b| row[b as usize].total_cmp(&row[a as usize]));
        idx.truncate(TF_TOPK);
        idx.sort_by(|&a, &b| row[b as usize].total_cmp(&row[a as usize]));
        for &t in &idx {
            buf.extend_from_slice(&t.to_le_bytes());
            buf.extend_from_slice(&(row[t as usize] - lse).to_le_bytes());
        }
    }
    if let Ok(mut f) = f.lock() {
        let _ = f.write_all(&buf);
    }
    Ok(())
}

/// `TITAN_LAYER_DUMP_BIN=<file>`: every layer output (record `layer`) and mixer output plus residual
/// (record `1000 + layer`) of every prompt pass (or prompt chunk) of batch 1,
/// exact: per record `layer, first position, rows, cols` as u32 LE, then the rows as F32 LE. For
/// diffing a chunked prompt pass against the whole one (see titan-engine m4/chunk/layerbin.py).
fn layer_dump_bin(layer: usize, start: usize, x: &Tensor) -> Result<()> {
    static FILE: std::sync::OnceLock<Option<std::sync::Mutex<std::fs::File>>> = std::sync::OnceLock::new();
    let Some(f) = FILE.get_or_init(|| {
        let path = std::env::var("TITAN_LAYER_DUMP_BIN").ok()?;
        std::fs::File::create(path).ok().map(std::sync::Mutex::new)
    }) else {
        return Ok(());
    };
    if x.dim(0)? != 1 {
        return Ok(());
    }
    use std::io::Write;
    let rows: Vec<Vec<f32>> = x.to_dtype(DType::F32)?.squeeze(0)?.to_vec2()?;
    let mut buf = Vec::with_capacity(16 + rows.len() * rows.first().map_or(0, Vec::len) * 4);
    for v in [layer, start, rows.len(), rows.first().map_or(0, Vec::len)] {
        buf.extend_from_slice(&(v as u32).to_le_bytes());
    }
    for v in rows.iter().flatten() {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    if let Ok(mut f) = f.lock() {
        let _ = f.write_all(&buf);
    }
    Ok(())
}

/// `TITAN_LAYER_DUMP=<file>`: per layer output of the prompt pass, in the shape llama.cpp's
/// llama-eval-callback prints `l_out-N` (sum over the tensor, first/last 3 values of each token row),
/// for a layer-by-layer diff against llama.cpp (see titan-engine m4/layerdiff.py).
fn layer_dump(layer: usize, x: &Tensor) -> Result<()> {
    static FILE: std::sync::OnceLock<Option<std::sync::Mutex<std::fs::File>>> = std::sync::OnceLock::new();
    let Some(f) = FILE.get_or_init(|| {
        let path = std::env::var("TITAN_LAYER_DUMP").ok()?;
        std::fs::File::create(path).ok().map(std::sync::Mutex::new)
    }) else {
        return Ok(());
    };
    use std::io::Write;
    let rows: Vec<Vec<f32>> = x.to_dtype(DType::F32)?.squeeze(0)?.to_vec2()?;
    let sum: f64 = rows.iter().flatten().map(|v| *v as f64).sum();
    let mut line = format!("{layer} {sum:.6}");
    for r in &rows {
        let n = r.len();
        for v in r[..3].iter().chain(&r[n - 3..]) {
            line.push_str(&format!(" {v:.4}"));
        }
    }
    line.push('\n');
    if let Ok(mut f) = f.lock() {
        let _ = f.write_all(line.as_bytes());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{permute_units, split_ba, v_gather_index, v_head_perm};

    // qwen3next.cpp: mixed_ba viewed as [2r, n_k], b = elements 0..r of each K head's group, a = r..2r
    #[test]
    fn split_ba_matches_llama_cpp_views() {
        let (n_k, r, rb) = (4, 3, 2);
        let rows: Vec<u8> = (0..2 * r * n_k).flat_map(|j| [j as u8; 2]).collect();
        let (b, a) = split_ba(&rows, rb, n_k, r);
        let first = |v: &[u8]| v.chunks(rb).map(|c| c[0] as usize).collect::<Vec<_>>();
        let want_b: Vec<usize> = (0..n_k).flat_map(|kh| (0..r).map(move |i| kh * 2 * r + i)).collect();
        let want_a: Vec<usize> = want_b.iter().map(|j| j + r).collect();
        assert_eq!(first(&b), want_b);
        assert_eq!(first(&a), want_a);
    }

    // conversion/qwen.py _reorder_v_heads: [n_k, r, d] -> [r, n_k, d]
    fn converter_reorder(grouped: &[u8], n_k: usize, r: usize, d: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(grouped.len());
        for vi in 0..r {
            for kh in 0..n_k {
                let h = kh * r + vi;
                out.extend_from_slice(&grouped[h * d..(h + 1) * d]);
            }
        }
        out
    }

    #[test]
    fn v_gather_index_matches_converter() {
        // the activation must be reordered exactly as the converter reordered ssm_out's input columns
        for (n_k, n_v, d) in [(16, 32, 128), (16, 48, 128), (4, 12, 3)] {
            let grouped: Vec<u32> = (0..(n_v * d) as u32).collect();
            let idx = v_gather_index(&v_head_perm(n_k, n_v), d);
            let gathered: Vec<u32> = idx.iter().map(|&i| grouped[i as usize]).collect();
            let bytes: Vec<u8> = grouped.iter().flat_map(|v| v.to_le_bytes()).collect();
            let tiled: Vec<u32> = converter_reorder(&bytes, n_k, n_v / n_k, d * 4)
                .chunks(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(gathered, tiled, "n_k={n_k} n_v={n_v}");
        }
    }

    #[test]
    fn v_head_perm_inverts_converter() {
        for (n_k, n_v, d) in [(16, 32, 3), (16, 48, 3), (16, 48, 1), (4, 20, 2), (8, 8, 2)] {
            let grouped: Vec<u8> = (0..n_v * d).map(|i| i as u8).collect();
            let mut data = converter_reorder(&grouped, n_k, n_v / n_k, d);
            permute_units(&mut data, 0, 1, 0, d, &v_head_perm(n_k, n_v));
            assert_eq!(data, grouped, "n_k={n_k} n_v={n_v}");
        }
    }
}
