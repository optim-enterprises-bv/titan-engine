//! MLA weight caching for efficient decode operations.

#[cfg(all(feature = "cuda", target_family = "unix"))]
use std::sync::Mutex;

#[cfg(all(feature = "cuda", target_family = "unix"))]
use candle_core::{Device, Result, Tensor, D};

#[cfg(all(feature = "cuda", target_family = "unix"))]
use mistralrs_quant::QuantMethod;

/// Cached MLA weight matrices for efficient decode operations.
///
/// Stores the precomputed w_uk and w_uv_t matrices extracted from kv_b_proj.
/// These are computed lazily on first use and cached for subsequent calls.
pub struct MlaWeights {
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    weights: Option<Mutex<Option<(Tensor, Tensor)>>>,
    #[cfg(not(all(feature = "cuda", target_family = "unix")))]
    _phantom: std::marker::PhantomData<()>,
}

impl MlaWeights {
    /// Create a new MlaWeights instance.
    ///
    /// If `paged_attn_enabled` is true and we're on CUDA, allocates the mutex for caching.
    /// Otherwise, the weights are not cached (MLA decode won't be used).
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    pub fn new(paged_attn_enabled: bool, device: Option<&Device>) -> Self {
        let weights = if paged_attn_enabled {
            if let Some(device) = device {
                if matches!(device, Device::Cuda(_)) {
                    Some(Mutex::new(None))
                } else {
                    None
                }
            } else {
                // If no device is provided, assume we may need it
                Some(Mutex::new(None))
            }
        } else {
            None
        };
        Self { weights }
    }

    #[cfg(not(all(feature = "cuda", target_family = "unix")))]
    pub fn new(_paged_attn_enabled: bool, _device: Option<&candle_core::Device>) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Compute MLA weights from the kv_b_proj layer.
    ///
    /// Extracts w_uk (for K nope projection) and w_uv_t (transposed V projection)
    /// from the kv_b_proj weight matrix.
    ///
    /// # Arguments
    /// * `kv_b_proj` - The kv_b_proj quantized layer
    /// * `device` - Target device for the weights
    /// * `num_attention_heads` - Number of attention heads
    /// * `kv_lora_rank` - KV latent dimension
    /// * `qk_nope_head_dim` - Non-positional head dimension
    /// * `v_head_dim` - Value head dimension
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    pub fn compute_weights(
        kv_b_proj: &dyn QuantMethod,
        device: &Device,
        num_attention_heads: usize,
        kv_lora_rank: usize,
        qk_nope_head_dim: usize,
        v_head_dim: usize,
    ) -> Result<(Tensor, Tensor)> {
        let mut w = kv_b_proj.dequantize_w()?;
        if !w.device().same_device(device) {
            w = w.to_device(device)?;
        }
        let (out_dim, in_dim) = w.dims2()?;
        if in_dim != kv_lora_rank {
            candle_core::bail!(
                "kv_b_proj weight in_dim mismatch: expected {}, got {}",
                kv_lora_rank,
                in_dim
            );
        }
        let per_head_dim = qk_nope_head_dim + v_head_dim;
        if out_dim != num_attention_heads * per_head_dim {
            candle_core::bail!(
                "kv_b_proj weight out_dim mismatch: expected {}, got {}",
                num_attention_heads * per_head_dim,
                out_dim
            );
        }
        let w = w.reshape((num_attention_heads, per_head_dim, kv_lora_rank))?;
        let w_uk = w.narrow(D::Minus2, 0, qk_nope_head_dim)?.contiguous()?;
        let w_uv = w
            .narrow(D::Minus2, qk_nope_head_dim, v_head_dim)?
            .contiguous()?;
        let w_uv_t = w_uv.transpose(1, 2)?.contiguous()?;
        Ok((w_uk, w_uv_t))
    }

    /// Get or compute the MLA weights.
    ///
    /// Returns cached weights if available, otherwise computes and caches them.
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    pub fn get_or_compute(
        &self,
        kv_b_proj: &dyn QuantMethod,
        device: &Device,
        num_attention_heads: usize,
        kv_lora_rank: usize,
        qk_nope_head_dim: usize,
        v_head_dim: usize,
    ) -> Result<(Tensor, Tensor)> {
        if kv_b_proj.is_dynamic_lora_active() {
            candle_core::bail!(
                "cached MLA weights cannot be used with an active dynamic LoRA projection"
            );
        }
        let Some(mla_weights) = &self.weights else {
            candle_core::bail!("MLA weights are not initialized on this device");
        };
        let mut guard = mla_weights.lock().expect("MLA weights mutex was poisoned");
        if let Some((w_uk, w_uv_t)) = guard.as_ref() {
            return Ok((w_uk.clone(), w_uv_t.clone()));
        }
        let (w_uk, w_uv_t) = Self::compute_weights(
            kv_b_proj,
            device,
            num_attention_heads,
            kv_lora_rank,
            qk_nope_head_dim,
            v_head_dim,
        )?;
        *guard = Some((w_uk.clone(), w_uv_t.clone()));
        Ok((w_uk, w_uv_t))
    }

    #[cfg(not(all(feature = "cuda", target_family = "unix")))]
    #[allow(dead_code)]
    pub fn get_or_compute(
        &self,
        _kv_b_proj: &dyn mistralrs_quant::QuantMethod,
        _device: &candle_core::Device,
        _num_attention_heads: usize,
        _kv_lora_rank: usize,
        _qk_nope_head_dim: usize,
        _v_head_dim: usize,
    ) -> candle_core::Result<(candle_core::Tensor, candle_core::Tensor)> {
        candle_core::bail!("MLA weights require CUDA support")
    }
}

/// m4/deq: `GgufMatMul::dequantize_w` is the exact f32 dequantize (QTensor::dequantize: llama.cpp's to_float /
/// convert.cu f32 values) through the `dyn QuantMethod` every caller holds, on CPU and CUDA, and the MLA caller
/// (`compute_weights`) gets those exact values. The old path (dequantize_f16 -> f32) rounds these weights, so the
/// test fails on it. (Lives in mistralrs-core: the nvcc-free mistralrs-quant test binary cannot link the
/// nvcc-only stubs, which only mistralrs-core's oxide feature pulls in.)
#[cfg(all(test, feature = "cuda", target_family = "unix"))]
mod deq_tests {
    use super::MlaWeights;
    use candle_core::quantized::{GgmlDType, QStorage, QTensor};
    use candle_core::{DType, Device, Result, Tensor};
    use mistralrs_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};
    use std::sync::Arc;

    fn bits(t: &Tensor) -> Result<Vec<u32>> {
        Ok(t.flatten_all()?.to_vec1::<f32>()?.iter().map(|v| v.to_bits()).collect())
    }

    fn check(device: &Device) -> Result<()> {
        // 4 heads x (nope 64 + v 64) rows, kv_lora_rank 512 columns
        let (heads, nope, vd, rank) = (4usize, 64usize, 64usize, 512usize);
        let (rows, cols) = (heads * (nope + vd), rank);
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 7919 % 1013) as f32 - 506.0) / 506.0 * (1.0 + (i / cols) as f32 * 0.037))
            .collect();
        let dense = Tensor::from_vec(values, (rows, cols), &Device::Cpu)?;
        for dtype in [
            GgmlDType::Q4_0,
            GgmlDType::Q8_0,
            GgmlDType::Q4K,
            GgmlDType::Q6K,
            GgmlDType::MXFP4,
            GgmlDType::IQ4XS,
            GgmlDType::Q8_1,
        ] {
            // IQ4_XS (dequantized on the host; its CUDA dequantize_f16 used to fail) cannot be quantized by
            // candle: random blocks with a finite scale d = 0.1.
            let bytes = if dtype == GgmlDType::IQ4XS {
                let mut b: Vec<u8> =
                    (0..rows * cols / 256 * 136).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
                for blk in b.chunks_exact_mut(136) {
                    blk[..2].copy_from_slice(&half::f16::from_f32(0.1).to_bits().to_le_bytes());
                }
                b
            } else {
                QTensor::quantize(&dense, dtype)?.data()?.to_vec()
            };
            let q = QTensor::new(QStorage::from_data(std::borrow::Cow::Owned(bytes), device, dtype)?, (rows, cols))?;
            let want = q.dequantize(device)?;
            let old = q.dequantize_f16(device)?.to_dtype(DType::F32)?;
            let layer: Arc<dyn QuantMethod> =
                Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf { q_weight: Arc::new(q), b: None })?);
            let got = layer.dequantize_w()?;
            assert_eq!(got.dtype(), DType::F32, "{dtype:?}");
            let (g, w, o) = (bits(&got)?, bits(&want)?, bits(&old)?);
            let differ = g.iter().zip(&w).filter(|(a, b)| a != b).count();
            let old_differ = o.iter().zip(&w).filter(|(a, b)| a != b).count();
            assert_eq!(differ, 0, "{dtype:?} on {device:?}: dequantize_w differs from dequantize");
            if !matches!(dtype, GgmlDType::MXFP4 | GgmlDType::IQ4XS) {
                assert!(old_differ > 0, "{dtype:?}: the old f16 path should round some values");
            }
            // the MLA caller: w_uk = rows [h*(nope+v), h*(nope+v)+nope) of the exact weight
            let (w_uk, _w_uv_t) = MlaWeights::compute_weights(layer.as_ref(), device, heads, rank, nope, vd)?;
            let exact_uk = want.reshape((heads, nope + vd, rank))?.narrow(1, 0, nope)?.contiguous()?;
            let uk_differ = bits(&w_uk)?.iter().zip(&bits(&exact_uk)?).filter(|(a, b)| a != b).count();
            assert_eq!(uk_differ, 0, "{dtype:?}: MLA w_uk differs from the exact dequantize");
            println!(
                "dequantize_w {dtype:?} on {}: exact ({} values), MLA w_uk exact; the old f16 path differed on {old_differ}",
                if device.is_cpu() { "cpu" } else { "cuda" },
                w.len()
            );
        }
        Ok(())
    }

    #[test]
    fn dequantize_w_is_exact_f32_cpu() -> Result<()> {
        check(&Device::Cpu)
    }

    #[test]
    fn dequantize_w_is_exact_f32_cuda() -> Result<()> {
        check(&Device::new_cuda(0)?)
    }
}
