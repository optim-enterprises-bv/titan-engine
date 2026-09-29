#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::MemoryUsage;

use candle_core::{Device, Result, Tensor};
use mistralrs_quant::MatMul;

use crate::attention::{chunked_attention, SdpaParams};

/// Queries up to this long (a decode step, a speculative draft / verification) allocate almost nothing.
const SMALL_QUERY: usize = 16;

/// Before a prompt-pass attention, sync when free VRAM is low so pending stream-ordered frees land
/// before the large score allocation. A short query (decode, speculative draft or verification) allocates
/// almost nothing, so it skips both the sync and the ~20 us free-memory query (which ran on every layer
/// of every token and, below 4 GiB free, forced a device sync per layer). Timing only: results unchanged.
pub(crate) fn maybe_synchronize_for(q: &Tensor) -> Result<()> {
    if q.rank() == 4 && q.dim(2)? <= SMALL_QUERY {
        return Ok(());
    }
    maybe_synchronize(q.device())
}

/// Not *really* sure why this is necessary but it is.
pub(crate) fn maybe_synchronize(device: &Device) -> Result<()> {
    if matches!(device, Device::Cpu) {
        return Ok(());
    }

    // If less that 4 GB available, synchronize
    #[cfg(target_pointer_width = "64")]
    const FOUR_GIB: usize = 4 * 1024 * 1024 * 1024;
    #[cfg(not(target_pointer_width = "64"))]
    const FOUR_GIB: usize = usize::MAX;
    if MemoryUsage.query(device)?.available() < FOUR_GIB {
        device.synchronize()?;
    }
    Ok(())
}

/// Computes softmax(QK^T*sqrt(d_k))V
pub(crate) fn naive_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    maybe_synchronize_for(q)?;

    // Use chunked attention with a closure that captures the necessary parameters
    chunked_attention(q, k, v, mask, |q_chunk, k, v, mask_chunk| {
        let mut att =
            MatMul.matmul_affine_mul(q_chunk, &k.t()?, sdpa_params.softmax_scale.into())?;

        if let Some(softcap) = sdpa_params.softcap {
            att = (att / softcap as f64)?;
            att = att.tanh()?;
            att = (att * softcap as f64)?;
        }

        if let Some(mask) = mask_chunk {
            att = att.broadcast_add(mask)?;
        }

        // Compute softmax in F32 for precision (BF16 exp() loses information).
        let att_dtype = att.dtype();
        if att_dtype == candle_core::DType::BF16 || att_dtype == candle_core::DType::F16 {
            att = att.to_dtype(candle_core::DType::F32)?;
        }
        att = candle_nn::ops::softmax_last_dim(&att)?;
        if att.dtype() != att_dtype {
            att = att.to_dtype(att_dtype)?;
        }
        MatMul.matmul(&att, v)
    })
}
