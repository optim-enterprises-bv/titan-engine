//! titan `oxide` build: launchers whose C ABI changed between mistral.rs 84b53bf and v0.9.4.
//!
//! titan-oxide-ffi exports the cuda-oxide twins of the 84b53bf launchers under their C names. For
//! a launcher upstream changed, the upstream declaration is either cfg'd out and replaced by an
//! adapter here with the same Rust name and signature (when the old kernel computes the same
//! thing for the arguments it gets), or redirected to an aborting `titan_nvcc_only_*` stub (see
//! the titan-nvcc-only crate). The adapters keep the numerics of the fork's gated builds.
#![allow(clippy::too_many_arguments)]

use std::ffi::c_void;

use candle_core::{DType, Result, Tensor};

extern "C" {
    /// 84b53bf ABI of `qk_rms_norm_rope` (no `output_token_major`).
    #[link_name = "qk_rms_norm_rope"]
    fn qk_rms_norm_rope_v084(
        q: *const c_void,
        k: *const c_void,
        q_weight: *const c_void,
        k_weight: *const c_void,
        cos: *const c_void,
        sin: *const c_void,
        q_out: *mut c_void,
        k_out: *mut c_void,
        q_stride_b: i64,
        q_stride_h: i64,
        q_stride_s: i64,
        q_stride_d: i64,
        k_stride_b: i64,
        k_stride_h: i64,
        k_stride_s: i64,
        k_stride_d: i64,
        batch: i32,
        q_heads: i32,
        k_heads: i32,
        seq_len: i32,
        head_dim: i32,
        rot_dim: i32,
        cos_batch_stride: i32,
        q_eps: f32,
        k_eps: f32,
        is_neox: i32,
        dtype: i32,
        stream: i64,
    );

    /// 84b53bf ABI of `qk_rms_norm_rope_positions` (no `output_token_major`).
    #[link_name = "qk_rms_norm_rope_positions"]
    fn qk_rms_norm_rope_positions_v084(
        q: *const c_void,
        k: *const c_void,
        q_weight: *const c_void,
        k_weight: *const c_void,
        cos: *const c_void,
        sin: *const c_void,
        positions: *const c_void,
        q_out: *mut c_void,
        k_out: *mut c_void,
        q_stride_b: i64,
        q_stride_h: i64,
        q_stride_s: i64,
        q_stride_d: i64,
        k_stride_b: i64,
        k_stride_h: i64,
        k_stride_s: i64,
        k_stride_d: i64,
        batch: i32,
        q_heads: i32,
        k_heads: i32,
        seq_len: i32,
        head_dim: i32,
        rot_dim: i32,
        q_eps: f32,
        k_eps: f32,
        is_neox: i32,
        dtype: i32,
        stream: i64,
    );

    /// 84b53bf ABI of `top1_large_f32_packed` (no device token output).
    #[link_name = "top1_large_f32_packed"]
    fn top1_large_f32_packed_v084(
        input: *const f32,
        block_values: *mut f32,
        block_indices: *mut u32,
        packed_out: *mut f32,
        ncols: i32,
        chunk_size: i32,
        nblocks: i32,
        stream: i64,
    );
}

fn nvcc_only(what: &str) -> ! {
    eprintln!(
        "titan oxide build: {what} needs an nvcc-only kernel variant that is not ported to cuda-oxide"
    );
    std::process::abort()
}

/// v0.9.4 `qk_rms_norm_rope` on the 84b53bf kernel: heads-first output only (the oxide build's
/// `try_cuda_qk_rms_norm_rope` returns `None` for token-major output before launching).
pub(crate) unsafe extern "C" fn qk_rms_norm_rope(
    q: *const c_void,
    k: *const c_void,
    q_weight: *const c_void,
    k_weight: *const c_void,
    cos: *const c_void,
    sin: *const c_void,
    q_out: *mut c_void,
    k_out: *mut c_void,
    q_stride_b: i64,
    q_stride_h: i64,
    q_stride_s: i64,
    q_stride_d: i64,
    k_stride_b: i64,
    k_stride_h: i64,
    k_stride_s: i64,
    k_stride_d: i64,
    batch: i32,
    q_heads: i32,
    k_heads: i32,
    seq_len: i32,
    head_dim: i32,
    rot_dim: i32,
    cos_batch_stride: i32,
    q_eps: f32,
    k_eps: f32,
    is_neox: i32,
    dtype: i32,
    output_token_major: i32,
    stream: i64,
) {
    if output_token_major != 0 {
        nvcc_only("qk_rms_norm_rope with token-major output");
    }
    qk_rms_norm_rope_v084(
        q,
        k,
        q_weight,
        k_weight,
        cos,
        sin,
        q_out,
        k_out,
        q_stride_b,
        q_stride_h,
        q_stride_s,
        q_stride_d,
        k_stride_b,
        k_stride_h,
        k_stride_s,
        k_stride_d,
        batch,
        q_heads,
        k_heads,
        seq_len,
        head_dim,
        rot_dim,
        cos_batch_stride,
        q_eps,
        k_eps,
        is_neox,
        dtype,
        stream,
    )
}

/// v0.9.4 `qk_rms_norm_rope_positions` on the 84b53bf kernel (heads-first output only).
pub(crate) unsafe extern "C" fn qk_rms_norm_rope_positions(
    q: *const c_void,
    k: *const c_void,
    q_weight: *const c_void,
    k_weight: *const c_void,
    cos: *const c_void,
    sin: *const c_void,
    positions: *const c_void,
    q_out: *mut c_void,
    k_out: *mut c_void,
    q_stride_b: i64,
    q_stride_h: i64,
    q_stride_s: i64,
    q_stride_d: i64,
    k_stride_b: i64,
    k_stride_h: i64,
    k_stride_s: i64,
    k_stride_d: i64,
    batch: i32,
    q_heads: i32,
    k_heads: i32,
    seq_len: i32,
    head_dim: i32,
    rot_dim: i32,
    q_eps: f32,
    k_eps: f32,
    is_neox: i32,
    dtype: i32,
    output_token_major: i32,
    stream: i64,
) {
    if output_token_major != 0 {
        nvcc_only("qk_rms_norm_rope_positions with token-major output");
    }
    qk_rms_norm_rope_positions_v084(
        q,
        k,
        q_weight,
        k_weight,
        cos,
        sin,
        positions,
        q_out,
        k_out,
        q_stride_b,
        q_stride_h,
        q_stride_s,
        q_stride_d,
        k_stride_b,
        k_stride_h,
        k_stride_s,
        k_stride_d,
        batch,
        q_heads,
        k_heads,
        seq_len,
        head_dim,
        rot_dim,
        q_eps,
        k_eps,
        is_neox,
        dtype,
        stream,
    )
}

/// Per-device buffers of [`top1_f32_packed`].
struct Top1Workspace {
    ncols: usize,
    nblocks: usize,
    location: candle_core::DeviceLocation,
    block_values: candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
    block_indices: candle_core::cuda_backend::cudarc::driver::CudaSlice<u32>,
    packed: candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
}

static TOP1_WS: std::sync::Mutex<Option<Top1Workspace>> = std::sync::Mutex::new(None);

/// The fork's (84b53bf / ktrace) greedy CUDA top-1 of the last logits row: `[max, index]`.
/// Used for `ops::cuda_top1_logits_f32_cached` in the oxide build, where v0.9.4's submission
/// ring (device token output, BF16/F16 and batched variants) is nvcc-only.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn top1_f32_packed(input: &Tensor) -> Result<[f32; 2]> {
    use candle_core::backend::{BackendDevice, BackendStorage};
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::CudaStorageSlice;

    const CHUNK_SIZE: usize = 2048;

    let input = {
        let dims = input.dims();
        let vocab = *dims.last().unwrap_or(&0);
        if vocab == 0 {
            candle_core::bail!("top1: empty logits");
        }
        let rows = input.elem_count() / vocab;
        input
            .reshape((rows, vocab))?
            .narrow(0, rows - 1, 1)?
            .reshape(vocab)?
            .contiguous()?
    };
    let input = if input.dtype() == DType::F32 {
        input
    } else {
        input.to_dtype(DType::F32)?
    };
    let ncols = input.elem_count();
    let nblocks = ncols.div_ceil(CHUNK_SIZE);

    let (storage, layout) = input.storage_and_layout();
    let storage = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("top1: logits must be on CUDA"),
    };
    let dev = storage.device();
    let location = dev.location();
    let mut guard = TOP1_WS.lock().expect("top1 workspace mutex poisoned");
    if guard.as_ref().is_none_or(|ws| {
        ws.ncols != ncols || ws.nblocks != nblocks || ws.location != location
    }) {
        *guard = Some(Top1Workspace {
            ncols,
            nblocks,
            location,
            block_values: unsafe { dev.alloc::<f32>(nblocks) }?,
            block_indices: unsafe { dev.alloc::<u32>(nblocks) }?,
            packed: unsafe { dev.alloc::<f32>(2) }?,
        });
    }
    let ws = guard.as_mut().expect("allocated above");
    let stream = dev.cuda_stream();
    let (src_ptr, src_guard) = match &storage.slice {
        CudaStorageSlice::F32(inp) => inp.device_ptr(&stream),
        _ => candle_core::bail!("top1: F32 logits expected"),
    };
    let src_ptr = unsafe { (src_ptr as *const f32).add(layout.start_offset()) };
    {
        let (bv, bv_guard) = ws.block_values.device_ptr_mut(&stream);
        let (bi, bi_guard) = ws.block_indices.device_ptr_mut(&stream);
        let (pk, pk_guard) = ws.packed.device_ptr_mut(&stream);
        unsafe {
            top1_large_f32_packed_v084(
                src_ptr,
                bv as *mut f32,
                bi as *mut u32,
                pk as *mut f32,
                ncols as i32,
                CHUNK_SIZE as i32,
                nblocks as i32,
                stream.cu_stream() as i64,
            );
        }
        drop((bv_guard, bi_guard, pk_guard));
    }
    drop(src_guard);
    dev.synchronize()?;
    let packed = dev.clone_dtoh(&ws.packed)?;
    Ok([packed[0], packed[1]])
}

/// Model unload: free the top-1 buffers.
pub(crate) fn release_workspaces() {
    *TOP1_WS.lock().unwrap_or_else(|e| e.into_inner()) = None;
}
