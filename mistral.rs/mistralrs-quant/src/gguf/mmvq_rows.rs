//! Row-exact batched GGUF matvec (titan-engine M6).
//!
//! `plain_rows(w, xs)` computes `w @ xs^T` for 1..=8 activation rows with one launch whose every
//! row is bit-identical to `fast_mmvq::plain` on that row alone (the batch-1 MMVQ reduction:
//! 8 warps x 2 weight rows, same k split, same shared-memory handoff and butterfly per column),
//! while the weight rows are read once for all activation rows. The batch-2..8 MMVQ kernels split
//! the dot products differently and round differently; speculative verification must reproduce
//! the decode step exactly, hence this entry.
//!
//! The kernels are cuda-oxide (`titan-engine/oxide-kernels/mmvq-rows`, gated bit-identical against
//! nvcc's batch-1 `mmvq_gguf_*_plain_cuda1`), embedded as PTX and launched from Rust, so the nvcc
//! and the nvcc-free `oxide` builds share them. The activations are quantized by the same
//! `launch_mmvq_gguf_quantize_q8_1_*` launcher `fast_mmvq::plain` uses in either build.
//! Regenerate mmvq_rows_oxide.ptx with oxide-kernels/mmvq-rows/export_ptx.sh.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::{
    cuda::{
        cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg},
        CudaDevice, DeviceId, WrapErr,
    },
    quantized::{GgmlDType, QTensor},
    CudaStorage, DType, Device, Result, Shape, Storage, Tensor,
};

use super::ffi;
use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

const Q8_1_BLOCK_SIZE: usize = 32;
const Q8_1_TYPE_SIZE: usize = 36;
const MATRIX_ROW_PADDING: usize = 512;
pub const MAX_ROWS: usize = 8;

const PTX: &str = include_str!("mmvq_rows_oxide.ptx");
const MODULE: &str = "titan_mmvq_rows";

/// Kernel name suffix and index of the formats the rows kernels cover.
fn fmt_index(dtype: GgmlDType) -> Option<(usize, &'static str)> {
    Some(match dtype {
        GgmlDType::Q4_0 => (0, "q4_0"),
        GgmlDType::Q4_1 => (1, "q4_1"),
        GgmlDType::Q5_0 => (2, "q5_0"),
        GgmlDType::Q5_1 => (3, "q5_1"),
        GgmlDType::Q8_0 => (4, "q8_0"),
        GgmlDType::Q2K => (5, "q2_k"),
        GgmlDType::Q3K => (6, "q3_k"),
        GgmlDType::Q4K => (7, "q4_k"),
        GgmlDType::Q5K => (8, "q5_k"),
        GgmlDType::Q6K => (9, "q6_k"),
        _ => return None,
    })
}

type FnCache = Mutex<HashMap<(DeviceId, usize), CudaFunction>>;
static FNS: OnceLock<FnCache> = OnceLock::new();

/// `mmvq_gguf_<fmt>_<dst>_rows<n>`, resolved once per device (a `get_or_load_custom_func` per
/// launch costs a module-map lock, a CString and a cuModuleGetFunction).
/// `TITAN_Q3K_WARPS=4|16`: use the alternate warp geometry for the single-row Q3_K kernel.
fn q3k_warps() -> usize {
    crate::titan_cfg::var("TITAN_Q3K_WARPS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n == 4 || *n == 16)
        .unwrap_or(0)
}

fn rows_func_w(dev: &CudaDevice, fmt: (usize, &str), dst: (usize, &str), n: usize, w: usize) -> Result<CudaFunction> {
    let name = format!("mmvq_gguf_{}_{}_rows{n}w{w}", fmt.1, dst.1);
    Ok(dev.get_or_load_custom_func(&name, MODULE, PTX)?.into_cuda_function())
}

fn rows_func(dev: &CudaDevice, fmt: (usize, &str), dst: (usize, &str), n: usize) -> Result<CudaFunction> {
    let key = (dev.id(), (fmt.0 * 3 + dst.0) * MAX_ROWS + n - 1);
    let mut map = FNS.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap();
    if let Some(f) = map.get(&key) {
        return Ok(f.clone());
    }
    let name = format!("mmvq_gguf_{}_{}_rows{n}", fmt.1, dst.1);
    let f = dev.get_or_load_custom_func(&name, MODULE, PTX)?.into_cuda_function();
    map.insert(key, f.clone());
    Ok(f)
}

/// Whether `plain_rows` handles this weight / activation pair.
pub fn supports(w: &QTensor, xs: &Tensor) -> bool {
    w.device().is_cuda() && fmt_index(w.dtype()).is_some() && matches!(xs.dtype(), DType::BF16 | DType::F16 | DType::F32)
}

/// `w @ xs^T` for `xs` `[b, k]` or `[b, m, k]` with `b * m` rows in `1..=8`, each row exactly as
/// `fast_mmvq::plain` computes it alone. Output dtype = input dtype.
pub fn plain_rows(w: &QTensor, xs: &Tensor) -> Result<Tensor> {
    plain_rows_q8(w, &quantize_rows(xs)?)
}

/// Activation rows quantized to Q8_1 once, for every weight that multiplies them (`plain_rows_q8`).
pub struct Q8Rows {
    scratch: CudaSlice<u8>,
    dims: Vec<usize>,
    rows: usize,
    k: usize,
    dtype: DType,
    dev: CudaDevice,
}

/// The Q8_1 quantization `plain_rows` runs before its matvec.
pub fn quantize_rows(xs: &Tensor) -> Result<Q8Rows> {
    let Device::Cuda(dev) = xs.device() else {
        candle_core::bail!("mmvq_rows: input must live on CUDA");
    };
    let (rows, k) = match xs.dims() {
        [b, k] => (*b, *k),
        [b, m, k] => (*b * *m, *k),
        other => candle_core::bail!("mmvq_rows: unexpected input rank {other:?}"),
    };
    if rows == 0 || rows > MAX_ROWS {
        candle_core::bail!("mmvq_rows: {rows} input rows");
    }
    let quantize: unsafe extern "C" fn(*const std::ffi::c_void, *mut std::ffi::c_void, i32, i32, i32, *mut std::ffi::c_void) =
        match xs.dtype() {
            DType::BF16 => ffi::launch_mmvq_gguf_quantize_q8_1_bf16,
            DType::F16 => ffi::launch_mmvq_gguf_quantize_q8_1_f16,
            DType::F32 => ffi::launch_mmvq_gguf_quantize_q8_1_f32,
            d => candle_core::bail!("mmvq_rows: input dtype {d:?}"),
        };
    let stream = dev.cuda_stream();
    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        candle_core::bail!("mmvq_rows: input must live on CUDA");
    };
    let k_padded = k.div_ceil(MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
    let mut scratch = unsafe { dev.alloc::<u8>(rows * (k_padded / Q8_1_BLOCK_SIZE) * Q8_1_TYPE_SIZE)? };
    macro_rules! quantize {
        ($t:ty) => {{
            let slice = xs_cuda.as_cuda_slice::<$t>()?;
            let (x_ptr, _xg) = slice_ptr_on_stream(slice, xs_layout.start_offset(), &stream);
            let (s_ptr, _sg) = slice_ptr_mut_on_stream(&mut scratch, 0, &stream);
            unsafe {
                quantize(
                    x_ptr as *const std::ffi::c_void,
                    s_ptr as *mut std::ffi::c_void,
                    k as i32,
                    k_padded as i32,
                    rows as i32,
                    stream.cu_stream() as *mut std::ffi::c_void,
                );
            }
        }};
    }
    match xs.dtype() {
        DType::BF16 => quantize!(half::bf16),
        DType::F16 => quantize!(half::f16),
        _ => quantize!(f32),
    }
    Ok(Q8Rows { scratch, dims: xs.dims().to_vec(), rows, k, dtype: xs.dtype(), dev: dev.clone() })
}

/// `plain_rows` of `w` on rows `quantize_rows` already quantized. Output dtype = the rows' dtype.
pub fn plain_rows_q8(w: &QTensor, x: &Q8Rows) -> Result<Tensor> {
    let dev = &x.dev;
    let (nrows, ncols) = w.shape().dims2()?;
    let (rows, k) = (x.rows, x.k);
    if k != ncols {
        candle_core::bail!("mmvq_rows: weight [{nrows}, {ncols}] vs input {:?}", x.dims);
    }
    let fmt = fmt_index(w.dtype()).ok_or_else(|| candle_core::Error::msg(format!("mmvq_rows: unsupported {:?}", w.dtype())))?;
    let dst = match x.dtype {
        DType::BF16 => (0, "bf16"),
        DType::F16 => (1, "f16"),
        _ => (2, "f32"),
    };
    // Issue-bound single-row Q3_K: allow a different warp geometry (same dot, same order per warp).
    let nw = if rows == 1 && w.dtype() == GgmlDType::Q3K { q3k_warps() } else { 0 };
    let func = if nw > 0 { rows_func_w(dev, fmt, dst, rows, nw)? } else { rows_func(dev, fmt, dst, rows)? };

    let stream = dev.cuda_stream();
    let blocks_per_row = k.div_ceil(MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING / Q8_1_BLOCK_SIZE;
    let (w_ptr, _w_guard) = w.device_ptr_with_guard(&stream)?;
    let out_shape = {
        let mut d = x.dims.clone();
        *d.last_mut().unwrap() = nrows;
        Shape::from(d)
    };
    let (ncols_x, nrows_x, stride_col_y, stride_col_dst) = (k as i32, nrows as i32, blocks_per_row as i32, nrows as i32);
    let block_warps = if nw > 0 { nw as u32 } else { 8 };
    let cfg = LaunchConfig { grid_dim: (nrows.div_ceil(2) as u32, 1, 1), block_dim: (32, block_warps, 1), shared_mem_bytes: 0 };

    macro_rules! run {
        ($t:ty) => {{
            let mut out = unsafe { dev.alloc::<$t>(nrows * rows)? };
            {
                let (s_ptr, _sg) = slice_ptr_on_stream(&x.scratch, 0, &stream);
                let (o_ptr, _og) = slice_ptr_mut_on_stream(&mut out, 0, &stream);
                let mut b = stream.launch_builder(&func);
                let (w_addr, s_addr, o_addr) = (w_ptr as u64, s_ptr as u64, o_ptr as u64);
                b.arg(&w_addr).arg(&s_addr).arg(&o_addr).arg(&ncols_x).arg(&nrows_x).arg(&stride_col_y).arg(&stride_col_dst);
                unsafe { b.launch(cfg) }.w()?;
            }
            Ok(Tensor::from((Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())), out_shape)))
        }};
    }
    match x.dtype {
        DType::BF16 => run!(half::bf16),
        DType::F16 => run!(half::f16),
        _ => run!(f32),
    }
}

/// JIT the rows module now (~2 s for 240 kernels) instead of inside the first verification.
pub fn warm_up(dev: &CudaDevice) -> Result<()> {
    rows_func(dev, (7, "q4_k"), (2, "f32"), 2).map(|_| ())
}
