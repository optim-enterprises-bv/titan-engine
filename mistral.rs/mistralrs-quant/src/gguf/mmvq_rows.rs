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
        cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg},
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
    let Device::Cuda(dev) = w.device() else {
        candle_core::bail!("mmvq_rows: weight must live on CUDA");
    };
    let (nrows, ncols) = w.shape().dims2()?;
    let (rows, k) = match xs.dims() {
        [b, k] => (*b, *k),
        [b, m, k] => (*b * *m, *k),
        other => candle_core::bail!("mmvq_rows: unexpected input rank {other:?}"),
    };
    if k != ncols || rows == 0 || rows > MAX_ROWS {
        candle_core::bail!("mmvq_rows: weight [{nrows}, {ncols}] vs input {:?}", xs.dims());
    }
    let fmt = fmt_index(w.dtype()).ok_or_else(|| candle_core::Error::msg(format!("mmvq_rows: unsupported {:?}", w.dtype())))?;
    let (dst, quantize): (_, unsafe extern "C" fn(*const std::ffi::c_void, *mut std::ffi::c_void, i32, i32, i32, *mut std::ffi::c_void)) =
        match xs.dtype() {
            DType::BF16 => ((0, "bf16"), ffi::launch_mmvq_gguf_quantize_q8_1_bf16),
            DType::F16 => ((1, "f16"), ffi::launch_mmvq_gguf_quantize_q8_1_f16),
            DType::F32 => ((2, "f32"), ffi::launch_mmvq_gguf_quantize_q8_1_f32),
            d => candle_core::bail!("mmvq_rows: input dtype {d:?}"),
        };
    let func = rows_func(&dev, fmt, dst, rows)?;

    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        candle_core::bail!("mmvq_rows: input must live on CUDA");
    };
    let k_padded = k.div_ceil(MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
    let blocks_per_row = k_padded / Q8_1_BLOCK_SIZE;
    let mut scratch = unsafe { dev.alloc::<u8>(rows * blocks_per_row * Q8_1_TYPE_SIZE)? };
    let (w_ptr, _w_guard) = w.device_ptr_with_guard(&stream)?;
    let out_shape = {
        let mut d = xs.dims().to_vec();
        *d.last_mut().unwrap() = nrows;
        Shape::from(d)
    };
    let (ncols_x, nrows_x, stride_col_y, stride_col_dst) = (k as i32, nrows as i32, blocks_per_row as i32, nrows as i32);
    let cfg = LaunchConfig { grid_dim: (nrows.div_ceil(2) as u32, 1, 1), block_dim: (32, 8, 1), shared_mem_bytes: 0 };

    macro_rules! run {
        ($t:ty) => {{
            let slice = xs_cuda.as_cuda_slice::<$t>()?;
            let mut out = unsafe { dev.alloc::<$t>(nrows * rows)? };
            {
                let (x_ptr, _xg) = slice_ptr_on_stream(slice, xs_layout.start_offset(), &stream);
                let (s_ptr, _sg) = slice_ptr_mut_on_stream(&mut scratch, 0, &stream);
                let (o_ptr, _og) = slice_ptr_mut_on_stream(&mut out, 0, &stream);
                unsafe {
                    quantize(x_ptr as *const std::ffi::c_void, s_ptr as *mut std::ffi::c_void, k as i32, k_padded as i32, rows as i32, stream_ptr);
                }
                let mut b = stream.launch_builder(&func);
                let (w_addr, s_addr, o_addr) = (w_ptr as u64, s_ptr as u64, o_ptr as u64);
                b.arg(&w_addr).arg(&s_addr).arg(&o_addr).arg(&ncols_x).arg(&nrows_x).arg(&stride_col_y).arg(&stride_col_dst);
                unsafe { b.launch(cfg) }.w()?;
            }
            Ok(Tensor::from((Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())), out_shape)))
        }};
    }
    match xs.dtype() {
        DType::BF16 => run!(half::bf16),
        DType::F16 => run!(half::f16),
        _ => run!(f32),
    }
}

/// JIT the rows module now (~2 s for 240 kernels) instead of inside the first verification.
pub fn warm_up(dev: &CudaDevice) -> Result<()> {
    rows_func(dev, (7, "q4_k"), (2, "f32"), 2).map(|_| ())
}
