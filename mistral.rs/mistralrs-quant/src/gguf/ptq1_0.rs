//! PrismML PTQ1_0 (GGML type 143) with the prism Hadamard, on CUDA (titan).
//!
//! Ternary Bonsai 2 folds a normalized Sylvester-Walsh-Hadamard rotation (1024-wide blocks along the input,
//! explicit +-1 signs applied first; GGUF `prism.hadamard.*`) into every PTQ1_0 weight, so the activation of
//! each folded matmul is rotated the same way first: `y = W' (H S x)`. The token embedding stores latent rows:
//! after a lookup, `h = S (H z)`. The kernels are cuda-oxide ports of sudoingX/llama.cpp bonsai2 v1.1
//! (titan-engine/oxide-kernels/ptq1_0, bit-identical to its nvcc build), embedded as PTX:
//!
//! - decode and verify rows (1..=4 columns): `fwht + quantize_q8_1<pt>` fused into one launch writing the
//!   planar-transposed Q8_1 activations, then `mul_mat_vec_ptq1_0_pt<n>` (PrismML-Eng/llama.cpp#218), whose
//!   per-column arithmetic does not depend on the column count: a verify row equals its decode step bit for
//!   bit, as MTP needs; the gated FFN pair runs as one launch (`<n, 4, true, true>`, `up * silu(gate)`);
//! - prompt chunks (5+ columns): the FWHT to f32, then `quantize_mmq_q8_1<D4>` + `mul_mat_q<PTQ1_0, J>`
//!   (PrismML's PTQ1_0 tile loader on llama.cpp's MMQ), through `fast_mmq::llama_fmt_plain`.
//!
//! Regenerate ptq1_0_{mmvq,mmq}_oxide.ptx with oxide-kernels/ptq1_0/export_ptx.py.

use candle_core::{Device, Result, Tensor};

/// The rotation of one folded weight's input: width (a multiple of 1024) and the explicit signs (`None`: the
/// GGUF's `identity` sign mode). Every folded weight of one input width shares one sign vector.
#[derive(Debug)]
pub struct PrismRot {
    pub width: usize,
    /// F32 (width), on the model device.
    pub signs: Option<Tensor>,
}

impl PrismRot {
    pub fn new(width: usize, signs: Option<&[f32]>, device: &Device) -> Result<Self> {
        if width % 1024 != 0 {
            candle_core::bail!("prism hadamard: width {width} not a multiple of 1024");
        }
        let signs = match signs {
            Some(s) => {
                if s.len() != width {
                    candle_core::bail!("prism hadamard: {} signs for width {width}", s.len());
                }
                Some(Tensor::from_slice(s, width, device)?)
            }
            None => None,
        };
        Ok(Self { width, signs })
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::*;

#[cfg(feature = "cuda")]
mod cuda_impl {
    use std::sync::{Mutex, OnceLock};

    use candle_core::cuda::cudarc::driver::{DevicePtrMut, LaunchConfig, PushKernelArg};
    use candle_core::cuda::WrapErr;
    use candle_core::{
        quantized::{GgmlDType, QTensor},
        CudaDevice, CudaStorage, DType, Device, Result, Shape, Storage, Tensor,
    };

    use super::PrismRot;
    use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

    const MMVQ_PTX: &str = include_str!("ptq1_0_mmvq_oxide.ptx");
    const MMVQ_MODULE: &str = "titan_ptq1_0_mmvq";
    /// Function-cache slots (fast_mmvq::q1_0_func) of this module: 2000 + index.
    const SLOT: usize = 2000;
    /// Columns the PT mat-vec takes (llama.cpp PTQ1_0_PT_MAX_COLS); 5 and more go to the MMQ.
    pub const PT_MAX_COLS: usize = 4;

    static PT_WS_ROT: OnceLock<crate::gguf::fast_mmvq::Q1WsMap> = OnceLock::new();
    static PT_WS_PLAIN: OnceLock<crate::gguf::fast_mmvq::Q1WsMap> = OnceLock::new();

    /// The last rotated prompt activation (input tensor id, width) -> f32 rotation, so Q/K/V/gate and
    /// gate/up of one prompt chunk rotate once.
    type RotMemo = Mutex<Option<(candle_core::TensorId, usize, Tensor)>>;
    static ROT_MEMO: OnceLock<RotMemo> = OnceLock::new();

    pub(crate) fn release_workspaces() {
        crate::gguf::free_leaked(&PT_WS_ROT);
        crate::gguf::free_leaked(&PT_WS_PLAIN);
        if let Some(m) = ROT_MEMO.get() {
            *m.lock().unwrap_or_else(|e| e.into_inner()) = None;
        }
    }

    fn func(dev: &CudaDevice, idx: usize, name: impl FnOnce() -> String) -> Result<candle_core::cuda::cudarc::driver::CudaFunction> {
        crate::gguf::fast_mmvq::q1_0_func(dev, SLOT + idx, MMVQ_MODULE, MMVQ_PTX, name)
    }

    fn cuda_dev(t: &Tensor) -> Result<CudaDevice> {
        match t.device() {
            Device::Cuda(d) => Ok(d.clone()),
            _ => candle_core::bail!("ptq1_0: tensor must live on CUDA"),
        }
    }

    /// Device pointer of a contiguous tensor's first element (valid while the tensor lives; the stream orders use).
    fn tptr(t: &Tensor, stream: &candle_core::cuda::cudarc::driver::CudaStream) -> Result<u64> {
        let (st, l) = t.storage_and_layout();
        let candle_core::Storage::Cuda(c) = &*st else {
            candle_core::bail!("ptq1_0: tensor must live on CUDA");
        };
        let o = l.start_offset();
        Ok(match t.dtype() {
            DType::F32 => slice_ptr_on_stream(c.as_cuda_slice::<f32>()?, o, stream).0,
            DType::BF16 => slice_ptr_on_stream(c.as_cuda_slice::<half::bf16>()?, o, stream).0,
            DType::F16 => slice_ptr_on_stream(c.as_cuda_slice::<half::f16>()?, o, stream).0,
            other => candle_core::bail!("ptq1_0: unsupported dtype {other:?}"),
        })
    }

    /// ptq1_0_pt_rows_per_cta (mmvq-ptq1_0.cuh): rows per 128-thread CTA filling whole iterations.
    pub fn rows_per_cta(bpr: i32, nc: i32) -> i32 {
        const ROWS: i32 = 4;
        let mut rmax = 4096 / (nc * bpr);
        rmax = if rmax < ROWS { ROWS } else if rmax > 16 { 16 } else { rmax };
        rmax -= rmax % ROWS;
        let (mut best, mut best_util) = (ROWS, 0.0f64);
        let mut r = ROWS;
        while r <= rmax {
            let items = (r / ROWS) * bpr;
            let iters = (items + 127) / 128;
            let util = items as f64 / (iters * 128) as f64;
            if util > best_util + 1e-9 {
                best_util = util;
                best = r;
            }
            if util > 0.999 {
                break;
            }
            r += ROWS;
        }
        best
    }

    /// `x` (.., k), contiguous F32 / BF16 -> (rows, k).
    fn rows_k(x: &Tensor) -> Result<(usize, usize)> {
        let k = x.dim(candle_core::D::Minus1)?;
        Ok((x.elem_count() / k.max(1), k))
    }

    /// FWHT of `x` (.., width) into a new F32 tensor of the same shape. `mode` 0: none, 1: signs first,
    /// 2: signs after (the latent-embedding inverse).
    pub fn fwht(x: &Tensor, rot: &PrismRot, mode: u32) -> Result<Tensor> {
        let dev = cuda_dev(x)?;
        let x = x.contiguous()?;
        let (rows, k) = rows_k(&x)?;
        if k != rot.width {
            candle_core::bail!("ptq1_0 fwht: input width {k}, rotation width {}", rot.width);
        }
        let ts = match x.dtype() {
            DType::F32 => "f32",
            DType::BF16 => "bf16",
            other => candle_core::bail!("ptq1_0 fwht: input dtype {other:?}"),
        };
        let mode = if rot.signs.is_none() { 0 } else { mode };
        let idx = 10 + mode as usize * 2 + (ts == "bf16") as usize;
        let f = func(&dev, idx, || format!("ptq1_0_fwht_{ts}_m{mode}"))?;
        let stream = dev.cuda_stream();
        let n_blk = (k / 1024) as i32;
        let n_rows = (rows * k / 1024) as i64;
        let mut out = unsafe { dev.alloc::<f32>(rows * k)? };
        {
            let src = tptr(&x, &stream)?;
            let signs = match &rot.signs {
                Some(s) => tptr(s, &stream)?,
                None => 0u64,
            };
            let (dst, _g) = slice_ptr_mut_on_stream(&mut out, 0, &stream);
            let scale = 1.0f32 / 32.0;
            let mut b = stream.launch_builder(&f);
            b.arg(&src).arg(&dst).arg(&n_rows).arg(&scale).arg(&signs).arg(&n_blk);
            let cfg = LaunchConfig { grid_dim: (n_rows as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            unsafe { b.launch(cfg) }.w()?;
        }
        Ok(Tensor::from((Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev)), x.shape().clone())))
    }

    /// The latent embedding rows `z` (n, width) -> `S (H z)` in F32.
    pub fn inverse_rows(z: &Tensor, rot: &PrismRot) -> Result<Tensor> {
        fwht(z, rot, 2)
    }

    /// The input rotated for a prompt chunk (memoized on the input tensor and width).
    fn rotated(x: &Tensor, rot: &PrismRot) -> Result<Tensor> {
        let memo = ROT_MEMO.get_or_init(|| Mutex::new(None));
        if let Some((id, w, t)) = memo.lock().unwrap().as_ref() {
            if *id == x.id() && *w == rot.width {
                return Ok(t.clone());
            }
        }
        let t = fwht(x, rot, 1)?;
        *memo.lock().unwrap() = Some((x.id(), rot.width, t.clone()));
        Ok(t)
    }

    /// Quantize `x` (rows <= 4, k) into the PT activation workspace (rotated first when `rot` is given),
    /// unless the workspace already holds exactly this input; then run `launch(ws_ptr)`.
    fn with_pt_activations<T>(
        dev: &CudaDevice,
        x: &Tensor,
        rot: Option<&PrismRot>,
        launch: impl FnOnce(u64) -> Result<T>,
    ) -> Result<T> {
        let (rows, k) = rows_k(x)?;
        let kp = k.div_ceil(512) * 512;
        let bytes = rows * kp / 32 * 36;
        let ws = if rot.is_some() { &PT_WS_ROT } else { &PT_WS_PLAIN };
        let mut w = crate::gguf::fast_mmvq::q1_0_workspace(ws, dev, bytes)?;
        let tag: crate::gguf::fast_mmvq::Q1Tag = (x.id(), x.dtype(), x.layout().start_offset(), x.dims().to_vec());
        let quantize = w.tag.as_ref() != Some(&tag);
        w.tag = None;
        let stream = dev.cuda_stream();
        let out;
        {
            let (y_ptr, _yg) = w.slice.device_ptr_mut(&stream);
            if quantize {
                let bf = x.dtype() == DType::BF16;
                let src = tptr(x, &stream)?;
                match rot {
                    Some(rot) if k % 512 == 0 && k == rot.width => {
                        let mode = rot.signs.is_some() as u32;
                        let f = func(dev, 20 + mode as usize * 2 + bf as usize, || {
                            format!("ptq1_0_fwht_q8pt_{}_m{mode}", if bf { "bf16" } else { "f32" })
                        })?;
                        let signs = match &rot.signs {
                            Some(s) => tptr(s, &stream)?,
                            None => 0u64,
                        };
                        let n_blk = (k / 1024) as i32;
                        let n_rows = (rows * k / 1024) as i64;
                        let scale = 1.0f32 / 32.0;
                        let mut b = stream.launch_builder(&f);
                        b.arg(&src).arg(&y_ptr).arg(&n_rows).arg(&scale).arg(&signs).arg(&n_blk);
                        let cfg = LaunchConfig { grid_dim: (n_rows as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                        unsafe { b.launch(cfg) }.w()?;
                    }
                    Some(rot) => {
                        // width not a multiple of 512: rotate, then quantize with padding
                        let xr = fwht(x, rot, 1)?;
                        quantize_pt(dev, &xr, y_ptr, rows, k, kp)?;
                    }
                    None => quantize_pt(dev, x, y_ptr, rows, k, kp)?,
                }
            }
            out = launch(y_ptr)?;
        }
        w.tag = Some(tag);
        Ok(out)
    }

    /// `quantize_q8_1<pt>` of `x` (rows, k) into the PT columns at `y_ptr` (column stride kp * 9 / 8).
    fn quantize_pt(dev: &CudaDevice, x: &Tensor, y_ptr: u64, rows: usize, k: usize, kp: usize) -> Result<()> {
        let bf = x.dtype() == DType::BF16;
        let f = func(dev, 30 + bf as usize, || format!("ptq1_0_quantize_pt_{}", if bf { "bf16" } else { "f32" }))?;
        let stream = dev.cuda_stream();
        let src = tptr(x, &stream)?;
        let one = crate::gguf::fast_mmvq::q1_0_fastdiv(1);
        let (ne00, s01, s02, ne0) = (k as i64, k as i64, (k * rows) as i64, kp as i64);
        let ne1 = rows as u32;
        let mut b = stream.launch_builder(&f);
        b.arg(&src).arg(&y_ptr).arg(&ne00).arg(&s01).arg(&s02).arg(&s02).arg(&ne0).arg(&ne1);
        b.arg(&one[0]).arg(&one[1]).arg(&one[2]);
        let cfg = LaunchConfig { grid_dim: ((kp as u32).div_ceil(256), ne1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe { b.launch(cfg) }.w()?;
        Ok(())
    }

    /// One `mul_mat_vec_ptq1_0_pt` launch: `up` (and `gate` for the GLU form) against the PT columns.
    #[allow(clippy::too_many_arguments)]
    fn mmvq_launch(
        dev: &CudaDevice,
        up: &QTensor,
        gate: Option<&QTensor>,
        y_ptr: u64,
        dst_ptr: u64,
        rows: usize,
        k: usize,
        nrows: usize,
        bf16_out: bool,
    ) -> Result<()> {
        let nc = rows as i32;
        let bpr = (k / 128) as i32;
        let rpc = rows_per_cta(bpr, nc);
        let fd = crate::gguf::fast_mmvq::q1_0_fastdiv(bpr as u32);
        let glu = gate.is_some();
        let smem = (nc * rpc * bpr * 4 * if glu { 2 } else { 1 }) as u32;
        let (idx, name) = if glu {
            (40 + rows, format!("ptq1_0_mmvq_pt_glu_c{rows}"))
        } else if bf16_out {
            (50 + rows, format!("ptq1_0_mmvq_pt_c{rows}_bf16"))
        } else {
            (60 + rows, format!("ptq1_0_mmvq_pt_c{rows}"))
        };
        let f = func(dev, idx, || name)?;
        if smem > 48 * 1024 {
            use candle_core::cuda::cudarc::driver::sys::CUfunction_attribute_enum;
            f.set_attribute(CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32).w()?;
        }
        let stream = dev.cuda_stream();
        let (w_ptr, _wg) = up.device_ptr_with_guard(&stream)?;
        let w_ptr = w_ptr as u64;
        let g_ptr = match gate {
            Some(g) => g.device_ptr_with_guard(&stream)?.0 as u64,
            None => 0u64,
        };
        let (z64, z32) = (0u64, 0u32);
        let glu_op = 2u32; // GGML_GLU_OP_SWIGLU
        let (ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst) =
            (k as i32, nrows as i32, bpr, (k.div_ceil(512) * 512 / 32) as i32, nrows as i32);
        let mut b = stream.launch_builder(&f);
        b.arg(&w_ptr).arg(&y_ptr);
        b.arg(&z64).arg(&g_ptr).arg(&z64).arg(&z64).arg(&z64).arg(&glu_op).arg(&z32);
        b.arg(&dst_ptr).arg(&ncols_x).arg(&nrows_x).arg(&stride_row_x).arg(&stride_col_y).arg(&stride_col_dst).arg(&rpc);
        b.arg(&fd[0]).arg(&fd[1]).arg(&fd[2]);
        let cfg = LaunchConfig {
            grid_dim: (((nrows as i32 + rpc - 1) / rpc) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: smem,
        };
        unsafe { b.launch(cfg) }.w()?;
        Ok(())
    }

    fn out_shape(x: &Tensor, n: usize) -> Shape {
        let mut d = x.dims().to_vec();
        *d.last_mut().unwrap() = n;
        Shape::from(d)
    }

    /// `w @ x` for a PTQ1_0 weight (`rot`: its folded Hadamard). Rows 1..=4: the PT mat-vec (bit-identical per
    /// row for any row count); more: FWHT + MMQ. Output in `x`'s dtype (F32 or BF16).
    pub fn forward(w: &QTensor, x: &Tensor, rot: Option<&PrismRot>) -> Result<Tensor> {
        if w.dtype() != GgmlDType::PTQ1_0 {
            candle_core::bail!("ptq1_0::forward on {:?}", w.dtype());
        }
        let dev = cuda_dev(x)?;
        let (nrows, k) = w.shape().dims2()?;
        let x = x.contiguous()?;
        let (rows, kx) = rows_k(&x)?;
        if kx != k || k % 128 != 0 {
            candle_core::bail!("ptq1_0: input width {kx}, weight ({nrows}, {k})");
        }
        if !matches!(x.dtype(), DType::F32 | DType::BF16) {
            candle_core::bail!("ptq1_0: input dtype {:?}", x.dtype());
        }
        if rows == 0 {
            return Tensor::zeros(out_shape(&x, nrows), x.dtype(), x.device());
        }
        if rows > PT_MAX_COLS {
            let xr = match rot {
                Some(r) => rotated(&x, r)?,
                None => x.clone(),
            };
            let Some(out) = crate::gguf::fast_mmq::llama_fmt_plain(w, &xr)? else {
                candle_core::bail!("ptq1_0: no MMQ path for ({nrows}, {k}) x {rows}");
            };
            return out.reshape(out_shape(&x, nrows))?.to_dtype(x.dtype());
        }
        let bf16_out = x.dtype() == DType::BF16;
        let stream = dev.cuda_stream();
        let out_storage = if bf16_out {
            let mut out = unsafe { dev.alloc::<half::bf16>(rows * nrows)? };
            with_pt_activations(&dev, &x, rot, |y| {
                let (d, _g) = slice_ptr_mut_on_stream(&mut out, 0, &stream);
                mmvq_launch(&dev, w, None, y, d, rows, k, nrows, true)
            })?;
            CudaStorage::wrap_cuda_slice(out, dev.clone())
        } else {
            let mut out = unsafe { dev.alloc::<f32>(rows * nrows)? };
            with_pt_activations(&dev, &x, rot, |y| {
                let (d, _g) = slice_ptr_mut_on_stream(&mut out, 0, &stream);
                mmvq_launch(&dev, w, None, y, d, rows, k, nrows, false)
            })?;
            CudaStorage::wrap_cuda_slice(out, dev.clone())
        };
        Ok(Tensor::from((Storage::Cuda(out_storage), out_shape(&x, nrows))))
    }

    /// `up(x) * silu(gate(x))` in F32 for 1..=4 rows in one launch (both weights PTQ1_0 of one shape,
    /// sharing the input rotation); `None` for more rows or other weights.
    pub fn glu_forward(gate: &QTensor, up: &QTensor, x: &Tensor, rot: Option<&PrismRot>) -> Result<Option<Tensor>> {
        if gate.dtype() != GgmlDType::PTQ1_0 || up.dtype() != GgmlDType::PTQ1_0 || gate.shape() != up.shape() {
            return Ok(None);
        }
        let (rows, k) = rows_k(x)?;
        let (nrows, kw) = up.shape().dims2()?;
        if rows == 0 || rows > PT_MAX_COLS || kw != k || k % 128 != 0 || !matches!(x.dtype(), DType::F32 | DType::BF16) {
            return Ok(None);
        }
        let dev = cuda_dev(x)?;
        let x = x.contiguous()?;
        let stream = dev.cuda_stream();
        let mut out = unsafe { dev.alloc::<f32>(rows * nrows)? };
        with_pt_activations(&dev, &x, rot, |y| {
            let (d, _g) = slice_ptr_mut_on_stream(&mut out, 0, &stream);
            mmvq_launch(&dev, up, Some(gate), y, d, rows, k, nrows, false)
        })?;
        Ok(Some(Tensor::from((Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())), out_shape(&x, nrows)))))
    }
}
