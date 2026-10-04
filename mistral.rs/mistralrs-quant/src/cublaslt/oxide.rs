//! titan noblas: the cuBLASLt wrapper's batch matmuls (`api.rs`) on candle's cuda-oxide GEMM
//! (`candle_core::cuda::oxide_gemm`), for builds without the `cublas` feature. Same semantics as `api.rs`: `a`
//! `(batch, m, k)` (or `(batch, k, m)` with `transa == false`), `b` `(batch, n, k)`, result `(batch, n, m)` =
//! `alpha * b a^T + beta * out + bias` (bias of length m, added per row of the result); with `out` given the result
//! is written into a copy of `out` (C), as api.rs does. The activation epilogue and the F8 variant are not ported (no
//! roster model uses them): they fail loudly.
use candle_core::cuda::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle_core::cuda::oxide_gemm::{self, Dt, Problem};
use candle_core::backend::BackendStorage;
use candle_core::cuda::CudaDType;
use candle_core::{CpuStorage, DType, Device, Layout, Result, Shape, Storage, Tensor};
use half::{bf16, f16};
use std::ffi::c_int;

/// cuBLASLt's epilogue activations (accepted by the API, refused at run time).
#[derive(Debug, Clone, Copy)]
pub enum Activation {
    Relu,
    Gelu,
}

/// Stand-in for the cuBLASLt handle: the oxide GEMM needs none.
#[derive(Debug, Clone)]
pub struct CublasLt;

impl CublasLt {
    pub fn new(device: &Device) -> Result<Self> {
        match device {
            Device::Cuda(_) => Ok(Self),
            _ => candle_core::bail!("`device` must be a `cuda` device"),
        }
    }
}

struct OxBatchMatmul {
    c: Option<Tensor>,
    alpha: Option<f32>,
    beta: Option<f32>,
    transa: bool,
}

impl OxBatchMatmul {
    fn fwd<T: CudaDType + candle_core::cuda::cudarc::driver::DeviceRepr>(
        &self,
        dt: Dt,
        a: &candle_core::CudaStorage,
        a_l: &Layout,
        b: &candle_core::CudaStorage,
        b_l: &Layout,
        bias: Option<(&candle_core::CudaStorage, &Layout)>,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        let dev = a.device();
        let (batch, m, k) = if self.transa {
            a_l.shape().dims3()?
        } else {
            let (batch, k, m) = a_l.shape().dims3()?;
            (batch, m, k)
        };
        let (b_0, n, b_2) = b_l.shape().dims3()?;
        if b_2 != k {
            candle_core::bail!("This layer only supports TN layout");
        }
        if b_0 != batch {
            candle_core::bail!("`b` must have the same batch size as `a`")
        }
        let out_shape = Shape::from((batch, n, m));
        let stream = dev.cuda_stream();
        let es = dt.size() as u64;
        let (ap, _ga) = a.as_cuda_slice::<T>()?.device_ptr(&stream);
        let (bp, _gb) = b.as_cuda_slice::<T>()?.device_ptr(&stream);
        let (bias_ptr, bias_s1, _gbias) = match bias {
            Some((bs, bl)) => {
                if bl.shape().dims1()? != m {
                    candle_core::bail!("Bias does not have the correct shape");
                }
                let (p, g) = bs.as_cuda_slice::<T>()?.device_ptr(&stream);
                (p + bl.start_offset() as u64 * es, bl.stride()[0] as i64, Some(g))
            }
            None => (0, 0, None),
        };
        // the result: a copy of `out` (api.rs clones it too; D = C in place on the copy) or a new buffer
        let (mut out, sd) = if let Some(c) = &self.c {
            let (c, c_l) = c.storage_and_layout();
            let c = match &*c {
                Storage::Cuda(storage) => storage.as_cuda_slice::<T>()?,
                _ => candle_core::bail!("`c` must be a cuda tensor"),
            };
            match c_l.contiguous_offsets() {
                Some((o1, o2)) => {
                    if o1 != 0 {
                        candle_core::bail!("`c` start offset must be 0");
                    }
                    if o2 != out_shape.elem_count() {
                        candle_core::bail!("`c` end offset must be {}", out_shape.elem_count())
                    }
                }
                None => candle_core::bail!("`c` has to be contiguous"),
            };
            if c_l.shape().dims3()? != (batch, n, m) {
                candle_core::bail!("`c` does not have the correct shape");
            }
            (c.clone(), c_l.stride()[0])
        } else {
            (unsafe { dev.alloc::<T>(out_shape.elem_count())? }, n * m)
        };
        let (dp, _gd) = out.device_ptr_mut(&stream);
        let (as_, bs_) = (a_l.stride(), b_l.stride());
        // kernel problem: rows i over b's n, columns j over a's m: D(i, j) = sum_k b(i, k) a^T(k, j)
        let (b_s0, b_s1) = if self.transa { (as_[2] as i64, as_[1] as i64) } else { (as_[1] as i64, as_[2] as i64) };
        let beta = if self.c.is_some() { self.beta.unwrap_or(0.0) } else { 0.0 };
        let p = Problem {
            dt,
            batch: batch as i64,
            m: n as i64,
            n: m as i64,
            k: k as i64,
            a_s0: bs_[1] as i64,
            a_s1: bs_[2] as i64,
            b_s0,
            b_s1,
            d_s0: m as i64,
            d_s1: 1,
            c_s0: m as i64,
            c_s1: 1,
            bias_s0: 0,
            bias_s1,
            sa: bs_[0] as i64,
            sb: as_[0] as i64,
            sd: sd as i64,
            sc: sd as i64,
            alpha: self.alpha.unwrap_or(1.0),
            beta,
            has_c: beta != 0.0,
            has_bias: bias.is_some(),
            a_addr: bp + b_l.start_offset() as u64 * es,
            b_addr: ap + a_l.start_offset() as u64 * es,
            d_addr: dp,
        };
        oxide_gemm::gemm(dev, &p, if p.has_c { dp } else { 0 }, bias_ptr)?;
        drop((_ga, _gb, _gbias, _gd));
        Ok((candle_core::CudaStorage::wrap_cuda_slice(out, dev.clone()), out_shape))
    }

    fn dispatch(
        &self,
        a: &candle_core::CudaStorage,
        a_l: &Layout,
        b: &candle_core::CudaStorage,
        b_l: &Layout,
        bias: Option<(&candle_core::CudaStorage, &Layout)>,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        match a.dtype() {
            DType::F16 => self.fwd::<f16>(Dt::F16, a, a_l, b, b_l, bias),
            DType::BF16 => self.fwd::<bf16>(Dt::Bf16, a, a_l, b, b_l, bias),
            DType::F32 => self.fwd::<f32>(Dt::F32, a, a_l, b, b_l, bias),
            dt => candle_core::bail!("oxide batch matmul is only supported for f16/bf16/f32 ({dt:?})"),
        }
    }
}

impl candle_core::CustomOp2 for OxBatchMatmul {
    fn name(&self) -> &'static str {
        "oxide-batch-matmul"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("no cpu support for oxide-batch-matmul")
    }

    fn cuda_fwd(
        &self,
        a: &candle_core::CudaStorage,
        a_l: &Layout,
        b: &candle_core::CudaStorage,
        b_l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        self.dispatch(a, a_l, b, b_l, None)
    }
}

impl candle_core::CustomOp3 for OxBatchMatmul {
    fn name(&self) -> &'static str {
        "oxide-batch-matmul-add"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("no cpu support for oxide-batch-matmul-add")
    }

    fn cuda_fwd(
        &self,
        a: &candle_core::CudaStorage,
        a_l: &Layout,
        b: &candle_core::CudaStorage,
        b_l: &Layout,
        bias: &candle_core::CudaStorage,
        bias_l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        self.dispatch(a, a_l, b, b_l, Some((bias, bias_l)))
    }
}

/// `api::fused_batch_matmul` on the oxide GEMM (an activation fails: not ported).
#[allow(clippy::too_many_arguments)]
pub fn fused_batch_matmul(
    a: &Tensor,
    b: &Tensor,
    out: Option<&Tensor>,
    alpha: Option<f32>,
    beta: Option<f32>,
    bias: Option<&Tensor>,
    act: Option<Activation>,
    _cublaslt: CublasLt,
) -> Result<Tensor> {
    if let Some(act) = act {
        candle_core::bail!("oxide batch matmul: the {act:?} epilogue is not ported (nvcc-only, cuBLASLt)");
    }
    let op = OxBatchMatmul { c: out.cloned(), alpha, beta, transa: true };
    if let Some(bias) = bias {
        a.apply_op3(b, bias, op)
    } else {
        a.apply_op2(b, op)
    }
}

/// `api::fused_batch_matmul_nn`: `a` given as `(batch, k, m)`.
pub fn fused_batch_matmul_nn(a: &Tensor, b: &Tensor, alpha: Option<f32>, _cublaslt: CublasLt) -> Result<Tensor> {
    a.apply_op2(b, OxBatchMatmul { c: None, alpha, beta: None, transa: false })
}

/// `api::fused_batch_matmul_heur`: cuBLASLt's algorithm hint has no oxide meaning; the plain batch matmul.
pub fn fused_batch_matmul_heur(
    a: &Tensor,
    b: &Tensor,
    alpha: Option<f32>,
    _heuristic_batch: (c_int, i64),
    _cublaslt: CublasLt,
) -> Result<Tensor> {
    a.apply_op2(b, OxBatchMatmul { c: None, alpha, beta: None, transa: true })
}

/// `api::fused_batch_matmul_f8`: not ported (FP8 safetensors models only).
#[allow(clippy::too_many_arguments)]
pub fn fused_batch_matmul_f8(
    _a: &Tensor,
    _b: &Tensor,
    _dequant_a_scale: &Tensor,
    _dequant_b_scale: &Tensor,
    _quantize_scale: &Tensor,
    _out: Option<&Tensor>,
    _alpha: Option<f32>,
    _beta: Option<f32>,
    _bias: Option<&Tensor>,
    _act: Option<Activation>,
    _cublaslt: CublasLt,
) -> Result<Tensor> {
    candle_core::bail!("oxide build: the FP8 cuBLASLt batch matmul is not ported (nvcc-only)")
}
