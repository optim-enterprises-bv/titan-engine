//! NanoQuant factored binary linears: `W = diag(s_post) U V diag(s_pre)` with `U`, `V` in {-1, +1}, bit packed lsb
//! first along the input dimension (a set bit is -1), 32 columns per word. `y = s_post * (U (V (s_pre * x)))`.

use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};

use crate::{QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedSerde};

/// Output rows per warp and warps per block of `nq_mv`.
const ROWS_PER_WARP: usize = 4;
const WARPS: usize = 8;
const ROWS_PER_BLOCK: usize = ROWS_PER_WARP * WARPS;

#[cfg(feature = "cuda")]
const SRC: &str = r#"
#define ROWS_PER_WARP 4

// out[task, row] = s_out[e, row] * sum_c sign(bits[e, row, c]) * x[task / src_div, c] * s_in[e, c], e = ids[task] (or 0).
// Lane l takes column 32 w + l of every word w, so the x reads coalesce and each word is a broadcast load.
extern "C" __global__ void nq_mv(const unsigned* __restrict__ bits, long long mat_words, int wpr, int rows,
                                 const float* __restrict__ x, int ldx, int src_div, const unsigned* __restrict__ ids,
                                 const float* __restrict__ s_in, const float* __restrict__ s_out, float* __restrict__ out) {
    const int task = blockIdx.y;
    const unsigned e = ids ? ids[task] : 0u;
    const int cols = wpr * 32;
    const float* xr = x + (long long)(task / src_div) * ldx;
    const unsigned* m = bits + (long long)e * mat_words;
    const float* si = s_in ? s_in + (long long)e * cols : 0;
    const int lane = threadIdx.x & 31;
    const int r0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * ROWS_PER_WARP;
    if (r0 >= rows) return;
    const int nr = min(ROWS_PER_WARP, rows - r0);
    float acc[ROWS_PER_WARP] = {0.f, 0.f, 0.f, 0.f};
    for (int w = 0; w < wpr; ++w) {
        const int c = w * 32 + lane;
        float xv = xr[c];
        if (si) xv *= si[c];
        const unsigned xb = __float_as_uint(xv);
#pragma unroll
        for (int i = 0; i < ROWS_PER_WARP; ++i) {
            if (i < nr) {
                const unsigned word = m[(long long)(r0 + i) * wpr + w];
                acc[i] += __uint_as_float(xb ^ (((word >> lane) & 1u) << 31));
            }
        }
    }
#pragma unroll
    for (int i = 0; i < ROWS_PER_WARP; ++i) {
        float v = acc[i];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0 && i < nr) {
            const int r = r0 + i;
            out[(long long)task * rows + r] = s_out ? v * s_out[(long long)e * rows + r] : v;
        }
    }
}
"#;

#[cfg(feature = "cuda")]
fn ptx() -> Result<&'static str> {
    use std::sync::OnceLock;
    static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        candle_core::cuda::cudarc::nvrtc::safe::compile_ptx_with_opts(SRC, candle_core::cuda::cudarc::nvrtc::safe::CompileOptions::default())
            .map(|p| p.to_src())
            .map_err(|e| format!("{e:?}"))
    })
    .as_ref()
    .map(String::as_str)
    .map_err(|e| candle_core::Error::msg(format!("nq NVRTC compile failed: {e}")))
}

/// One factored projection, or a stack of `experts` of them (the scales stack the same way).
#[derive(Debug, Clone)]
pub struct NqMat {
    /// `[experts * out * rank/32]` U32 words
    u: Tensor,
    /// `[experts * rank * in/32]` U32 words
    v: Tensor,
    /// `[experts, in]` F32
    s_pre: Tensor,
    /// `[experts, out]` F32
    s_post: Tensor,
    pub out_dim: usize,
    pub in_dim: usize,
    pub rank: usize,
    pub experts: usize,
}

impl NqMat {
    /// From the raw GGUF bytes of `nq_u` / `nq_v` (u32 words) and the F32 scales; `experts` 1 for a plain linear.
    pub fn new(u_bytes: &[u8], v_bytes: &[u8], s_pre: Tensor, s_post: Tensor, experts: usize, device: &Device) -> Result<Self> {
        let words = |b: &[u8]| -> Vec<u32> { b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() };
        let (u, v) = (words(u_bytes), words(v_bytes));
        let in_dim = s_pre.elem_count() / experts;
        let out_dim = s_post.elem_count() / experts;
        if in_dim % 32 != 0 || out_dim == 0 || u.len() % (experts * out_dim) != 0 {
            candle_core::bail!("nq: U of {} words for {experts} x [{out_dim}, {in_dim}]", u.len());
        }
        let rank = u.len() / (experts * out_dim) * 32;
        if v.len() != experts * rank * in_dim / 32 {
            candle_core::bail!("nq: V of {} words, want {experts} x {rank} x {in_dim} bits", v.len());
        }
        Ok(Self {
            u: Tensor::from_vec(u, u_bytes.len() / 4, device)?,
            v: Tensor::from_vec(v, v_bytes.len() / 4, device)?,
            s_pre: s_pre.to_dtype(DType::F32)?.reshape((experts, in_dim))?.to_device(device)?,
            s_post: s_post.to_dtype(DType::F32)?.reshape((experts, out_dim))?.to_device(device)?,
            out_dim,
            in_dim,
            rank,
            experts,
        })
    }

    /// `x` `[rows, in]` f32 (row `t / src_div` feeds task `t`), `ids` `[tasks]` u32 expert per task (None: expert 0,
    /// one task per row). Returns `[tasks, out]` f32.
    #[cfg(feature = "cuda")]
    pub fn forward_tasks(&self, x: &Tensor, ids: Option<&Tensor>, src_div: usize) -> Result<Tensor> {
        let tasks = match ids {
            Some(i) => i.elem_count(),
            None => x.dim(0)?,
        };
        let t = imp::mv(&self.v, self.rank * self.in_dim / 32, self.in_dim / 32, self.rank, x, src_div, ids, Some(&self.s_pre), None, tasks)?;
        imp::mv(&self.u, self.out_dim * self.rank / 32, self.rank / 32, self.out_dim, &t, 1, ids, None, Some(&self.s_post), tasks)
    }

    /// CPU / reference: the dense weight of expert `e` (`[out, in]` f32).
    pub fn dense(&self, e: usize) -> Result<Tensor> {
        let unpack = |w: &Tensor, rows: usize, cols: usize| -> Result<Tensor> {
            let words: Vec<u32> = w.narrow(0, e * rows * cols / 32, rows * cols / 32)?.to_vec1()?;
            let vals: Vec<f32> = words.iter().flat_map(|&x| (0..32).map(move |b| if (x >> b) & 1 == 1 { -1f32 } else { 1f32 })).collect();
            Tensor::from_vec(vals, (rows, cols), &Device::Cpu)
        };
        let u = unpack(&self.u, self.out_dim, self.rank)?;
        let v = unpack(&self.v, self.rank, self.in_dim)?;
        let sp = self.s_post.get(e)?.to_device(&Device::Cpu)?.reshape((self.out_dim, 1))?;
        let si = self.s_pre.get(e)?.to_device(&Device::Cpu)?.reshape((1, self.in_dim))?;
        u.matmul(&v)?.broadcast_mul(&sp)?.broadcast_mul(&si)
    }
}

#[cfg(feature = "cuda")]
mod imp {
    use super::*;
    use candle_core::cuda::cudarc::driver::{DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
    use candle_core::cuda::{CudaDevice, CudaStorage, WrapErr};
    use candle_core::{Shape, Storage};

    fn addr<T: candle_core::cuda::CudaDType>(t: &Tensor, dev: &CudaDevice) -> Result<u64> {
        let (storage, layout) = t.storage_and_layout();
        let Storage::Cuda(cuda) = &*storage else {
            candle_core::bail!("nq: tensor not on CUDA")
        };
        let stream = dev.cuda_stream();
        let (base, _g) = cuda.as_cuda_slice::<T>()?.device_ptr(&stream);
        Ok(base + (layout.start_offset() * std::mem::size_of::<T>()) as u64)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn mv(
        bits: &Tensor,
        mat_words: usize,
        wpr: usize,
        rows: usize,
        x: &Tensor,
        src_div: usize,
        ids: Option<&Tensor>,
        s_in: Option<&Tensor>,
        s_out: Option<&Tensor>,
        tasks: usize,
    ) -> Result<Tensor> {
        let Device::Cuda(dev) = x.device() else {
            candle_core::bail!("nq: input not on CUDA")
        };
        let x = x.to_dtype(DType::F32)?.contiguous()?;
        let ldx = x.dim(x.rank() - 1)?;
        if ldx != wpr * 32 {
            candle_core::bail!("nq: input width {ldx} for {wpr} words per row");
        }
        let ids = ids.map(|i| i.to_dtype(DType::U32)?.flatten_all()?.contiguous()).transpose()?;
        let mut out = unsafe { dev.alloc::<f32>((tasks * rows).max(1)) }?;
        let stream = dev.cuda_stream();
        let (out_p, g) = out.device_ptr_mut(&stream);
        drop(g);
        let bits_p = addr::<u32>(bits, dev)?;
        let x_p = addr::<f32>(&x, dev)?;
        let ids_p = ids.as_ref().map(|i| addr::<u32>(i, dev)).transpose()?.unwrap_or(0);
        let si_p = s_in.map(|s| addr::<f32>(s, dev)).transpose()?.unwrap_or(0);
        let so_p = s_out.map(|s| addr::<f32>(s, dev)).transpose()?.unwrap_or(0);
        let f = dev.get_or_load_custom_func("nq_mv", "titan_nq", ptx()?)?;
        let (mw, wi, ri, li, di) = (mat_words as i64, wpr as i32, rows as i32, ldx as i32, src_div as i32);
        let mut b = f.builder();
        b.arg(&bits_p).arg(&mw).arg(&wi).arg(&ri).arg(&x_p).arg(&li).arg(&di).arg(&ids_p).arg(&si_p).arg(&so_p).arg(&out_p);
        let cfg = LaunchConfig { grid_dim: (rows.div_ceil(ROWS_PER_BLOCK) as u32, tasks as u32, 1), block_dim: ((WARPS * 32) as u32, 1, 1), shared_mem_bytes: 0 };
        unsafe { b.launch(cfg) }.w()?;
        Ok(Tensor::from((Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())), Shape::from_dims(&[tasks, rows]))))
    }
}

/// A factored binary linear as a `QuantMethod`.
#[derive(Debug)]
pub struct NqLinear {
    pub mat: NqMat,
}

impl NqLinear {
    pub fn new_from(mat: NqMat) -> Arc<dyn QuantMethod> {
        Arc::new(Self { mat })
    }
}

impl QuantMethod for NqLinear {
    fn new(_method: QuantMethodConfig) -> Result<Self> {
        candle_core::bail!("NqLinear is built from GGUF nq_* tensors (NqLinear::new_from)")
    }
    fn dequantize_w(&self) -> Result<Tensor> {
        self.mat.dense(0)
    }
    fn forward_raw(&self, a: &Tensor) -> Result<Tensor> {
        let dims = a.dims().to_vec();
        let x = a.reshape(((), self.mat.in_dim))?;
        #[cfg(feature = "cuda")]
        let y = self.mat.forward_tasks(&x, None, 1)?;
        #[cfg(not(feature = "cuda"))]
        let y = x.to_dtype(DType::F32)?.matmul(&self.mat.dense(0)?.t()?)?;
        let mut out = dims;
        *out.last_mut().expect("rank > 0") = self.mat.out_dim;
        y.reshape(out)?.to_dtype(a.dtype())
    }
    fn quantized_act_type(&self) -> Option<DType> {
        None
    }
    fn dtype_and_device(&self) -> (DType, Device) {
        (DType::F32, self.mat.u.device().clone())
    }
    fn plan_isq(&self, request: &crate::IsqRequest) -> Result<crate::IsqPlanParams> {
        Ok(crate::plan_weight_isq(DType::F32, self.mat.u.device().clone(), Vec::new(), request, false))
    }
    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("NqLinear: no LoRA deltas")
    }
    fn apply_isq(
        self: Arc<Self>,
        _dtype: Option<crate::IsqType>,
        _device: Device,
        _n_quantized: &std::sync::atomic::AtomicUsize,
        _imatrix_weight: Option<Vec<f32>>,
        _guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        Ok(self)
    }
}

impl QuantizedSerde for NqLinear {
    fn name(&self) -> &'static str {
        "nanoquant"
    }
}
