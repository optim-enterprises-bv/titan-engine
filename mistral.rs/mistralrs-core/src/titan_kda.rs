//! Fused Kimi Delta Attention scan: one block per (head, sequence), one thread per value column, so the
//! per-key-channel decay, the prediction `S^T k` and the output `S^T q` need no cross-thread reduction.
//! Compiled with NVRTC on first use (no nvcc in the build).

use candle_core::{Result, Tensor};

pub(crate) const HEAD_DIM: usize = 128;

#[cfg(feature = "cuda")]
const SRC: &str = r#"
#define D 128
extern "C" __global__ void kda_scan(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ g, const float* __restrict__ beta,
    const float* __restrict__ state_in, float* __restrict__ state_out, float* __restrict__ out,
    int T, int H)
{
    const int h = blockIdx.x;
    const int n = blockIdx.y;
    const int j = threadIdx.x;
    __shared__ float ks[D];
    __shared__ float qs[D];
    __shared__ float gs[D];
    float s[D];
    const size_t sbase = (size_t)(n * H + h) * D * D;
    #pragma unroll
    for (int a = 0; a < D; a++) s[a] = state_in[sbase + (size_t)a * D + j];
    for (int t = 0; t < T; t++) {
        const size_t row = ((size_t)(n * T + t) * H + h) * D;
        ks[j] = k[row + j];
        qs[j] = q[row + j];
        gs[j] = expf(g[row + j]);
        __syncthreads();
        const float b = beta[(size_t)(n * T + t) * H + h];
        float pred = 0.f;
        #pragma unroll
        for (int a = 0; a < D; a++) {
            s[a] *= gs[a];
            pred = fmaf(s[a], ks[a], pred);
        }
        const float d = (v[row + j] - pred) * b;
        float o = 0.f;
        #pragma unroll
        for (int a = 0; a < D; a++) {
            s[a] = fmaf(ks[a], d, s[a]);
            o = fmaf(s[a], qs[a], o);
        }
        out[row + j] = o;
        __syncthreads();
    }
    #pragma unroll
    for (int a = 0; a < D; a++) state_out[sbase + (size_t)a * D + j] = s[a];
}
"#;

#[cfg(feature = "cuda")]
fn ptx() -> Result<&'static str> {
    use std::sync::OnceLock;
    static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        candle_core::cuda_backend::cudarc::nvrtc::safe::compile_ptx_with_opts(
            SRC,
            candle_core::cuda_backend::cudarc::nvrtc::safe::CompileOptions::default(),
        )
        .map(|p| p.to_src())
        .map_err(|e| format!("{e:?}"))
    })
    .as_ref()
    .map(String::as_str)
    .map_err(|e| candle_core::Error::msg(format!("kda_scan NVRTC compile failed: {e}")))
}

/// `q`, `k`, `v`, `g` are `(n, t, heads, 128)` F32, `beta` is `(n, t, heads)`, `state` is `(n, heads, 128, 128)`
/// with the key axis first. Returns `(o, new_state)` shaped like `v` and `state`.
#[cfg(feature = "cuda")]
pub(crate) fn scan(q: &Tensor, k: &Tensor, v: &Tensor, g: &Tensor, beta: &Tensor, state: &Tensor) -> Result<(Tensor, Tensor)> {
    use candle_core::cuda::cudarc::driver::{DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
    use candle_core::cuda::{CudaStorage, WrapErr};
    use candle_core::{Device, Shape, Storage};

    let (n, t, heads, dim) = q.dims4()?;
    if dim != HEAD_DIM {
        candle_core::bail!("kda_scan: head dim {dim} is not {HEAD_DIM}");
    }
    let Device::Cuda(dev) = q.device().clone() else {
        candle_core::bail!("kda_scan: tensors must be on CUDA")
    };
    let stream = dev.cuda_stream();

    let [q, k, v, g, beta, state] = [q, k, v, g, beta, state].map(Tensor::contiguous);
    let (q, k, v, g, beta, state) = (q?, k?, v?, g?, beta?, state?);

    macro_rules! input {
        ($name:ident, $tensor:expr) => {
            let (storage, layout) = $tensor.storage_and_layout();
            let Storage::Cuda(cuda) = &*storage else {
                candle_core::bail!("kda_scan: {} must be on CUDA", stringify!($name))
            };
            let slice = cuda.as_cuda_slice::<f32>()?;
            let (base, _guard) = slice.device_ptr(&stream);
            let $name = base + (layout.start_offset() * std::mem::size_of::<f32>()) as u64;
        };
    }
    input!(q_ptr, q);
    input!(k_ptr, k);
    input!(v_ptr, v);
    input!(g_ptr, g);
    input!(beta_ptr, beta);
    input!(state_ptr, state);

    let mut out = unsafe { dev.alloc::<f32>(n * t * heads * dim) }?;
    let mut state_out = unsafe { dev.alloc::<f32>(n * heads * dim * dim) }?;
    let (out_ptr, out_guard) = out.device_ptr_mut(&stream);
    let (state_out_ptr, state_out_guard) = state_out.device_ptr_mut(&stream);

    let f = dev.get_or_load_custom_func("kda_scan", "titan_kda", ptx()?)?;
    let (ti, hi) = (t as i32, heads as i32);
    let mut b = f.builder();
    b.arg(&q_ptr).arg(&k_ptr).arg(&v_ptr).arg(&g_ptr).arg(&beta_ptr);
    b.arg(&state_ptr).arg(&state_out_ptr).arg(&out_ptr).arg(&ti).arg(&hi);
    unsafe {
        b.launch(LaunchConfig {
            grid_dim: (heads as u32, n as u32, 1),
            block_dim: (HEAD_DIM as u32, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .w()?;
    drop((out_guard, state_out_guard));

    let wrap = |buf, shape: &[usize]| {
        Tensor::from((Storage::Cuda(CudaStorage::wrap_cuda_slice(buf, dev.clone())), Shape::from_dims(shape)))
    };
    Ok((wrap(out, &[n, t, heads, dim]), wrap(state_out, &[n, heads, dim, dim])))
}
