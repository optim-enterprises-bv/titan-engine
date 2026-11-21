//! Fused Inkling 2 kernels (short convolution, depth mixing, row clip/RMS), compiled with NVRTC on first use.

use candle_core::{Result, Tensor};

/// Most states one depth mix reads (blocks plus the current one).
pub(crate) const MIX_MAX: usize = 16;
/// Most convolution taps.
pub(crate) const CONV_MAX_TAPS: usize = 8;
/// Attention: aim for this many (kv head, query, split) blocks, never fewer than this many keys per split.
const ATTN_TARGET_BLOCKS: usize = 512;
const ATTN_MIN_KEYS: usize = 64;

#[cfg(feature = "cuda")]
const SRC: &str = r#"
#define MIX_MAX 16
#define CONV_MAX_TAPS 8

// one thread per (sequence, channel): x = [history, cur], y_t = (cur_t + sum_k x[t+k] w[c,k] + dynamic taps) * inv_denom
extern "C" __global__ void ink_conv(
    const float* __restrict__ cur, const float* __restrict__ state, int state_ch, int ch_off,
    const float* __restrict__ w, const float* __restrict__ dyn, int heads,
    float* __restrict__ y, float* __restrict__ state_out, int T, int C, int K, float inv_denom)
{
    const int n = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float win[CONV_MAX_TAPS];
    float wc[CONV_MAX_TAPS];
    const size_t sbase = ((size_t)n * state_ch + ch_off + c) * (K - 1);
    for (int j = 0; j < K - 1; j++) win[j] = state ? state[sbase + j] : 0.f;
    for (int k = 0; k < K; k++) wc[k] = w[(size_t)c * K + k];
    const int per_head = dyn ? C / heads : 1;
    const int hh = dyn ? c / per_head : 0;
    for (int t = 0; t < T; t++) {
        const size_t row = (size_t)n * T + t;
        const float xt = cur[row * C + c];
        win[K - 1] = xt;
        float acc = 0.f;
        for (int k = 0; k < K; k++) acc = fmaf(win[k], wc[k], acc);
        if (dyn) {
            const float* d = dyn + (row * heads + hh) * K;
            for (int lag = 0; lag < K; lag++) acc = fmaf(win[K - 1 - lag], d[lag], acc);
        }
        y[row * C + c] = (xt + acc) * inv_denom;
        for (int k = 0; k < K - 1; k++) win[k] = win[k + 1];
    }
    for (int j = 0; j < K - 1; j++) state_out[sbase + j] = win[j];
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

// block per (row, query): softmax over the RMS-normalized states of (state . query)/sqrt(E), then the weighted sum
extern "C" __global__ void ink_mix(
    const float* p0, const float* p1, const float* p2, const float* p3,
    const float* p4, const float* p5, const float* p6, const float* p7,
    const float* p8, const float* p9, const float* p10, const float* p11,
    const float* p12, const float* p13, const float* p14, const float* p15,
    int nb, const float* __restrict__ query, int nq, float* __restrict__ out, int E, float eps)
{
    const float* ps[MIX_MAX] = {p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15};
    const int row = blockIdx.x;
    const int qi = blockIdx.y;
    const float* q = query + (size_t)qi * E;
    __shared__ float red[2 * MIX_MAX][32];
    __shared__ float coef[MIX_MAX];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int i = 0; i < nb; i++) {
        const float* x = ps[i] + (size_t)row * E;
        float ss = 0.f, dd = 0.f;
        for (int e = threadIdx.x; e < E; e += blockDim.x) {
            const float v = x[e];
            ss = fmaf(v, v, ss);
            dd = fmaf(v, q[e], dd);
        }
        ss = warp_sum(ss);
        dd = warp_sum(dd);
        if (lane == 0) { red[2 * i][warp] = ss; red[2 * i + 1][warp] = dd; }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float score[MIX_MAX];
        float m = -3.402823466e38f;
        for (int i = 0; i < nb; i++) {
            float ss = 0.f, dd = 0.f;
            for (int w2 = 0; w2 < nw; w2++) { ss += red[2 * i][w2]; dd += red[2 * i + 1][w2]; }
            const float inv = rsqrtf(ss / E + eps);
            coef[i] = inv;
            score[i] = dd * inv * rsqrtf((float)E);
            m = fmaxf(m, score[i]);
        }
        float z = 0.f;
        for (int i = 0; i < nb; i++) { score[i] = expf(score[i] - m); z += score[i]; }
        for (int i = 0; i < nb; i++) coef[i] *= score[i] / z;
    }
    __syncthreads();
    float* o = out + ((size_t)row * nq + qi) * E;
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < nb; i++) acc = fmaf(coef[i], ps[i][(size_t)row * E + e], acc);
        o[e] = acc;
    }
}

__device__ __forceinline__ float bf16f(unsigned short b) { return __uint_as_float(((unsigned)b) << 16); }

// flash-decoding style: block (4 warps) per (head, query, sequence); each warp walks every 4th key with an
// online softmax, lanes hold 4 of the 128 dims. logit = q.k * scale + table[dist]; one extra zero-logit sink.
// keys: element (h, j, d) at k + h * kh + j * kj + d (bf16); positions: key j sits at first_key + j.
extern "C" __global__ void ink_attn(
    const float* __restrict__ q, const unsigned short* __restrict__ k, const unsigned short* __restrict__ v,
    long long kh, long long kj, long long vh, long long vj,
    const float* __restrict__ table, float* __restrict__ out,
    int T, int H, int Hkv, int L, int ext, int start, int first_key, int swa, int n_swa, float scale)
{
    const int h = blockIdx.x, ti = blockIdx.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int hk = h / (H / Hkv);
    const int p = start + ti;
    const float* qr = q + ((size_t)ti * H + h) * 128;
    const float4 qv = *(const float4*)(qr + lane * 4);
    const float* tb = table + ((size_t)ti * H + h) * (ext + 1);
    float m = -3.402823466e38f, s = 0.f;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    const unsigned short* kb = k + hk * kh;
    const unsigned short* vb = v + hk * vh;
    for (int j = warp; j < L; j += 4) {
        const int kp = first_key + j;
        if (kp > p || (swa && p - kp >= n_swa)) continue;
        const unsigned short* kr = kb + j * kj + lane * 4;
        float dot = qv.x * bf16f(kr[0]) + qv.y * bf16f(kr[1]) + qv.z * bf16f(kr[2]) + qv.w * bf16f(kr[3]);
        dot = warp_sum(dot);
        const int dist = p - kp;
        const float logit = dot * scale + tb[dist < ext ? dist : ext];
        const float mn = fmaxf(m, logit);
        const float c = __expf(m - mn), e = __expf(logit - mn);
        const unsigned short* vr = vb + j * vj + lane * 4;
        s = s * c + e;
        a0 = a0 * c + e * bf16f(vr[0]);
        a1 = a1 * c + e * bf16f(vr[1]);
        a2 = a2 * c + e * bf16f(vr[2]);
        a3 = a3 * c + e * bf16f(vr[3]);
        m = mn;
    }
    __shared__ float sm[4], ss[4];
    __shared__ float sa[4][128];
    if (lane == 0) { sm[warp] = m; ss[warp] = s; }
    sa[warp][lane * 4] = a0; sa[warp][lane * 4 + 1] = a1; sa[warp][lane * 4 + 2] = a2; sa[warp][lane * 4 + 3] = a3;
    __syncthreads();
    float M = 0.f;
    for (int w = 0; w < 4; w++) M = fmaxf(M, sm[w]);
    float S = __expf(-M);
    for (int w = 0; w < 4; w++) S += ss[w] * __expf(sm[w] - M);
    const int d = threadIdx.x;
    float acc = 0.f;
    for (int w = 0; w < 4; w++) acc += sa[w][d] * __expf(sm[w] - M);
    out[((size_t)ti * H + h) * 128 + d] = acc / S;
}

// split-K flash decoding: block (4 warps) per (kv head, query, key split); the G query heads of the kv head share
// each K/V row. Writes per (split, head) the running max, sum and unnormalized 128-wide accumulator.
#define ATTN_G_MAX 8
extern "C" __global__ void ink_attn_split(
    const float* __restrict__ q, const unsigned short* __restrict__ k, const unsigned short* __restrict__ v,
    long long kh, long long kj, long long vh, long long vj,
    const float* __restrict__ table, float* __restrict__ part,
    int T, int H, int Hkv, int L, int ext, int start, int first_key, int swa, int n_swa, float scale, int S)
{
    const int hk = blockIdx.x, ti = blockIdx.y, sp = blockIdx.z;
    const int G = H / Hkv;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int p = start + ti;
    const int chunk = (L + S - 1) / S;
    const int j0 = sp * chunk, j1 = min(L, j0 + chunk);
    float4 qv[ATTN_G_MAX];
    const float* tb[ATTN_G_MAX];
    float m[ATTN_G_MAX], s[ATTN_G_MAX], a[ATTN_G_MAX][4];
    for (int g = 0; g < G; g++) {
        const int h = hk * G + g;
        qv[g] = *(const float4*)(q + ((size_t)ti * H + h) * 128 + lane * 4);
        tb[g] = table + ((size_t)ti * H + h) * (ext + 1);
        m[g] = -3.402823466e38f; s[g] = 0.f;
        a[g][0] = a[g][1] = a[g][2] = a[g][3] = 0.f;
    }
    const unsigned short* kb = k + hk * kh;
    const unsigned short* vb = v + hk * vh;
    for (int j = j0 + warp; j < j1; j += 4) {
        const int kp = first_key + j;
        if (kp > p || (swa && p - kp >= n_swa)) continue;
        const unsigned short* kr = kb + j * kj + lane * 4;
        const float k0 = bf16f(kr[0]), k1 = bf16f(kr[1]), k2 = bf16f(kr[2]), k3 = bf16f(kr[3]);
        const unsigned short* vr = vb + j * vj + lane * 4;
        const float v0 = bf16f(vr[0]), v1 = bf16f(vr[1]), v2 = bf16f(vr[2]), v3 = bf16f(vr[3]);
        const int dist = p - kp;
        const int idx = dist < ext ? dist : ext;
        for (int g = 0; g < G; g++) {
            float dot = qv[g].x * k0 + qv[g].y * k1 + qv[g].z * k2 + qv[g].w * k3;
            dot = warp_sum(dot);
            const float logit = dot * scale + tb[g][idx];
            const float mn = fmaxf(m[g], logit);
            const float c = __expf(m[g] - mn), e = __expf(logit - mn);
            s[g] = s[g] * c + e;
            a[g][0] = a[g][0] * c + e * v0;
            a[g][1] = a[g][1] * c + e * v1;
            a[g][2] = a[g][2] * c + e * v2;
            a[g][3] = a[g][3] * c + e * v3;
            m[g] = mn;
        }
    }
    __shared__ float sm[4][ATTN_G_MAX], ss[4][ATTN_G_MAX];
    __shared__ float sa[4][ATTN_G_MAX][128];
    for (int g = 0; g < G; g++) {
        if (lane == 0) { sm[warp][g] = m[g]; ss[warp][g] = s[g]; }
        sa[warp][g][lane * 4] = a[g][0]; sa[warp][g][lane * 4 + 1] = a[g][1];
        sa[warp][g][lane * 4 + 2] = a[g][2]; sa[warp][g][lane * 4 + 3] = a[g][3];
    }
    __syncthreads();
    const int d = threadIdx.x;
    for (int g = 0; g < G; g++) {
        float M = -3.402823466e38f;
        for (int w = 0; w < 4; w++) M = fmaxf(M, sm[w][g]);
        float S2 = 0.f, acc = 0.f;
        for (int w = 0; w < 4; w++) {
            const float c = sm[w][g] > -3.0e38f ? __expf(sm[w][g] - M) : 0.f;
            S2 += ss[w][g] * c;
            acc += sa[w][g][d] * c;
        }
        const int h = hk * G + g;
        float* pr = part + (((size_t)ti * H + h) * S + sp) * 130;
        if (d == 0) { pr[0] = M; pr[1] = S2; }
        pr[2 + d] = acc;
    }
}

// combine the splits of one (head, query) with the zero-logit sink
extern "C" __global__ void ink_attn_combine(const float* __restrict__ part, float* __restrict__ out, int H, int S)
{
    const int h = blockIdx.x, ti = blockIdx.y, d = threadIdx.x;
    const float* pr = part + ((size_t)ti * H + h) * S * 130;
    float M = 0.f;
    for (int i = 0; i < S; i++) M = fmaxf(M, pr[i * 130]);
    float den = __expf(-M), acc = 0.f;
    for (int i = 0; i < S; i++) {
        const float mi = pr[i * 130];
        if (mi < -3.0e38f) continue;
        const float c = __expf(mi - M);
        den += pr[i * 130 + 1] * c;
        acc += pr[i * 130 + 2 + d] * c;
    }
    out[((size_t)ti * H + h) * 128 + d] = acc / den;
}

// block per row of D: mode 0 clip (optional per-head scale, then L2 norm capped at thr, then optional per-token
// tau), mode 1 unweighted RMS norm
extern "C" __global__ void ink_rows(
    const float* __restrict__ x, const float* __restrict__ scale, int H, int D, float thr,
    const float* __restrict__ tau, float* __restrict__ out, int mode, float eps)
{
    const size_t r = blockIdx.x;
    const int head = (int)(r % H);
    __shared__ float red[32];
    float ss = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = x[r * D + d];
        if (scale) v *= scale[(size_t)head * D + d];
        ss = fmaf(v, v, ss);
    }
    ss = warp_sum(ss);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    if (threadIdx.x == 0) { float s = 0.f; for (int i = 0; i < nw; i++) s += red[i]; red[0] = s; }
    __syncthreads();
    ss = red[0];
    float f;
    if (mode == 0) {
        f = fminf(fmaxf(thr / (sqrtf(ss) + 1e-6f), 0.f), 1.f);
        if (tau) f *= tau[r / H];
    } else {
        f = rsqrtf(ss / D + eps);
    }
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = x[r * D + d];
        if (scale) v *= scale[(size_t)head * D + d];
        out[r * D + d] = v * f;
    }
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
    .map_err(|e| candle_core::Error::msg(format!("titan_inkling NVRTC compile failed: {e}")))
}

#[cfg(feature = "cuda")]
mod imp {
    use super::*;
    use candle_core::cuda::cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
    use candle_core::cuda::{CudaDevice, CudaStorage, WrapErr};
    use candle_core::{DType, Device, Shape, Storage};

    fn cuda_dev(t: &Tensor) -> Result<CudaDevice> {
        match t.device() {
            Device::Cuda(d) => Ok(d.clone()),
            _ => candle_core::bail!("titan_inkling: tensors must be on CUDA"),
        }
    }

    /// Device address of a contiguous F32 tensor's first element (the tensor must outlive the launch).
    fn addr(t: &Tensor, dev: &CudaDevice) -> Result<u64> {
        if t.dtype() != DType::F32 || !t.is_contiguous() {
            candle_core::bail!("titan_inkling: want a contiguous F32 tensor, got {:?}", t.dtype());
        }
        let (storage, layout) = t.storage_and_layout();
        let Storage::Cuda(cuda) = &*storage else {
            candle_core::bail!("titan_inkling: tensor not on CUDA")
        };
        let stream = dev.cuda_stream();
        let (base, _guard) = cuda.as_cuda_slice::<f32>()?.device_ptr(&stream);
        Ok(base + (layout.start_offset() * std::mem::size_of::<f32>()) as u64)
    }

    fn wrap(buf: CudaSlice<f32>, dev: &CudaDevice, shape: &[usize]) -> Tensor {
        Tensor::from((Storage::Cuda(CudaStorage::wrap_cuda_slice(buf, dev.clone())), Shape::from_dims(shape)))
    }

    fn alloc(dev: &CudaDevice, n: usize) -> Result<(CudaSlice<f32>, u64)> {
        let mut buf = unsafe { dev.alloc::<f32>(n.max(1)) }?;
        let stream = dev.cuda_stream();
        let (p, g) = buf.device_ptr_mut(&stream);
        drop(g);
        Ok((buf, p))
    }

    /// Causal short convolution of `cur` `(n, t, C)` against `w` `(C, K)`; the history is channels
    /// `ch_off..ch_off + C` of `state` `(n, state_ch, K - 1)` (zeros when `None`), the next history goes to the same
    /// channels of `state_out` (same shape, written in place). `dynamic` `(n, t, heads * K)` adds per-head taps.
    pub(crate) fn conv(
        cur: &Tensor,
        w: &Tensor,
        state: Option<&Tensor>,
        state_out: &Tensor,
        ch_off: usize,
        dynamic: Option<(&Tensor, usize)>,
        inv_denom: f32,
    ) -> Result<Tensor> {
        let dev = cuda_dev(cur)?;
        let cur = cur.contiguous()?;
        let (n, t, c) = cur.dims3()?;
        let (wc, k) = w.dims2()?;
        if wc != c || k > CONV_MAX_TAPS || k < 2 {
            candle_core::bail!("titan_inkling conv: weight {:?} for {c} channels", w.shape());
        }
        let state_ch = state_out.dim(1)?;
        let (y, y_ptr) = alloc(&dev, n * t * c)?;
        let cur_p = addr(&cur, &dev)?;
        let w_p = addr(w, &dev)?;
        let st_p = match state {
            Some(s) => addr(s, &dev)?,
            None => 0u64,
        };
        let so_p = addr(state_out, &dev)?;
        let (dyn_p, heads) = match dynamic {
            Some((d, h)) => (addr(d, &dev)?, h as i32),
            None => (0u64, 1),
        };
        let f = dev.get_or_load_custom_func("ink_conv", "titan_inkling", ptx()?)?;
        let (sc, co, ti, ci, ki) = (state_ch as i32, ch_off as i32, t as i32, c as i32, k as i32);
        let mut b = f.builder();
        b.arg(&cur_p).arg(&st_p).arg(&sc).arg(&co).arg(&w_p).arg(&dyn_p).arg(&heads);
        b.arg(&y_ptr).arg(&so_p).arg(&ti).arg(&ci).arg(&ki).arg(&inv_denom);
        unsafe {
            b.launch(LaunchConfig { grid_dim: (c.div_ceil(256) as u32, n as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
        }
        .w()?;
        Ok(wrap(y, &dev, &[n, t, c]))
    }

    /// Depth mix: `states` (each `(rows.., E)` contiguous F32, same shape), `query` `(q, E)`. Returns `(rows.., q, E)`.
    pub(crate) fn mix(states: &[&Tensor], query: &Tensor, eps: f32) -> Result<Tensor> {
        if states.is_empty() || states.len() > MIX_MAX {
            candle_core::bail!("titan_inkling mix: {} states (max {MIX_MAX})", states.len());
        }
        let dev = cuda_dev(states[0])?;
        let owned: Vec<Tensor> = states.iter().map(|s| s.contiguous()).collect::<Result<_>>()?;
        let dims = owned[0].dims().to_vec();
        let e = *dims.last().expect("rank > 0");
        let rows = owned[0].elem_count() / e;
        let query = query.contiguous()?;
        let (nq, qe) = query.dims2()?;
        if qe != e {
            candle_core::bail!("titan_inkling mix: query width {qe} for states of {e}");
        }
        let mut ptrs = [0u64; MIX_MAX];
        for (p, s) in ptrs.iter_mut().zip(&owned) {
            *p = addr(s, &dev)?;
        }
        let q_p = addr(&query, &dev)?;
        let (out, out_p) = alloc(&dev, rows * nq * e)?;
        let f = dev.get_or_load_custom_func("ink_mix", "titan_inkling", ptx()?)?;
        let (nb, nqi, ei) = (owned.len() as i32, nq as i32, e as i32);
        let mut b = f.builder();
        for p in &ptrs {
            b.arg(p);
        }
        b.arg(&nb).arg(&q_p).arg(&nqi).arg(&out_p).arg(&ei).arg(&eps);
        unsafe {
            b.launch(LaunchConfig { grid_dim: (rows as u32, nq as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
        }
        .w()?;
        let mut shape = dims[..dims.len() - 1].to_vec();
        shape.extend([nq, e]);
        Ok(wrap(out, &dev, &shape))
    }

    /// Decode attention for one sequence: `q` `(t, H, 128)` F32, `k`/`v` `(Hkv, L, 128)` BF16 views (last dim
    /// contiguous), `table` `(t, H, ext + 1)`. Returns `(t, H * 128)`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attn(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        table: &Tensor,
        start: usize,
        first_key: usize,
        swa: Option<usize>,
        scale: f32,
    ) -> Result<Tensor> {
        let dev = cuda_dev(q)?;
        let q = q.contiguous()?;
        let table = table.contiguous()?;
        let (t, h, d) = q.dims3()?;
        let (hkv, l, kd) = k.dims3()?;
        let ext = table.dim(2)? - 1;
        if d != 128 || kd != 128 || k.dtype() != DType::BF16 || v.dtype() != DType::BF16 || h % hkv != 0 || h / hkv > 8 {
            candle_core::bail!("titan_inkling attn: want 128-wide heads and BF16 K/V");
        }
        let bf = |x: &Tensor| -> Result<(u64, i64, i64)> {
            let (storage, layout) = x.storage_and_layout();
            let Storage::Cuda(cuda) = &*storage else { candle_core::bail!("titan_inkling attn: K/V not on CUDA") };
            let st = layout.stride();
            if st[2] != 1 {
                candle_core::bail!("titan_inkling attn: K/V last dim must be contiguous");
            }
            let stream = dev.cuda_stream();
            let (base, _g) = cuda.as_cuda_slice::<half::bf16>()?.device_ptr(&stream);
            Ok((base + (layout.start_offset() * 2) as u64, st[0] as i64, st[1] as i64))
        };
        let (k_p, kh, kj) = bf(k)?;
        let (v_p, vh, vj) = bf(v)?;
        let q_p = addr(&q, &dev)?;
        let t_p = addr(&table, &dev)?;
        let (out, out_p) = alloc(&dev, t * h * d)?;
        let (ti, hi, hkvi, li, exti) = (t as i32, h as i32, hkv as i32, l as i32, ext as i32);
        let (st, fk, sw, nsw) = (start as i32, first_key as i32, i32::from(swa.is_some()), swa.unwrap_or(0) as i32);
        // enough (kv head, query, split) blocks to fill the GPU, each split at least ATTN_MIN_KEYS keys
        let splits = (ATTN_TARGET_BLOCKS / (hkv * t).max(1)).clamp(1, l.div_ceil(ATTN_MIN_KEYS).max(1));
        let si = splits as i32;
        let (part, part_p) = alloc(&dev, t * h * splits * 130)?;
        let f = dev.get_or_load_custom_func("ink_attn_split", "titan_inkling", ptx()?)?;
        let mut b = f.builder();
        b.arg(&q_p).arg(&k_p).arg(&v_p).arg(&kh).arg(&kj).arg(&vh).arg(&vj).arg(&t_p).arg(&part_p);
        b.arg(&ti).arg(&hi).arg(&hkvi).arg(&li).arg(&exti).arg(&st).arg(&fk).arg(&sw).arg(&nsw).arg(&scale).arg(&si);
        unsafe {
            b.launch(LaunchConfig { grid_dim: (hkv as u32, t as u32, splits as u32), block_dim: (128, 1, 1), shared_mem_bytes: 0 })
        }
        .w()?;
        let f = dev.get_or_load_custom_func("ink_attn_combine", "titan_inkling", ptx()?)?;
        let mut b = f.builder();
        b.arg(&part_p).arg(&out_p).arg(&hi).arg(&si);
        unsafe {
            b.launch(LaunchConfig { grid_dim: (h as u32, t as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 })
        }
        .w()?;
        drop(part);
        Ok(wrap(out, &dev, &[t, h * d]))
    }

    /// Rows of the last dim `D` of `x` `(.., H, D)`: clip (mode 0) or RMS (mode 1); see the kernel.
    pub(crate) fn rows(x: &Tensor, scale: Option<&Tensor>, thr: f32, tau: Option<&Tensor>, rms_eps: Option<f32>) -> Result<Tensor> {
        let dev = cuda_dev(x)?;
        let x = x.contiguous()?;
        let dims = x.dims().to_vec();
        let d = dims[dims.len() - 1];
        let h = if dims.len() >= 2 { dims[dims.len() - 2] } else { 1 };
        let rows = x.elem_count() / d;
        let x_p = addr(&x, &dev)?;
        let s_p = match scale {
            Some(s) => addr(s, &dev)?,
            None => 0u64,
        };
        let t_p = match tau {
            Some(t) => addr(t, &dev)?,
            None => 0u64,
        };
        let (out, out_p) = alloc(&dev, x.elem_count())?;
        let f = dev.get_or_load_custom_func("ink_rows", "titan_inkling", ptx()?)?;
        let (hi, di, mode, eps) = (h as i32, d as i32, i32::from(rms_eps.is_some()), rms_eps.unwrap_or(0.0));
        let mut b = f.builder();
        b.arg(&x_p).arg(&s_p).arg(&hi).arg(&di).arg(&thr).arg(&t_p).arg(&out_p).arg(&mode).arg(&eps);
        let threads = (d.div_ceil(32) * 32).min(256) as u32;
        unsafe {
            b.launch(LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 })
        }
        .w()?;
        Ok(wrap(out, &dev, &dims))
    }
}

#[cfg(feature = "cuda")]
pub(crate) use imp::{attn, conv, mix, rows};
