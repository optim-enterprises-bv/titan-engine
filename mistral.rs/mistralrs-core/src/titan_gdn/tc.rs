//! Chunked gated delta rule prefill on tensor cores (`TITAN_GDN_TC`, default on; `TITAN_GDN_TC_CHECK=1` compares
//! against the sequential kernel). Per 16-token chunk from state S0 with in-chunk cumulative decay gamma:
//! delta = beta v - beta gamma (K S0) - A delta (A[t][i] = beta_t gamma_t/gamma_i k_t.k_i, i < t),
//! y = gamma (Q S0) + M delta (M[t][i] = gamma_t/gamma_i q_t.k_i, i <= t), S = gamma_C S0 + (K gamma_C/gamma)^T delta.
//! Products on mma.m16n8k8 tf32 with f32 accumulation and f32 state; not bit-identical to the per-token order.

use candle_core::{Result, Tensor};

/// Prompt chunks from this many tokens take the tensor-core path.
pub const MIN_TOKENS: usize = 16;

#[cfg(feature = "cuda")]
const SRC: &str = r#"
__device__ __forceinline__ unsigned int tf32(float x) {
    unsigned int r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r;
}
__device__ __forceinline__ void mma(float* d, unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3, unsigned int b0, unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

#define CT 16     // tokens per chunk
#define KD 128    // key dim (state rows)
#define VS 32     // state columns per block
#define SLD 40    // state row stride (bank-conflict free B fragments)
#define QLD 132   // Q/K row stride
#define DLD 40    // delta / K.S0 row stride
#define MLD 20    // A / M row stride

// grid (V / VS, BH), block 128 (4 warps). q, k: [BH, S, 128]; v, out: [BH, S, V]; g, beta: [BH, S];
// state: [BH, 128, V] (row = key index), updated in place.
extern "C" __global__ void __launch_bounds__(128) titan_gdn_tc(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ g, const float* __restrict__ beta, float* __restrict__ state,
    float* __restrict__ out, int S, int V)
{
    __shared__ float Ss[KD * SLD];
    __shared__ float Qs[CT * QLD];
    __shared__ float Ks[CT * QLD];
    __shared__ float Vv[CT * VS];
    __shared__ float KS[CT * DLD];
    __shared__ float Ds[CT * DLD];
    __shared__ float Am[CT * MLD];
    __shared__ float Mm[CT * MLD];
    __shared__ float Gc[CT], Bt[CT], Wd[CT];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int gid = lane >> 2, tig = lane & 3;
    const int v0 = blockIdx.x * VS;
    const int bh = blockIdx.y;
    const float* qb = q + (size_t)bh * S * KD;
    const float* kb = k + (size_t)bh * S * KD;
    const float* vb = v + (size_t)bh * S * V;
    const float* gb = g + (size_t)bh * S;
    const float* bb = beta + (size_t)bh * S;
    float* st = state + (size_t)bh * KD * V;
    float* ob = out + (size_t)bh * S * V;

    for (int e = tid; e < KD * VS; e += 128) {
        const int r = e / VS, c = e % VS;
        Ss[r * SLD + c] = st[(size_t)r * V + v0 + c];
    }

    for (int t0 = 0; t0 < S; t0 += CT) {
        const int valid = min(CT, S - t0);
        // ---- load the chunk (padding rows: q = k = v = 0, g = beta = 0: no effect on the state)
        for (int e = tid; e < CT * KD; e += 128) {
            const int r = e / KD, c = e % KD;
            const bool ok = r < valid;
            Qs[r * QLD + c] = ok ? qb[(size_t)(t0 + r) * KD + c] : 0.f;
            Ks[r * QLD + c] = ok ? kb[(size_t)(t0 + r) * KD + c] : 0.f;
        }
        for (int e = tid; e < CT * VS; e += 128) {
            const int r = e / VS, c = e % VS;
            Vv[e] = r < valid ? vb[(size_t)(t0 + r) * V + v0 + c] : 0.f;
        }
        if (tid == 0) {
            float acc = 0.f;
            for (int r = 0; r < CT; r++) {
                acc += r < valid ? gb[t0 + r] : 0.f;
                Gc[r] = acc;
                Bt[r] = r < valid ? bb[t0 + r] : 0.f;
            }
            for (int r = 0; r < CT; r++) Wd[r] = __expf(Gc[CT - 1] - Gc[r]);
        }
        __syncthreads();

        // ---- K.S0 (smem), Q.S0 (registers): warp w owns state columns 8w..8w+7
        float ks[4] = {0.f, 0.f, 0.f, 0.f}, qs[4] = {0.f, 0.f, 0.f, 0.f};
        {
            const int n0 = warp * 8;
            for (int kk = 0; kk < KD; kk += 8) {
                const unsigned int b0 = tf32(Ss[(kk + tig) * SLD + n0 + gid]);
                const unsigned int b1 = tf32(Ss[(kk + tig + 4) * SLD + n0 + gid]);
                mma(ks, tf32(Ks[gid * QLD + kk + tig]), tf32(Ks[(gid + 8) * QLD + kk + tig]),
                        tf32(Ks[gid * QLD + kk + tig + 4]), tf32(Ks[(gid + 8) * QLD + kk + tig + 4]), b0, b1);
                mma(qs, tf32(Qs[gid * QLD + kk + tig]), tf32(Qs[(gid + 8) * QLD + kk + tig]),
                        tf32(Qs[gid * QLD + kk + tig + 4]), tf32(Qs[(gid + 8) * QLD + kk + tig + 4]), b0, b1);
            }
            KS[gid * DLD + n0 + 2 * tig] = ks[0];
            KS[gid * DLD + n0 + 2 * tig + 1] = ks[1];
            KS[(gid + 8) * DLD + n0 + 2 * tig] = ks[2];
            KS[(gid + 8) * DLD + n0 + 2 * tig + 1] = ks[3];
        }
        // ---- K K^T (warps 0, 1) and Q K^T (warps 2, 3), one 16x8 tile each, scaled and masked into A / M
        {
            const float* X = warp < 2 ? Ks : Qs;
            const int n0 = (warp & 1) * 8;
            float d[4] = {0.f, 0.f, 0.f, 0.f};
            for (int kk = 0; kk < KD; kk += 8) {
                mma(d, tf32(X[gid * QLD + kk + tig]), tf32(X[(gid + 8) * QLD + kk + tig]),
                       tf32(X[gid * QLD + kk + tig + 4]), tf32(X[(gid + 8) * QLD + kk + tig + 4]),
                       tf32(Ks[(n0 + gid) * QLD + kk + tig]), tf32(Ks[(n0 + gid) * QLD + kk + tig + 4]));
            }
            for (int e = 0; e < 4; e++) {
                const int t = gid + 8 * (e >> 1), i = n0 + 2 * tig + (e & 1);
                const float dec = __expf(Gc[t] - Gc[i]);
                if (warp < 2) Am[t * MLD + i] = i < t ? Bt[t] * dec * d[e] : 0.f;
                else Mm[t * MLD + i] = i <= t ? dec * d[e] : 0.f;
            }
        }
        __syncthreads();

        // ---- forward substitution: delta_t = beta_t v_t - beta_t gamma_t (K S0)_t - sum_{i<t} A[t][i] delta_i
        if (warp == 0) {
            const int c = lane;
            float dl[CT];
            for (int t = 0; t < CT; t++) {
                float x = Bt[t] * (Vv[t * VS + c] - __expf(Gc[t]) * KS[t * DLD + c]);
                for (int i = 0; i < t; i++) x -= Am[t * MLD + i] * dl[i];
                dl[t] = x;
                Ds[t * DLD + c] = x;
            }
        }
        __syncthreads();

        // ---- y = gamma (Q S0) + M delta: warp w, columns 8w..8w+7
        {
            const int n0 = warp * 8;
            float y[4] = {__expf(Gc[gid]) * qs[0], __expf(Gc[gid]) * qs[1], __expf(Gc[gid + 8]) * qs[2], __expf(Gc[gid + 8]) * qs[3]};
            for (int kk = 0; kk < CT; kk += 8) {
                mma(y, tf32(Mm[gid * MLD + kk + tig]), tf32(Mm[(gid + 8) * MLD + kk + tig]),
                       tf32(Mm[gid * MLD + kk + tig + 4]), tf32(Mm[(gid + 8) * MLD + kk + tig + 4]),
                       tf32(Ds[(kk + tig) * DLD + n0 + gid]), tf32(Ds[(kk + tig + 4) * DLD + n0 + gid]));
            }
            for (int e = 0; e < 4; e++) {
                const int t = gid + 8 * (e >> 1);
                if (t < valid) ob[(size_t)(t0 + t) * V + v0 + n0 + 2 * tig + (e & 1)] = y[e];
            }
        }
        // ---- S <- gamma_C S0 + (K * gamma_C / gamma)^T delta: warp w, state rows 32w..32w+31, all 32 columns
        {
            const float gC = __expf(Gc[CT - 1]);
            for (int mt = 0; mt < 2; mt++) {
                const int m0 = warp * 32 + mt * 16;
                for (int nt = 0; nt < 4; nt++) {
                    const int n0 = nt * 8;
                    float s[4];
                    s[0] = gC * Ss[(m0 + gid) * SLD + n0 + 2 * tig];
                    s[1] = gC * Ss[(m0 + gid) * SLD + n0 + 2 * tig + 1];
                    s[2] = gC * Ss[(m0 + gid + 8) * SLD + n0 + 2 * tig];
                    s[3] = gC * Ss[(m0 + gid + 8) * SLD + n0 + 2 * tig + 1];
                    for (int kk = 0; kk < CT; kk += 8) {
                        // A[m][i] = k_i[m] * (gamma_C / gamma_i)
                        mma(s, tf32(Ks[(kk + tig) * QLD + m0 + gid] * Wd[kk + tig]),
                               tf32(Ks[(kk + tig) * QLD + m0 + gid + 8] * Wd[kk + tig]),
                               tf32(Ks[(kk + tig + 4) * QLD + m0 + gid] * Wd[kk + tig + 4]),
                               tf32(Ks[(kk + tig + 4) * QLD + m0 + gid + 8] * Wd[kk + tig + 4]),
                               tf32(Ds[(kk + tig) * DLD + n0 + gid]), tf32(Ds[(kk + tig + 4) * DLD + n0 + gid]));
                    }
                    Ss[(m0 + gid) * SLD + n0 + 2 * tig] = s[0];
                    Ss[(m0 + gid) * SLD + n0 + 2 * tig + 1] = s[1];
                    Ss[(m0 + gid + 8) * SLD + n0 + 2 * tig] = s[2];
                    Ss[(m0 + gid + 8) * SLD + n0 + 2 * tig + 1] = s[3];
                }
            }
        }
        __syncthreads();
    }
    for (int e = tid; e < KD * VS; e += 128) {
        const int r = e / VS, c = e % VS;
        st[(size_t)r * V + v0 + c] = Ss[r * SLD + c];
    }
}
"#;

#[cfg(feature = "cuda")]
fn ptx() -> Result<&'static str> {
    use std::sync::OnceLock;
    static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        let opts = candle_core::cuda_backend::cudarc::nvrtc::safe::CompileOptions { arch: Some("compute_80"), ..Default::default() };
        candle_core::cuda_backend::cudarc::nvrtc::safe::compile_ptx_with_opts(SRC, opts).map(|p| p.to_src()).map_err(|e| format!("{e:?}"))
    })
    .as_ref()
    .map(String::as_str)
    .map_err(|e| candle_core::Error::msg(format!("titan_gdn_tc NVRTC compile failed: {e}")))
}

/// Key dim 128 and value dim a multiple of 32.
pub fn supported(q: &Tensor, v: &Tensor) -> bool {
    cfg!(feature = "cuda")
        && q.device().is_cuda()
        && q.dims().len() == 3
        && q.dim(2).is_ok_and(|d| d == 128)
        && v.dim(2).is_ok_and(|d| d % 32 == 0)
        && mistralrs_quant::titan_cfg::flag("TITAN_GDN_TC", true)
}

/// q, k [BH, S, 128], v [BH, S, V], g, beta [BH, S], state [BH, 128, V] (all f32, state updated); returns [BH, S, V].
#[cfg(feature = "cuda")]
pub fn recurrence(q: &Tensor, k: &Tensor, v: &Tensor, g: &Tensor, beta: &Tensor, state: &mut Tensor) -> Result<Tensor> {
    use candle_core::cuda::cudarc::driver::{DevicePtr, LaunchConfig, PushKernelArg};
    use candle_core::{DType, Storage};
    let (bh, s, _) = q.dims3()?;
    let vd = v.dim(2)?;
    let candle_core::Device::Cuda(dev) = q.device() else { candle_core::bail!("titan_gdn_tc: not on CUDA") };
    mistralrs_quant::titan_cfg::note_once("titan_gdn_tc", || "titan GDN prefill: chunked tensor-core recurrence (titan_gdn_tc) on".to_string());
    let (q, k, v, g, beta) = (q.contiguous()?, k.contiguous()?, v.contiguous()?, g.contiguous()?, beta.contiguous()?);
    if !state.is_contiguous() {
        *state = state.contiguous()?;
    }
    for t in [&q, &k, &v, &g, &beta, &*state] {
        if t.dtype() != DType::F32 {
            candle_core::bail!("titan_gdn_tc: f32 inputs and state");
        }
    }
    let stream = dev.cuda_stream();
    let ptr = |t: &Tensor| -> Result<u64> {
        let (st, l) = t.storage_and_layout();
        let Storage::Cuda(cs) = &*st else { candle_core::bail!("titan_gdn_tc: not on CUDA") };
        let base = cs.as_cuda_slice::<f32>()?.device_ptr(&stream).0;
        Ok(base + 4 * l.start_offset() as u64)
    };
    let check = mistralrs_quant::titan_cfg::flag("TITAN_GDN_TC_CHECK", false);
    let state_before = if check { Some(state.copy()?) } else { None };
    let out = unsafe { dev.alloc::<f32>(bh * s * vd)? };
    let (pq, pk, pv, pg, pb, ps) = (ptr(&q)?, ptr(&k)?, ptr(&v)?, ptr(&g)?, ptr(&beta)?, ptr(state)?);
    let po = out.device_ptr(&stream).0;
    let f = dev.get_or_load_custom_func("titan_gdn_tc", "titan_gdn_tc", ptx()?)?;
    let (si, vi) = (s as i32, vd as i32);
    let mut b = f.builder();
    b.arg(&pq).arg(&pk).arg(&pv).arg(&pg).arg(&pb).arg(&ps).arg(&po).arg(&si).arg(&vi);
    let cfg = LaunchConfig { grid_dim: ((vd / 32) as u32, bh as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
    unsafe { b.launch(cfg) }.map_err(candle_core::Error::wrap)?;
    let out = Tensor::from((Storage::Cuda(candle_core::cuda::CudaStorage::wrap_cuda_slice(out, dev.clone())), candle_core::Shape::from((bh, s, vd))));
    if let Some(mut ref_state) = state_before {
        let want = super::cuda_gdn::sequential_recurrence(&q, &k, &v, &g, &beta, &mut ref_state)?;
        let rel = |a: &Tensor, b: &Tensor| -> Result<(f32, f32)> {
            let d = (a - b)?.abs()?;
            let scale = b.abs()?.max_all()?.to_scalar::<f32>()?.max(1e-30);
            let mean_scale = b.abs()?.mean_all()?.to_scalar::<f32>()?.max(1e-30);
            Ok((d.max_all()?.to_scalar::<f32>()? / scale, d.mean_all()?.to_scalar::<f32>()? / mean_scale))
        };
        let (ym, ya) = rel(&out, &want)?;
        let (sm, sa) = rel(state, &ref_state)?;
        tracing::info!("titan_gdn_tc check: S {s}: output rel max {ym:.2e} mean {ya:.2e}; state rel max {sm:.2e} mean {sa:.2e}");
    }
    Ok(out)
}
