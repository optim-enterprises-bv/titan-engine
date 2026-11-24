//! Fused Inkling 2 kernels (short convolution, depth mixing, row clip/RMS), compiled with NVRTC on first use.

use candle_core::{Result, Tensor};

/// Most states one depth mix reads (blocks plus the current one).
pub(crate) const MIX_MAX: usize = 16;
/// Most convolution taps.
pub(crate) const CONV_MAX_TAPS: usize = 8;
/// Attention: aim for this many (kv head, query, split) blocks, never fewer than this many keys per split.
const ATTN_TARGET_BLOCKS: usize = 512;
const ATTN_MIN_KEYS: usize = 64;
/// Chunks of at least this many queries use the tiled prefill kernel (`INKLING_FA=0`: never).
const FA_MIN_T: usize = 16;

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
// F8E4M3 (bias 7, no infinities) to f32, exact: normals 2^(e-7) * (1 + m/8), subnormals m * 2^-9
__device__ __forceinline__ float e4m3f(unsigned int b) {
    const unsigned int s = (b & 0x80u) << 24, e = (b >> 3) & 0xFu, m = b & 7u;
    if (e == 0) return __uint_as_float(s | __float_as_uint((float)m * 0.001953125f));
    return __uint_as_float(s | ((e + 120u) << 23) | (m << 20));
}
// 4 consecutive K/V elements starting at element `e` of a bf16 (fp8 == 0) or F8E4M3 cache
__device__ __forceinline__ float4 kv4(const void* base, long long e, int fp8) {
    if (fp8) {
        const unsigned int w = *(const unsigned int*)((const unsigned char*)base + e);
        return make_float4(e4m3f(w & 0xFF), e4m3f((w >> 8) & 0xFF), e4m3f((w >> 16) & 0xFF), e4m3f(w >> 24));
    }
    const unsigned short* p = (const unsigned short*)base + e;
    return make_float4(bf16f(p[0]), bf16f(p[1]), bf16f(p[2]), bf16f(p[3]));
}

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
    const float* __restrict__ q, const void* __restrict__ k, const void* __restrict__ v,
    long long kh, long long kj, long long vh, long long vj,
    const float* __restrict__ table, float* __restrict__ part,
    int T, int H, int Hkv, int L, int ext, int start, int first_key, int swa, int n_swa, float scale, int S, int fp8)
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
    for (int j = j0 + warp; j < j1; j += 4) {
        const int kp = first_key + j;
        if (kp > p || (swa && p - kp >= n_swa)) continue;
        const float4 kf = kv4(k, hk * kh + (long long)j * kj + lane * 4, fp8);
        const float4 vf = kv4(v, hk * vh + (long long)j * vj + lane * 4, fp8);
        const float k0 = kf.x, k1 = kf.y, k2 = kf.z, k3 = kf.w;
        const float v0 = vf.x, v1 = vf.y, v2 = vf.z, v3 = vf.w;
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
// SiTU of expert gate/up rows gu (rows, 2F), each row scaled by scale[ids[row]] first (scale may be null)
extern "C" __global__ void ink_situ(const float* __restrict__ gu, const unsigned* __restrict__ ids, const float* __restrict__ scale,
                                    float* __restrict__ out, int rows, int F, float beta, float lb)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * F) return;
    const int r = (int)(i / F), c = (int)(i % F);
    const float s = scale ? scale[ids[r]] : 1.f;
    const float g = gu[(long long)r * 2 * F + c] * s;
    float u = gu[(long long)r * 2 * F + F + c] * s;
    const float a = beta * tanhf(g / beta) * (1.f / (1.f + expf(-g)));
    if (lb > 0.f) u = lb * tanhf(u / lb);
    out[i] = a * u;
}

// out[t, e] = sum_j y[t, j, e] * w[t, j] * dscale[ids[t, j]] (dscale may be null)
extern "C" __global__ void ink_wsum(const float* __restrict__ y, const float* __restrict__ w, const unsigned* __restrict__ ids,
                                    const float* __restrict__ dscale, float* __restrict__ out, int T, int k, int E)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)T * E) return;
    const int t = (int)(i / E), e = (int)(i % E);
    float acc = 0.f;
    for (int j = 0; j < k; j++) {
        float ww = w[t * k + j];
        if (dscale) ww *= dscale[ids[t * k + j]];
        acc += y[((long long)t * k + j) * E + e] * ww;
    }
    out[i] = acc;
}

__device__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float s = 0.f;
    for (int i = 0; i < nw; i++) s += red[i];
    return s;
}

// loop controller, step 1: gate[r, j] = down[j] . [rms(cur[r]), rms(msg[r])]; block per (8 gate rows, row r)
extern "C" __global__ void ink_ctrl_gate(const float* __restrict__ cur, const float* __restrict__ msg, const float* __restrict__ down,
                                         float* __restrict__ gate, int E, int rank, float eps)
{
    __shared__ float red[32];
    const int r = blockIdx.y;
    const float* c = cur + (long long)r * E;
    const float* m = msg + (long long)r * E;
    float sc = 0.f, sm = 0.f;
    for (int e = threadIdx.x; e < E; e += blockDim.x) { sc = fmaf(c[e], c[e], sc); sm = fmaf(m[e], m[e], sm); }
    const float ic = rsqrtf(block_sum(sc, red) / E + eps);
    const float im = rsqrtf(block_sum(sm, red) / E + eps);
    const int lane = threadIdx.x & 31;
    const int j = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (j >= rank) return;
    const float* d = down + (long long)j * 2 * E;
    float acc = 0.f;
    for (int e = lane; e < E; e += 32) acc += d[e] * (c[e] * ic) + d[E + e] * (m[e] * im);
    acc = warp_sum(acc);
    if (lane == 0) gate[(long long)r * rank + j] = acc;
}

// loop controller, step 2: out = cur - tanh(forget[e] . g) * cur + tanh(gain[e] . g) * msg; warp per output
extern "C" __global__ void ink_ctrl_out(const float* __restrict__ cur, const float* __restrict__ msg, const float* __restrict__ gate,
                                        const float* __restrict__ gain, const float* __restrict__ forget, float* __restrict__ out,
                                        int E, int rank)
{
    const int r = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int e = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (e >= E) return;
    const float* g = gate + (long long)r * rank;
    float a = 0.f, f = 0.f;
    for (int j = lane; j < rank; j += 32) { a += gain[(long long)e * rank + j] * g[j]; f += forget[(long long)e * rank + j] * g[j]; }
    a = tanhf(warp_sum(a));
    f = tanhf(warp_sum(f));
    if (lane == 0) {
        const long long o = (long long)r * E + e;
        out[o] = (cur[o] - f * cur[o]) + a * msg[o];
    }
}
"#;

/// Prefill attention (tensor cores): compiled for compute_80 (mma.sync tf32), separately from `SRC`.
#[cfg(feature = "cuda")]
const FA_SRC: &str = r#"
__device__ __forceinline__ float bf16f(unsigned short b) { return __uint_as_float(((unsigned int)b) << 16); }
__device__ __forceinline__ float e4m3f(unsigned int b) {
    const unsigned int s = (b & 0x80u) << 24, e = (b >> 3) & 0xFu, m = b & 7u;
    if (e == 0) return __uint_as_float(s | __float_as_uint((float)m * 0.001953125f));
    return __uint_as_float(s | ((e + 120u) << 23) | (m << 20));
}
__device__ __forceinline__ unsigned int tf32(float x) {
    unsigned int r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r;
}
__device__ __forceinline__ void mma_tf32(float* d, const unsigned int* a, unsigned int b0, unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

#define KT 32
#define KLD 132
#define VLD 136
#define PLD 36

// Prefill attention with Inkling's relative-position bias and zero-logit sink. Block = 4 warps = 64 (query, head)
// rows of one kv head (row r = query (r / G) of the block's QT = 64 / G, head g = r % G); K/V tiles of 32 keys go
// through shared memory once per block. Warp w owns rows 16w..16w+15: S = Q K^T and O += P V on mma.m16n8k8 (tf32 in,
// f32 accumulate), online softmax in the accumulator registers (lane holds rows lane/4 and lane/4 + 8). The sink is
// the starting state: running max 0, running sum 1 (exp(0 - 0)).
extern "C" __global__ void __launch_bounds__(128) ink_attn_fa(
    const float* __restrict__ q, const void* __restrict__ k, const void* __restrict__ v,
    long long kh, long long kj, long long vh, long long vj,
    const float* __restrict__ table, float* __restrict__ out,
    int T, int H, int Hkv, int L, int ext, int start, int first_key, int swa, int n_swa, float scale, int fp8)
{
    __shared__ float Ks[KT * KLD];
    __shared__ float Vs[KT * VLD];
    __shared__ float Ps[4][16 * PLD];
    const int hk = blockIdx.x;
    const int G = H / Hkv;
    const int QT = 64 / G;
    const int q0 = blockIdx.y * QT;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int gid = lane >> 2, tig = lane & 3;

    // this lane's two rows
    int rq[2], rh[2], rp[2]; bool rv[2]; const float* rt[2];
    for (int u = 0; u < 2; u++) {
        const int r = warp * 16 + gid + 8 * u;
        const int qi = q0 + r / G;
        rv[u] = (r / G) < QT && qi < T;
        rq[u] = rv[u] ? qi : q0;
        rh[u] = hk * G + r % G;
        rp[u] = start + rq[u];
        rt[u] = table + ((size_t)rq[u] * H + rh[u]) * (ext + 1);
    }
    // Q fragments (tf32), 16 k-steps of 8 dims
    unsigned int qa[16][4];
    {
        const float* q0p = q + ((size_t)rq[0] * H + rh[0]) * 128;
        const float* q1p = q + ((size_t)rq[1] * H + rh[1]) * 128;
        for (int ks = 0; ks < 16; ks++) {
            qa[ks][0] = rv[0] ? tf32(q0p[ks * 8 + tig]) : 0u;
            qa[ks][1] = rv[1] ? tf32(q1p[ks * 8 + tig]) : 0u;
            qa[ks][2] = rv[0] ? tf32(q0p[ks * 8 + tig + 4]) : 0u;
            qa[ks][3] = rv[1] ? tf32(q1p[ks * 8 + tig + 4]) : 0u;
        }
    }
    float o[16][4];
    for (int n = 0; n < 16; n++) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float m[2] = {0.f, 0.f}, l[2] = {0.f, 0.f};  // l: this lane's partial sums (the sink's 1 is added at the end)

    const int qlast = min(q0 + QT, T) - 1;
    const int pmin = start + q0, pmax = start + qlast;
    const int j_end = min(L, pmax - first_key + 1);
    const int j_beg = swa ? max(0, pmin - n_swa + 1 - first_key) : 0;
    float* P = Ps[warp];

    for (int j0 = j_beg; j0 < j_end; j0 += KT) {
        __syncthreads();
        {   // K/V tile -> shared (f32): thread = (key t/4, 32 dims)
            const int kk = threadIdx.x >> 2, d0 = (threadIdx.x & 3) * 32;
            const int j = j0 + kk;
            if (j < j_end && fp8) {
                const uint4* kr = (const uint4*)((const unsigned char*)k + hk * kh + (size_t)j * kj + d0);
                const uint4* vr = (const uint4*)((const unsigned char*)v + hk * vh + (size_t)j * vj + d0);
                for (int c = 0; c < 2; c++) {
                    uint4 a = kr[c], b = vr[c];
                    unsigned int aw[4] = {a.x, a.y, a.z, a.w}, bw[4] = {b.x, b.y, b.z, b.w};
                    for (int e = 0; e < 16; e++) {
                        Ks[kk * KLD + d0 + c * 16 + e] = e4m3f((aw[e >> 2] >> (8 * (e & 3))) & 0xFF);
                        Vs[kk * VLD + d0 + c * 16 + e] = e4m3f((bw[e >> 2] >> (8 * (e & 3))) & 0xFF);
                    }
                }
            } else if (j < j_end) {
                const uint4* kr = (const uint4*)((const unsigned short*)k + hk * kh + (size_t)j * kj + d0);
                const uint4* vr = (const uint4*)((const unsigned short*)v + hk * vh + (size_t)j * vj + d0);
                for (int c = 0; c < 4; c++) {
                    uint4 a = kr[c], b = vr[c];
                    unsigned int aw[4] = {a.x, a.y, a.z, a.w}, bw[4] = {b.x, b.y, b.z, b.w};
                    for (int e = 0; e < 4; e++) {
                        Ks[kk * KLD + d0 + c * 8 + 2 * e] = __uint_as_float(aw[e] << 16);
                        Ks[kk * KLD + d0 + c * 8 + 2 * e + 1] = __uint_as_float(aw[e] & 0xffff0000u);
                        Vs[kk * VLD + d0 + c * 8 + 2 * e] = __uint_as_float(bw[e] << 16);
                        Vs[kk * VLD + d0 + c * 8 + 2 * e + 1] = __uint_as_float(bw[e] & 0xffff0000u);
                    }
                }
            } else {
                for (int c = 0; c < 32; c++) { Ks[kk * KLD + d0 + c] = 0.f; Vs[kk * VLD + d0 + c] = 0.f; }
            }
        }
        __syncthreads();
        // S = Q K^T: 4 n-subtiles of 8 keys
        float s[4][4];
        for (int n = 0; n < 4; n++) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
            for (int ks = 0; ks < 16; ks++) {
                const unsigned int b0 = __float_as_uint(Ks[(n * 8 + gid) * KLD + ks * 8 + tig]);
                const unsigned int b1 = __float_as_uint(Ks[(n * 8 + gid) * KLD + ks * 8 + tig + 4]);
                mma_tf32(s[n], qa[ks], b0, b1);
            }
        }
        // logits, mask, tile max per row
        float tmax[2] = {-3.402823466e38f, -3.402823466e38f};
        for (int n = 0; n < 4; n++)
            for (int e = 0; e < 4; e++) {
                const int u = e >> 1;
                const int j = j0 + n * 8 + 2 * tig + (e & 1);
                const int kp = first_key + j;
                const int p = rp[u];
                const bool ok = rv[u] && j < j_end && kp <= p && (!swa || p - kp < n_swa);
                float x = -3.402823466e38f;
                if (ok) {
                    const int dist = p - kp;
                    x = s[n][e] * scale + rt[u][dist < ext ? dist : ext];
                }
                s[n][e] = x;
                tmax[u] = fmaxf(tmax[u], x);
            }
        for (int u = 0; u < 2; u++) {
            tmax[u] = fmaxf(tmax[u], __shfl_xor_sync(0xffffffff, tmax[u], 1));
            tmax[u] = fmaxf(tmax[u], __shfl_xor_sync(0xffffffff, tmax[u], 2));
        }
        float c[2];
        for (int u = 0; u < 2; u++) {
            const float mn = fmaxf(m[u], tmax[u]);
            c[u] = __expf(m[u] - mn);
            m[u] = mn;
            l[u] *= c[u];
        }
        for (int n = 0; n < 16; n++) { o[n][0] *= c[0]; o[n][1] *= c[0]; o[n][2] *= c[1]; o[n][3] *= c[1]; }
        // P = exp(logit - m) -> shared (A layout for P V)
        __syncwarp();
        for (int n = 0; n < 4; n++)
            for (int e = 0; e < 4; e++) {
                const int u = e >> 1;
                const float pv = s[n][e] > -3.0e38f ? __expf(s[n][e] - m[u]) : 0.f;
                l[u] += pv;
                P[(gid + 8 * u) * PLD + n * 8 + 2 * tig + (e & 1)] = pv;
            }
        __syncwarp();
        // O += P V: 4 k-steps of 8 keys x 16 n-subtiles of 8 dims
        for (int ks = 0; ks < 4; ks++) {
            unsigned int pa[4];
            pa[0] = tf32(P[gid * PLD + ks * 8 + tig]);
            pa[1] = tf32(P[(gid + 8) * PLD + ks * 8 + tig]);
            pa[2] = tf32(P[gid * PLD + ks * 8 + tig + 4]);
            pa[3] = tf32(P[(gid + 8) * PLD + ks * 8 + tig + 4]);
            for (int n = 0; n < 16; n++) {
                const unsigned int b0 = __float_as_uint(Vs[(ks * 8 + tig) * VLD + n * 8 + gid]);
                const unsigned int b1 = __float_as_uint(Vs[(ks * 8 + tig + 4) * VLD + n * 8 + gid]);
                mma_tf32(o[n], pa, b0, b1);
            }
        }
    }
    // row sums over the 4 lanes of the row, plus the sink
    for (int u = 0; u < 2; u++) {
        l[u] += __shfl_xor_sync(0xffffffff, l[u], 1);
        l[u] += __shfl_xor_sync(0xffffffff, l[u], 2);
        l[u] += __expf(-m[u]);
    }
    for (int u = 0; u < 2; u++) {
        if (!rv[u]) continue;
        float* op = out + ((size_t)rq[u] * H + rh[u]) * 128;
        const float inv = 1.f / l[u];
        for (int n = 0; n < 16; n++) {
            op[n * 8 + 2 * tig] = o[n][2 * u] * inv;
            op[n * 8 + 2 * tig + 1] = o[n][2 * u + 1] * inv;
        }
    }
}
"#;

#[cfg(feature = "cuda")]
fn fa_ptx() -> Result<&'static str> {
    use std::sync::OnceLock;
    static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        let opts = candle_core::cuda_backend::cudarc::nvrtc::safe::CompileOptions { arch: Some("compute_80"), ..Default::default() };
        candle_core::cuda_backend::cudarc::nvrtc::safe::compile_ptx_with_opts(FA_SRC, opts).map(|p| p.to_src()).map_err(|e| format!("{e:?}"))
    })
    .as_ref()
    .map(String::as_str)
    .map_err(|e| candle_core::Error::msg(format!("titan_inkling FA NVRTC compile failed: {e}")))
}

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
        let fp8 = k.dtype() == DType::F8E4M3;
        if d != 128 || kd != 128 || !matches!(k.dtype(), DType::BF16 | DType::F8E4M3) || v.dtype() != k.dtype() || h % hkv != 0 || h / hkv > 8 {
            candle_core::bail!("titan_inkling attn: want 128-wide heads and BF16 or F8E4M3 K/V");
        }
        // prefill chunks: tiled tensor-core kernel (K/V read once per 64 (query, head) rows instead of once per query)
        let g = h / hkv;
        if t >= FA_MIN_T && g <= 64 && mistralrs_quant::titan_cfg::flag("INKLING_FA", true) {
            mistralrs_quant::titan_cfg::note_once("ink_attn_fa", || format!("titan_inkling: tiled prefill attention (ink_attn_fa) on: t {t}, H {h}, Hkv {hkv}, L {l}"));
            return attn_fa(&q, k, v, &table, start, first_key, swa, scale).inspect_err(|e| tracing::warn!("titan_inkling attn_fa: {e}"));
        }
        let (k_p, kh, kj) = kv_ptr(k, &dev, 4)?;
        let (v_p, vh, vj) = kv_ptr(v, &dev, 4)?;
        let fp8i = i32::from(fp8);
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
        b.arg(&ti).arg(&hi).arg(&hkvi).arg(&li).arg(&exti).arg(&st).arg(&fk).arg(&sw).arg(&nsw).arg(&scale).arg(&si).arg(&fp8i);
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

    /// Base address (bytes) and (head, row) strides in elements of a BF16 or F8E4M3 K/V view whose last dim is
    /// contiguous; rows and the start must be `align`-byte aligned (the kernels' vector loads).
    fn kv_ptr(x: &Tensor, dev: &CudaDevice, align: usize) -> Result<(u64, i64, i64)> {
        let (storage, layout) = x.storage_and_layout();
        let Storage::Cuda(cuda) = &*storage else { candle_core::bail!("titan_inkling attn: K/V not on CUDA") };
        let st = layout.stride();
        let es = x.dtype().size_in_bytes();
        if st[2] != 1 || (st[1] * es) % align != 0 || (layout.start_offset() * es) % align != 0 {
            candle_core::bail!("titan_inkling attn: K/V rows must be contiguous and {align}-byte aligned");
        }
        let stream = dev.cuda_stream();
        let base = match x.dtype() {
            DType::F8E4M3 => cuda.as_cuda_slice::<float8::F8E4M3>()?.device_ptr(&stream).0,
            _ => cuda.as_cuda_slice::<half::bf16>()?.device_ptr(&stream).0,
        };
        Ok((base + (layout.start_offset() * es) as u64, st[0] as i64, st[1] as i64))
    }

    /// `attn` for prefill chunks: `ink_attn_fa`, grid (Hkv, ceil(t / (64 / G))), 128 threads.
    #[allow(clippy::too_many_arguments)]
    fn attn_fa(q: &Tensor, k: &Tensor, v: &Tensor, table: &Tensor, start: usize, first_key: usize, swa: Option<usize>, scale: f32) -> Result<Tensor> {
        let dev = cuda_dev(q)?;
        let (t, h, d) = q.dims3()?;
        let (hkv, l, _) = k.dims3()?;
        let ext = table.dim(2)? - 1;
        let fp8i = i32::from(k.dtype() == DType::F8E4M3);
        let (k_p, kh, kj) = kv_ptr(k, &dev, 16)?;
        let (v_p, vh, vj) = kv_ptr(v, &dev, 16)?;
        let q_p = addr(q, &dev)?;
        let t_p = addr(table, &dev)?;
        let (out, out_p) = alloc(&dev, t * h * d)?;
        let qt = 64 / (h / hkv);
        let (ti, hi, hkvi, li, exti) = (t as i32, h as i32, hkv as i32, l as i32, ext as i32);
        let (st, fk, sw, nsw) = (start as i32, first_key as i32, i32::from(swa.is_some()), swa.unwrap_or(0) as i32);
        let f = dev.get_or_load_custom_func("ink_attn_fa", "titan_inkling_fa", fa_ptx()?)?;
        let mut b = f.builder();
        b.arg(&q_p).arg(&k_p).arg(&v_p).arg(&kh).arg(&kj).arg(&vh).arg(&vj).arg(&t_p).arg(&out_p);
        b.arg(&ti).arg(&hi).arg(&hkvi).arg(&li).arg(&exti).arg(&st).arg(&fk).arg(&sw).arg(&nsw).arg(&scale).arg(&fp8i);
        unsafe {
            b.launch(LaunchConfig { grid_dim: (hkv as u32, t.div_ceil(qt) as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 })
        }
        .w()?;
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

    fn addr_u32(t: &Tensor, dev: &CudaDevice) -> Result<u64> {
        if t.dtype() != DType::U32 || !t.is_contiguous() {
            candle_core::bail!("titan_inkling: want a contiguous U32 tensor, got {:?}", t.dtype());
        }
        let (storage, layout) = t.storage_and_layout();
        let Storage::Cuda(cuda) = &*storage else {
            candle_core::bail!("titan_inkling: tensor not on CUDA")
        };
        let stream = dev.cuda_stream();
        let (base, _guard) = cuda.as_cuda_slice::<u32>()?.device_ptr(&stream);
        Ok(base + (layout.start_offset() * std::mem::size_of::<u32>()) as u64)
    }

    fn cfg1d(n: usize) -> LaunchConfig {
        LaunchConfig { grid_dim: (n.div_ceil(256).max(1) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
    }

    /// SiTU of `gu` `(.., 2F)` (rows in `ids` order), each row first scaled by `scale[ids[row]]`. Returns `(.., F)`.
    pub(crate) fn situ(gu: &Tensor, ids: &Tensor, scale: Option<&Tensor>, beta: f32, lb: f32) -> Result<Tensor> {
        let dev = cuda_dev(gu)?;
        let gu = gu.contiguous()?;
        let dims = gu.dims().to_vec();
        let f2 = *dims.last().expect("rank > 0");
        let (rows, ff) = (gu.elem_count() / f2, f2 / 2);
        let ids = ids.flatten_all()?.contiguous()?;
        let (g_p, i_p) = (addr(&gu, &dev)?, addr_u32(&ids, &dev)?);
        let s_p = scale.map(|s| addr(s, &dev)).transpose()?.unwrap_or(0);
        let (out, out_p) = alloc(&dev, rows * ff)?;
        let (ri, fi) = (rows as i32, ff as i32);
        let f = dev.get_or_load_custom_func("ink_situ", "titan_inkling", ptx()?)?;
        let mut b = f.builder();
        b.arg(&g_p).arg(&i_p).arg(&s_p).arg(&out_p).arg(&ri).arg(&fi).arg(&beta).arg(&lb);
        unsafe { b.launch(cfg1d(rows * ff)) }.w()?;
        let mut shape = dims;
        *shape.last_mut().expect("rank > 0") = ff;
        Ok(wrap(out, &dev, &shape))
    }

    /// `y` `(T, k, E)` weighted by `w` `(T, k)` times `dscale[ids]`, summed over k: `(T, E)`.
    pub(crate) fn wsum(y: &Tensor, w: &Tensor, ids: &Tensor, dscale: Option<&Tensor>) -> Result<Tensor> {
        let dev = cuda_dev(y)?;
        let y = y.contiguous()?;
        let (t, k, e) = y.dims3()?;
        let w = w.to_dtype(DType::F32)?.contiguous()?;
        let ids = ids.contiguous()?;
        let (y_p, w_p, i_p) = (addr(&y, &dev)?, addr(&w, &dev)?, addr_u32(&ids, &dev)?);
        let d_p = dscale.map(|s| addr(s, &dev)).transpose()?.unwrap_or(0);
        let (out, out_p) = alloc(&dev, t * e)?;
        let (ti, ki, ei) = (t as i32, k as i32, e as i32);
        let f = dev.get_or_load_custom_func("ink_wsum", "titan_inkling", ptx()?)?;
        let mut b = f.builder();
        b.arg(&y_p).arg(&w_p).arg(&i_p).arg(&d_p).arg(&out_p).arg(&ti).arg(&ki).arg(&ei);
        unsafe { b.launch(cfg1d(t * e)) }.w()?;
        Ok(wrap(out, &dev, &[t, e]))
    }

    /// Loop controller: `cur`, `msg` `(.., E)`; `down` `(rank, 2E)`, `gain` / `forget` `(E, rank)`.
    pub(crate) fn control(cur: &Tensor, msg: &Tensor, down: &Tensor, gain: &Tensor, forget: &Tensor, eps: f32) -> Result<Tensor> {
        let dev = cuda_dev(cur)?;
        let (cur, msg) = (cur.contiguous()?, msg.contiguous()?);
        let dims = cur.dims().to_vec();
        let e = *dims.last().expect("rank > 0");
        let rows = cur.elem_count() / e;
        let rank = down.dim(0)?;
        let (c_p, m_p, d_p, g_p, f_p) = (addr(&cur, &dev)?, addr(&msg, &dev)?, addr(down, &dev)?, addr(gain, &dev)?, addr(forget, &dev)?);
        let (gate, gate_p) = alloc(&dev, rows * rank)?;
        let (out, out_p) = alloc(&dev, rows * e)?;
        let (ei, ri) = (e as i32, rank as i32);
        let f = dev.get_or_load_custom_func("ink_ctrl_gate", "titan_inkling", ptx()?)?;
        let mut b = f.builder();
        b.arg(&c_p).arg(&m_p).arg(&d_p).arg(&gate_p).arg(&ei).arg(&ri).arg(&eps);
        unsafe { b.launch(LaunchConfig { grid_dim: (rank.div_ceil(8) as u32, rows as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }.w()?;
        let f = dev.get_or_load_custom_func("ink_ctrl_out", "titan_inkling", ptx()?)?;
        let mut b = f.builder();
        b.arg(&c_p).arg(&m_p).arg(&gate_p).arg(&g_p).arg(&f_p).arg(&out_p).arg(&ei).arg(&ri);
        unsafe { b.launch(LaunchConfig { grid_dim: (e.div_ceil(8) as u32, rows as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }.w()?;
        drop(gate);
        Ok(wrap(out, &dev, &dims))
    }
}

#[cfg(feature = "cuda")]
pub(crate) use imp::{attn, control, conv, mix, rows, situ, wsum};
