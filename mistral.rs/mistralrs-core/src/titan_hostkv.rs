//! Host-RAM KV cache for long contexts (`TITAN_KV_HOST=1`).
//!
//! CUDA KV caches keep their K/V as F8E4M3 tensors on the CPU (32 KiB per token on the dense 27B, so 256k tokens
//! take 8 GiB of RAM instead of VRAM that is not there). `KvCache::append` moves the new rows over; attention with
//! a CUDA query over a CPU cache lands in `attend`: decode and verify rows run on the CPU, reading each cached key
//! once for all query heads of its KV head, and prompt chunks stream the cache to the GPU tile by tile
//! (`flash_prefill::attend_host_tiles`).
//!
//! Keys are cut into fixed `KEY_CHUNK` ranges merged in order, so a row's result depends only on its own key count
//! and an MTP verify row equals the decode step it replaces.

use std::sync::LazyLock;

use candle_core::{DType, Device, Result, Storage, Tensor};
use rayon::prelude::*;

use crate::attention::SdpaParams;

const HEAD_DIM: usize = 256;
/// Query rows (decode, MTP verify) that attend on the CPU; longer prompt chunks go to the GPU.
const CPU_MAX_ROWS: usize = 8;
/// Keys per CPU work unit; fixed so the split (and the rounding) never depends on the call's other rows.
const KEY_CHUNK: usize = 1024;
/// Default `TITAN_KV_HOST_TILE`: cached keys per host-to-device tile of a prompt chunk.
const DEFAULT_TILE: usize = 8192;
/// e4m3 shifted into an f16 is 2^-8 of its value (exponent bias 7 vs 15).
const E4M3_AS_F16_SCALE: f32 = 256.0;

pub(crate) fn enabled() -> bool {
    static ON: mistralrs_quant::titan_cfg::GenCell<bool> = mistralrs_quant::titan_cfg::GenCell::new();
    *ON.get_or_init(|| {
        let on = mistralrs_quant::titan_cfg::var("TITAN_KV_HOST").is_ok_and(|v| v == "1");
        if on {
            tracing::info!(
                "titan kv: host-RAM F8E4M3 KV cache, CPU attention up to {CPU_MAX_ROWS} rows, {} keys per prompt tile",
                tile()
            );
        }
        on
    })
}

fn tile() -> usize {
    static N: mistralrs_quant::titan_cfg::GenCell<usize> = mistralrs_quant::titan_cfg::GenCell::new();
    *N.get_or_init(|| {
        mistralrs_quant::titan_cfg::var("TITAN_KV_HOST_TILE").ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_TILE)
    })
}

static POOL: LazyLock<rayon::ThreadPool> = LazyLock::new(|| {
    rayon::ThreadPoolBuilder::new()
        .num_threads(std::thread::available_parallelism().map_or(1, |n| n.get()))
        .thread_name(|i| format!("titan-hostkv-{i}"))
        .build()
        .expect("titan host-kv thread pool")
});

/// Causal attention of the last `s` positions of `q` (1, h, s, 256) on CUDA over a host F8E4M3 cache `k`, `v`
/// (1, kvh, kv_len, 256): (1, h, s, 256) in `q`'s dtype and device.
pub(crate) fn attend(q: &Tensor, k: &Tensor, v: &Tensor, p: &SdpaParams) -> Result<Tensor> {
    let (b, h, s, d) = q.dims4()?;
    let (kb, kvh, kv_len, kd) = k.dims4()?;
    if b != 1 || kb != 1 || d != HEAD_DIM || kd != HEAD_DIM || h % kvh != 0 || kv_len < s || v.dims4()? != k.dims4()? {
        candle_core::bail!("titan host kv: unsupported q {:?} k {:?} v {:?}", q.shape(), k.shape(), v.shape());
    }
    if k.dtype() != DType::F8E4M3 || v.dtype() != DType::F8E4M3 {
        candle_core::bail!("titan host kv: K/V must be F8E4M3, got {:?}", k.dtype());
    }
    if s > CPU_MAX_ROWS {
        return crate::attention::flash_prefill_host_tiles(q, k, v, p, tile());
    }
    let qh = q.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.contiguous()?;
    let out = {
        let (qg, ql) = qh.storage_and_layout();
        let (kg, kl) = k.storage_and_layout();
        let (vg, vl) = v.storage_and_layout();
        let (Storage::Cpu(qc), Storage::Cpu(kc), Storage::Cpu(vc)) = (&*qg, &*kg, &*vg) else {
            candle_core::bail!("titan host kv: q / K / V not on the CPU");
        };
        let qd = &qc.as_slice::<f32>()?[ql.start_offset()..];
        let kd = as_bytes(&kc.as_slice::<float8::F8E4M3>()?[kl.start_offset()..]);
        let vd = as_bytes(&vc.as_slice::<float8::F8E4M3>()?[vl.start_offset()..]);
        let shape = Shape { h, kvh, s, kv_len, k_hs: k.stride()[1], v_hs: v.stride()[1], scale: p.softmax_scale };
        POOL.install(|| run(qd, kd, vd, &shape))
    };
    Tensor::from_vec(out, (1, h, s, HEAD_DIM), &Device::Cpu)?.to_device(q.device())?.to_dtype(q.dtype())
}

fn as_bytes(x: &[float8::F8E4M3]) -> &[u8] {
    // F8E4M3 is #[repr(transparent)] over u8
    unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len()) }
}

#[derive(Clone, Copy)]
struct Shape {
    h: usize,
    kvh: usize,
    s: usize,
    kv_len: usize,
    k_hs: usize,
    v_hs: usize,
    scale: f32,
}

/// One unit's online-softmax state per (row, query head of the group): un-normalised o, then (max, sum).
struct Partial(Vec<f32>);

const STRIDE: usize = HEAD_DIM + 2;
/// Keys dequantised per block (32 KiB of f32 K, reused from L1 by every query of the group).
const BK: usize = 32;

fn run(q: &[f32], k: &[u8], v: &[u8], sh: &Shape) -> Vec<f32> {
    let n_rep = sh.h / sh.kvh;
    let nq = n_rep * sh.s;
    let chunks = sh.kv_len.div_ceil(KEY_CHUNK);
    let units: Vec<(usize, usize)> = (0..sh.kvh).flat_map(|g| (0..chunks).map(move |c| (g, c))).collect();
    let parts: Vec<Partial> = units.par_iter().map(|&(g, c)| unit(q, k, v, sh, g, c * KEY_CHUNK)).collect();
    let mut out = vec![0f32; sh.h * sh.s * HEAD_DIM];
    out.par_chunks_mut(nq * HEAD_DIM).enumerate().for_each(|(g, out)| {
        let ps = &parts[g * chunks..(g + 1) * chunks];
        for qi in 0..nq {
            let m = ps.iter().map(|p| p.0[qi * STRIDE + HEAD_DIM]).fold(f32::NEG_INFINITY, f32::max);
            let o = &mut out[qi * HEAD_DIM..][..HEAD_DIM];
            let mut l = 0f32;
            for p in ps {
                let pm = p.0[qi * STRIDE + HEAD_DIM];
                if pm == f32::NEG_INFINITY {
                    continue;
                }
                let w = (pm - m).exp();
                l += w * p.0[qi * STRIDE + HEAD_DIM + 1];
                for (o, a) in o.iter_mut().zip(&p.0[qi * STRIDE..qi * STRIDE + HEAD_DIM]) {
                    *o += w * a;
                }
            }
            let inv = 1.0 / l;
            o.iter_mut().for_each(|x| *x *= inv);
        }
    });
    out
}

/// Keys `a .. a + KEY_CHUNK` of KV head `g` for every row and query head of the group (query `qi` = head
/// `qi / s` of the group, row `qi % s`), online softmax over blocks of `BK` keys. Row `r` sees keys
/// `< kv_len - s + r + 1`; only those enter its block, so its operations do not depend on the other rows.
fn unit(q: &[f32], k: &[u8], v: &[u8], sh: &Shape, g: usize, a: usize) -> Partial {
    let n_rep = sh.h / sh.kvh;
    let nq = n_rep * sh.s;
    let past = sh.kv_len - sh.s;
    let b = (a + KEY_CHUNK).min(sh.kv_len);
    let mut part = vec![0f32; nq * STRIDE];
    let mut m = vec![f32::NEG_INFINITY; nq];
    let mut l = vec![0f32; nq];
    let mut kb = vec![0f32; BK * HEAD_DIM];
    let mut vb = vec![0f32; BK * HEAD_DIM];
    let mut p = [0f32; BK];
    let mut j0 = a;
    while j0 < b {
        let n = BK.min(b - j0);
        for j in 0..n {
            dequant(&k[g * sh.k_hs + (j0 + j) * HEAD_DIM..][..HEAD_DIM], &mut kb[j * HEAD_DIM..][..HEAD_DIM]);
        }
        let mut v_ready = false;
        for qi in 0..nq {
            let vis = (past + qi % sh.s + 1).saturating_sub(j0).min(n);
            if vis == 0 {
                continue;
            }
            let qr = &q[((g * n_rep + qi / sh.s) * sh.s + qi % sh.s) * HEAD_DIM..][..HEAD_DIM];
            scores(qr, &kb, vis, sh.scale, &mut p);
            let bm = p[..vis].iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mn = m[qi].max(bm);
            let alpha = if m[qi] == f32::NEG_INFINITY { 0.0 } else { (m[qi] - mn).exp() };
            let ps = exp_sub_sum(&mut p[..vis], mn);
            l[qi] = l[qi] * alpha + ps;
            m[qi] = mn;
            if !v_ready {
                for j in 0..n {
                    dequant(&v[g * sh.v_hs + (j0 + j) * HEAD_DIM..][..HEAD_DIM], &mut vb[j * HEAD_DIM..][..HEAD_DIM]);
                }
                v_ready = true;
            }
            pv(&mut part[qi * STRIDE..][..HEAD_DIM], alpha, &p[..vis], &vb);
        }
        j0 += n;
    }
    for qi in 0..nq {
        part[qi * STRIDE + HEAD_DIM] = m[qi];
        part[qi * STRIDE + HEAD_DIM + 1] = l[qi];
    }
    Partial(part)
}

fn dequant(src: &[u8], dst: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if simd() {
        unsafe { avx::dequant(src.as_ptr(), dst.as_mut_ptr()) };
        return;
    }
    static LUT: LazyLock<[f32; 256]> = LazyLock::new(|| std::array::from_fn(|b| float8::F8E4M3::from_bits(b as u8).to_f32()));
    for (d, &b) in dst.iter_mut().zip(&src[..HEAD_DIM]) {
        *d = LUT[b as usize];
    }
}

/// `out[j] = scale * q . kb[j]` for the first `n` rows of the block `kb`.
fn scores(q: &[f32], kb: &[f32], n: usize, scale: f32, out: &mut [f32; BK]) {
    #[cfg(target_arch = "x86_64")]
    if simd() {
        unsafe { avx::scores(q.as_ptr(), kb.as_ptr(), n, scale, out.as_mut_ptr()) };
        return;
    }
    for (j, o) in out.iter_mut().enumerate().take(n) {
        *o = scale * q.iter().zip(&kb[j * HEAD_DIM..][..HEAD_DIM]).map(|(a, b)| a * b).sum::<f32>();
    }
}

/// `p[j] = exp(p[j] - m)`, returning their sum (summed in order).
fn exp_sub_sum(p: &mut [f32], m: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if simd() {
        return unsafe { avx::exp_sub_sum(p.as_mut_ptr(), p.len(), m) };
    }
    let mut s = 0f32;
    for x in p.iter_mut() {
        *x = (*x - m).exp();
        s += *x;
    }
    s
}

/// `acc = alpha * acc + sum_j p[j] * vb[j]`.
fn pv(acc: &mut [f32], alpha: f32, p: &[f32], vb: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    if simd() {
        unsafe { avx::pv(acc.as_mut_ptr(), alpha, p.as_ptr(), p.len(), vb.as_ptr()) };
        return;
    }
    for (d, a) in acc.iter_mut().enumerate() {
        let mut x = *a * alpha;
        for (j, pj) in p.iter().enumerate() {
            x += pj * vb[j * HEAD_DIM + d];
        }
        *a = x;
    }
}

#[cfg(target_arch = "x86_64")]
fn simd() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| {
        std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c")
    });
    *ON
}

#[cfg(target_arch = "x86_64")]
mod avx {
    use std::arch::x86_64::*;

    use super::{E4M3_AS_F16_SCALE, HEAD_DIM};

    /// 256 e4m3 bytes to f32: sign and the 7 exponent/mantissa bits shifted into an f16, converted, times 2^8
    /// (exact for normals and subnormals; this cache never holds NaN).
    #[target_feature(enable = "avx2,f16c,fma")]
    pub(super) unsafe fn dequant(src: *const u8, dst: *mut f32) {
        let scale = _mm256_set1_ps(E4M3_AS_F16_SCALE);
        let sign = _mm256_set1_epi16(0x80);
        let mag = _mm256_set1_epi16(0x7f);
        for i in (0..HEAD_DIM).step_by(16) {
            let b = _mm256_cvtepu8_epi16(_mm_loadu_si128(src.add(i) as *const __m128i));
            let h = _mm256_or_si256(
                _mm256_slli_epi16::<8>(_mm256_and_si256(b, sign)),
                _mm256_slli_epi16::<7>(_mm256_and_si256(b, mag)),
            );
            let lo = _mm256_cvtph_ps(_mm256_castsi256_si128(h));
            let hi = _mm256_cvtph_ps(_mm256_extracti128_si256::<1>(h));
            _mm256_storeu_ps(dst.add(i), _mm256_mul_ps(lo, scale));
            _mm256_storeu_ps(dst.add(i + 8), _mm256_mul_ps(hi, scale));
        }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps::<1>(s, s)))
    }

    /// Four keys per pass over q: each q load feeds four FMAs, one horizontal sum per key.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn scores(q: *const f32, kb: *const f32, n: usize, scale: f32, out: *mut f32) {
        let mut j = 0;
        while j + 4 <= n {
            let k0 = kb.add(j * HEAD_DIM);
            let mut a = [_mm256_setzero_ps(); 4];
            for d in (0..HEAD_DIM).step_by(8) {
                let qv = _mm256_loadu_ps(q.add(d));
                for (u, a) in a.iter_mut().enumerate() {
                    *a = _mm256_fmadd_ps(qv, _mm256_loadu_ps(k0.add(u * HEAD_DIM + d)), *a);
                }
            }
            for (u, a) in a.iter().enumerate() {
                *out.add(j + u) = scale * hsum(*a);
            }
            j += 4;
        }
        while j < n {
            let k0 = kb.add(j * HEAD_DIM);
            let mut a = _mm256_setzero_ps();
            for d in (0..HEAD_DIM).step_by(8) {
                a = _mm256_fmadd_ps(_mm256_loadu_ps(q.add(d)), _mm256_loadu_ps(k0.add(d)), a);
            }
            *out.add(j) = scale * hsum(a);
            j += 1;
        }
    }

    /// Cephes-style exp for 8 lanes (relative error ~1e-7 over the softmax range).
    #[target_feature(enable = "avx2,fma")]
    unsafe fn exp8(x: __m256) -> __m256 {
        let x = _mm256_max_ps(_mm256_min_ps(x, _mm256_set1_ps(88.376_26)), _mm256_set1_ps(-87.336_54));
        let fx = _mm256_floor_ps(_mm256_fmadd_ps(x, _mm256_set1_ps(std::f32::consts::LOG2_E), _mm256_set1_ps(0.5)));
        let x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693_359_4), x);
        let x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.121_944_4e-4), x);
        let mut y = _mm256_set1_ps(1.987_569_1e-4);
        for c in [1.398_199_9e-3, 8.333_452e-3, 4.166_579_6e-2, 0.166_666_65, 0.5] {
            y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(c));
        }
        let y = _mm256_add_ps(_mm256_fmadd_ps(y, _mm256_mul_ps(x, x), x), _mm256_set1_ps(1.0));
        let n = _mm256_slli_epi32::<23>(_mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127)));
        _mm256_mul_ps(y, _mm256_castsi256_ps(n))
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn exp_sub_sum(p: *mut f32, n: usize, m: f32) -> f32 {
        let mv = _mm256_set1_ps(m);
        let mut s = 0f32;
        let mut j = 0;
        while j + 8 <= n {
            let e = exp8(_mm256_sub_ps(_mm256_loadu_ps(p.add(j)), mv));
            _mm256_storeu_ps(p.add(j), e);
            s += hsum(e);
            j += 8;
        }
        if j < n {
            let mut t = [0f32; 8];
            std::ptr::copy_nonoverlapping(p.add(j), t.as_mut_ptr(), n - j);
            let e = exp8(_mm256_sub_ps(_mm256_loadu_ps(t.as_ptr()), mv));
            _mm256_storeu_ps(t.as_mut_ptr(), e);
            for (u, t) in t.iter().enumerate().take(n - j) {
                *p.add(j + u) = *t;
                s += t;
            }
        }
        s
    }

    /// `acc = alpha * acc + sum_j p[j] * vb[j]`, 8 dims at a time held in a register over the block's keys.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn pv(acc: *mut f32, alpha: f32, p: *const f32, n: usize, vb: *const f32) {
        let al = _mm256_set1_ps(alpha);
        for d in (0..HEAD_DIM).step_by(32) {
            let mut r = [
                _mm256_mul_ps(al, _mm256_loadu_ps(acc.add(d))),
                _mm256_mul_ps(al, _mm256_loadu_ps(acc.add(d + 8))),
                _mm256_mul_ps(al, _mm256_loadu_ps(acc.add(d + 16))),
                _mm256_mul_ps(al, _mm256_loadu_ps(acc.add(d + 24))),
            ];
            for j in 0..n {
                let w = _mm256_set1_ps(*p.add(j));
                let row = vb.add(j * HEAD_DIM + d);
                for (u, r) in r.iter_mut().enumerate() {
                    *r = _mm256_fmadd_ps(w, _mm256_loadu_ps(row.add(8 * u)), *r);
                }
            }
            for (u, r) in r.iter().enumerate() {
                _mm256_storeu_ps(acc.add(d + 8 * u), *r);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequant_matches_float8() {
        let src: Vec<u8> = (0..HEAD_DIM).map(|i| (i as u8).wrapping_mul(37)).filter(|b| b & 0x7f != 0x7f).cycle().take(HEAD_DIM).collect();
        let mut got = [0f32; HEAD_DIM];
        dequant(&src, &mut got);
        for (b, g) in src.iter().zip(got) {
            assert_eq!(float8::F8E4M3::from_bits(*b).to_f32(), g, "byte {b:#04x}");
        }
    }

    #[test]
    fn matches_reference_attention() {
        let (h, kvh, s, kv_len) = (6, 2, 3, 2500);
        let mut seed = 1u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let q: Vec<f32> = (0..h * s * HEAD_DIM).map(|_| rnd()).collect();
        let kf: Vec<float8::F8E4M3> = (0..kvh * kv_len * HEAD_DIM).map(|_| float8::F8E4M3::from_f32(4.0 * rnd())).collect();
        let vf: Vec<float8::F8E4M3> = (0..kvh * kv_len * HEAD_DIM).map(|_| float8::F8E4M3::from_f32(4.0 * rnd())).collect();
        let sh = Shape { h, kvh, s, kv_len, k_hs: kv_len * HEAD_DIM, v_hs: kv_len * HEAD_DIM, scale: 0.0625 };
        let got = run(&q, as_bytes(&kf), as_bytes(&vf), &sh);
        let n_rep = h / kvh;
        for hd in 0..h {
            let g = hd / n_rep;
            for r in 0..s {
                let lim = kv_len - s + r + 1;
                let qv = &q[(hd * s + r) * HEAD_DIM..][..HEAD_DIM];
                let sc: Vec<f64> = (0..lim)
                    .map(|j| (0..HEAD_DIM).map(|d| qv[d] as f64 * kf[(g * kv_len + j) * HEAD_DIM + d].to_f64()).sum::<f64>() * 0.0625)
                    .collect();
                let m = sc.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let w: Vec<f64> = sc.iter().map(|x| (x - m).exp()).collect();
                let l: f64 = w.iter().sum();
                for d in 0..HEAD_DIM {
                    let want = (0..lim).map(|j| w[j] * vf[(g * kv_len + j) * HEAD_DIM + d].to_f64()).sum::<f64>() / l;
                    let gotv = got[(hd * s + r) * HEAD_DIM + d] as f64;
                    assert!((want - gotv).abs() < 1e-4, "head {hd} row {r} dim {d}: {gotv} vs {want}");
                }
            }
        }
    }

    /// `cargo test --release -p mistralrs-core titan_hostkv::tests::bench -- --ignored --nocapture`: one layer of the
    /// dense 27B (24 query heads on 4 KV heads) at 57.6k keys, and the RAM bytes it reads.
    #[test]
    #[ignore]
    fn bench() {
        let (h, kvh, kv_len) = (24, 4, 57600);
        let kf: Vec<float8::F8E4M3> = (0..kvh * kv_len * HEAD_DIM).map(|i| float8::F8E4M3::from_bits((i % 113) as u8)).collect();
        for s in [1usize, 2] {
            let q: Vec<f32> = (0..h * s * HEAD_DIM).map(|i| (i % 7) as f32 * 0.01).collect();
            let sh = Shape { h, kvh, s, kv_len, k_hs: kv_len * HEAD_DIM, v_hs: kv_len * HEAD_DIM, scale: 0.0625 };
            let _ = POOL.install(|| run(&q, as_bytes(&kf), as_bytes(&kf), &sh));
            let t0 = std::time::Instant::now();
            for _ in 0..10 {
                let _ = POOL.install(|| run(&q, as_bytes(&kf), as_bytes(&kf), &sh));
            }
            let ms = t0.elapsed().as_secs_f64() * 100.0;
            let gb = (2 * kvh * kv_len * HEAD_DIM) as f64 / 1e9;
            println!("rows {s}: {ms:.2} ms per layer, {:.1} GB/s, {} threads", gb / ms * 1e3, POOL.current_num_threads());
        }
    }

    #[test]
    fn row_result_independent_of_call() {
        let (h, kvh, kv_len) = (4, 1, 3000);
        let mut seed = 7u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let q2: Vec<f32> = (0..h * 2 * HEAD_DIM).map(|_| rnd()).collect();
        let kf: Vec<float8::F8E4M3> = (0..kvh * kv_len * HEAD_DIM).map(|_| float8::F8E4M3::from_f32(rnd())).collect();
        let vf: Vec<float8::F8E4M3> = (0..kvh * kv_len * HEAD_DIM).map(|_| float8::F8E4M3::from_f32(rnd())).collect();
        let sh2 = Shape { h, kvh, s: 2, kv_len, k_hs: kv_len * HEAD_DIM, v_hs: kv_len * HEAD_DIM, scale: 0.0625 };
        let two = run(&q2, as_bytes(&kf), as_bytes(&vf), &sh2);
        // row 0 of the 2-row call is a 1-row call one key shorter
        let q1: Vec<f32> = (0..h).flat_map(|hd| q2[(hd * 2) * HEAD_DIM..][..HEAD_DIM].to_vec()).collect();
        let sh1 = Shape { s: 1, kv_len: kv_len - 1, ..sh2 };
        let one = run(&q1, as_bytes(&kf), as_bytes(&vf), &sh1);
        for hd in 0..h {
            assert_eq!(&two[(hd * 2) * HEAD_DIM..][..HEAD_DIM], &one[hd * HEAD_DIM..][..HEAD_DIM]);
        }
    }
}
