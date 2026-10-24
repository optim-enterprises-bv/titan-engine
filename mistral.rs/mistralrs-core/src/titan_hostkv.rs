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

/// Cached keys below which a call runs on the small pool. There a call costs latency (thread wake-up and clock
/// ramp after the ~3 ms of GPU work between two attentions, which a few pre-woken threads hide), above it memory
/// bandwidth (every core). Q4_K_M 27B decode on the 24-core 285HX, at ~0.3k / 12k / 40k context: 8 threads 41.3 /
/// 27.8 / 13.6 tok/s, every core 40.1 / 31.5 / 18.0.
const SMALL_POOL_MAX_KEYS: usize = 4096;
/// Default `TITAN_HOSTKV_SMALL_THREADS`.
const SMALL_POOL_THREADS: usize = 8;
/// Longest a pre-woken attention thread spins waiting for its call (see `prewarm`).
const PREWARM_SPIN_LIMIT: std::time::Duration = std::time::Duration::from_millis(5);

struct Pool {
    pool: rayon::ThreadPool,
    /// set by `prewarm`, cleared when the call arrives
    prewarming: std::sync::atomic::AtomicBool,
}

fn pool(env: &str, default: usize, name: &'static str) -> Pool {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let n = mistralrs_quant::titan_cfg::var(env).ok().and_then(|v| v.parse().ok()).unwrap_or(default).clamp(1, cores);
    Pool {
        pool: rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(move |i| format!("{name}-{i}"))
            .build()
            .expect("titan host-kv thread pool"),
        prewarming: std::sync::atomic::AtomicBool::new(false),
    }
}

/// `TITAN_HOSTKV_THREADS=<n>`: threads for caches of `SMALL_POOL_MAX_KEYS` keys or more (default: every core).
static POOL: LazyLock<Pool> = LazyLock::new(|| pool("TITAN_HOSTKV_THREADS", usize::MAX, "titan-hostkv"));
/// `TITAN_HOSTKV_SMALL_THREADS=<n>`: threads for shorter caches.
static SMALL_POOL: LazyLock<Pool> =
    LazyLock::new(|| pool("TITAN_HOSTKV_SMALL_THREADS", SMALL_POOL_THREADS, "titan-hostkv-small"));

fn pool_for(kv_len: usize) -> &'static Pool {
    if kv_len < SMALL_POOL_MAX_KEYS { &SMALL_POOL } else { &POOL }
}

/// Wake the threads that will attend over `kv_len` cached keys, ahead of a decode attention while the GPU still runs
/// the layer's queued work: waking them in the call itself cost ~0.4-0.6 ms per call at any context length (16 calls
/// a step). Each spins until the call arrives (`run_host`) or `PREWARM_SPIN_LIMIT` passes.
pub(crate) fn prewarm(kv_len: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    let p = pool_for(kv_len);
    if p.prewarming.swap(true, Relaxed) {
        return;
    }
    p.pool.spawn_broadcast(move |_| {
        let t0 = std::time::Instant::now();
        while p.prewarming.load(Relaxed) && t0.elapsed() < PREWARM_SPIN_LIMIT {
            std::hint::spin_loop();
        }
    });
}

/// `TITAN_HOSTKV_PROF=1`: per-phase wall time of decode host attentions, logged every `PROF_EVERY` calls.
pub(crate) fn prof_on() -> bool {
    static ON: mistralrs_quant::titan_cfg::GenCell<bool> = mistralrs_quant::titan_cfg::GenCell::new();
    *ON.get_or_init(|| mistralrs_quant::titan_cfg::var("TITAN_HOSTKV_PROF").is_ok_and(|v| v == "1"))
}

const PROF_EVERY: u64 = 2048;
pub(crate) const PROF_PHASES: [&str; 5] = ["gpu drain", "kv append", "q download", "cpu attention", "upload"];
static PROF_NS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];
static PROF_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Add `t0.elapsed()` to `phase` and restart `t0`.
pub(crate) fn prof_mark(phase: usize, t0: &mut std::time::Instant) {
    use std::sync::atomic::Ordering::Relaxed;
    PROF_NS[phase].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
    *t0 = std::time::Instant::now();
    if phase + 1 == PROF_PHASES.len() && PROF_CALLS.fetch_add(1, Relaxed) % PROF_EVERY == PROF_EVERY - 1 {
        let n = PROF_CALLS.load(Relaxed) as f64;
        let per = PROF_PHASES.iter().zip(&PROF_NS).map(|(p, ns)| format!("{p} {:.1} us", ns.load(Relaxed) as f64 / n / 1e3));
        tracing::info!("titan host kv prof ({n} calls): {}", per.collect::<Vec<_>>().join(", "));
    }
}

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
    prewarm(kv_len);
    let mut t0 = std::time::Instant::now();
    let qh = q.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.contiguous()?;
    if prof_on() {
        prof_mark(2, &mut t0);
    }
    let (qg, ql) = qh.storage_and_layout();
    let Storage::Cpu(qc) = &*qg else {
        candle_core::bail!("titan host kv: q not on the CPU");
    };
    let out = run_host(&qc.as_slice::<f32>()?[ql.start_offset()..], k, v, h, s, p.softmax_scale, pool_for(kv_len))?;
    drop(qg);
    if prof_on() {
        prof_mark(3, &mut t0);
    }
    let y = Tensor::from_vec(out, (1, h, s, HEAD_DIM), &Device::Cpu)?.to_device(q.device())?.to_dtype(q.dtype())?;
    if prof_on() {
        q.device().synchronize()?;
        prof_mark(4, &mut t0);
    }
    Ok(y)
}

/// `attend` for a batch whose sequences keep their own host caches: row `i` of `q` (b, h, s, 256) on CUDA over
/// `kvs[i]` (1, kvh, len_i, 256), with one device round trip for the batch. Each row equals its `attend` call.
pub(crate) fn attend_many(q: &Tensor, kvs: &[(Tensor, Tensor)], p: &SdpaParams) -> Result<Tensor> {
    let (b, h, s, d) = q.dims4()?;
    if b != kvs.len() || d != HEAD_DIM || s > CPU_MAX_ROWS {
        candle_core::bail!("titan host kv: unsupported batched q {:?} over {} caches", q.shape(), kvs.len());
    }
    let longest = kvs.iter().map(|(k, _)| k.dim(2)).collect::<Result<Vec<_>>>()?.into_iter().max().unwrap_or(0);
    prewarm(longest);
    let qh = q.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.contiguous()?;
    let (qg, ql) = qh.storage_and_layout();
    let Storage::Cpu(qc) = &*qg else {
        candle_core::bail!("titan host kv: q not on the CPU");
    };
    let qd = &qc.as_slice::<f32>()?[ql.start_offset()..];
    let per = h * s * HEAD_DIM;
    for (i, (k, v)) in kvs.iter().enumerate() {
        let (kb, kvh, kv_len, kd) = k.dims4()?;
        if kb != 1 || kd != HEAD_DIM || h % kvh != 0 || kv_len < s || v.dims4()? != k.dims4()? {
            candle_core::bail!("titan host kv: unsupported cache {:?} for batch row {i}", k.shape());
        }
    }
    let caches = kvs.iter().map(|(k, v)| HostCache::new(k, v)).collect::<Result<Vec<_>>>()?;
    let pool = pool_for(longest);
    pool.prewarming.store(false, std::sync::atomic::Ordering::Relaxed);
    // every cache's units in one parallel pass: one after another, a short cache left most threads idle
    let outs = pool.pool.install(|| {
        let jobs = caches
            .iter()
            .enumerate()
            .map(|(i, c)| c.job(&qd[i * per..][..per], h, s, p.softmax_scale))
            .collect::<Result<Vec<_>>>()?;
        Ok::<_, candle_core::Error>(run_jobs(&jobs))
    })?;
    drop(caches);
    drop(qg);
    Tensor::from_vec(outs.concat(), (b, h, s, HEAD_DIM), &Device::Cpu)?.to_device(q.device())?.to_dtype(q.dtype())
}

/// `run` of host queries `q` (h, s, 256) over the host F8E4M3 cache `k`, `v` (1, kvh, kv_len, 256).
fn run_host(q: &[f32], k: &Tensor, v: &Tensor, h: usize, s: usize, scale: f32, p: &Pool) -> Result<Vec<f32>> {
    let cache = HostCache::new(k, v)?;
    // release the pre-woken threads: they take this call's units within microseconds
    p.prewarming.store(false, std::sync::atomic::Ordering::Relaxed);
    p.pool.install(|| Ok(run_jobs(&[cache.job(q, h, s, scale)?]).pop().expect("one job")))
}

/// A host F8E4M3 cache `k`, `v` (1, kvh, kv_len, 256) held for reading.
struct HostCache<'a> {
    k: &'a Tensor,
    v: &'a Tensor,
    kg: std::sync::RwLockReadGuard<'a, Storage>,
    vg: std::sync::RwLockReadGuard<'a, Storage>,
}

impl<'a> HostCache<'a> {
    fn new(k: &'a Tensor, v: &'a Tensor) -> Result<Self> {
        if k.dtype() != DType::F8E4M3 || v.dtype() != DType::F8E4M3 {
            candle_core::bail!("titan host kv: K/V must be F8E4M3, got {:?}", k.dtype());
        }
        let (kg, _) = k.storage_and_layout();
        let (vg, _) = v.storage_and_layout();
        if !matches!((&*kg, &*vg), (Storage::Cpu(_), Storage::Cpu(_))) {
            candle_core::bail!("titan host kv: K / V not on the CPU");
        }
        Ok(Self { k, v, kg, vg })
    }

    /// Queries `q` (h, s, 256) over this cache, split for the calling pool.
    fn job<'q>(&'q self, q: &'q [f32], h: usize, s: usize, scale: f32) -> Result<Job<'q>> {
        let (Storage::Cpu(kc), Storage::Cpu(vc)) = (&*self.kg, &*self.vg) else {
            candle_core::bail!("titan host kv: K / V not on the CPU");
        };
        let (_, kvh, kv_len, _) = self.k.dims4()?;
        let k = as_bytes(&kc.as_slice::<float8::F8E4M3>()?[self.k.layout().start_offset()..]);
        let v = as_bytes(&vc.as_slice::<float8::F8E4M3>()?[self.v.layout().start_offset()..]);
        let sh = Shape { h, kvh, s, kv_len, k_hs: self.k.stride()[1], v_hs: self.v.stride()[1], scale };
        Ok(Job::new(q, k, v, sh))
    }
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

/// One unit's online-softmax state per query it covers: un-normalised o, then (max, sum).
struct Partial(Vec<f32>);

const STRIDE: usize = HEAD_DIM + 2;
/// Keys dequantised per block (32 KiB of f32 K, reused from L1 by every query of the group).
const BK: usize = 32;
/// Queries sharing each K/V load in the scores and P.V microkernels.
const QB: usize = 3;

/// Query split of a KV head's queries: with so few key chunks (short contexts) that most threads would idle, a
/// unit covers fewer of them. Each split dequantises the chunk again, so it only pays there. Each query runs the
/// same ops in any split.
fn query_groups(nq: usize, units: usize) -> usize {
    let threads = rayon::current_num_threads();
    if units * 2 > threads {
        return 1;
    }
    threads.div_ceil(units).clamp(1, nq.div_ceil(QB))
}

/// One cache's queries cut into units: (KV head, key chunk, query span).
struct Job<'a> {
    q: &'a [f32],
    k: &'a [u8],
    v: &'a [u8],
    sh: Shape,
    chunks: usize,
    per: usize,
    spans: Vec<(usize, usize)>,
}

impl<'a> Job<'a> {
    fn new(q: &'a [f32], k: &'a [u8], v: &'a [u8], sh: Shape) -> Self {
        let nq = sh.h / sh.kvh * sh.s;
        let chunks = sh.kv_len.div_ceil(KEY_CHUNK);
        let qg = query_groups(nq, sh.kvh * chunks);
        let per = nq.div_ceil(qg).next_multiple_of(QB).min(nq);
        let spans = (0..nq).step_by(per).map(|a| (a, (a + per).min(nq))).collect();
        Self { q, k, v, sh, chunks, per, spans }
    }

    fn units(&self) -> usize {
        self.sh.kvh * self.chunks * self.spans.len()
    }

    fn unit(&self, i: usize) -> Partial {
        let ns = self.spans.len();
        let (g, c, s) = (i / (self.chunks * ns), i / ns % self.chunks, i % ns);
        unit(self.q, self.k, self.v, &self.sh, g, c * KEY_CHUNK, self.spans[s])
    }
}

#[cfg(test)]
fn run(q: &[f32], k: &[u8], v: &[u8], sh: &Shape) -> Vec<f32> {
    run_jobs(&[Job::new(q, k, v, *sh)]).pop().expect("one job")
}

/// Every job's units in one parallel pass, then each job's chunks merged in key order.
fn run_jobs(jobs: &[Job]) -> Vec<Vec<f32>> {
    let units: Vec<(usize, usize)> = jobs.iter().enumerate().flat_map(|(j, job)| (0..job.units()).map(move |i| (j, i))).collect();
    let parts: Vec<Partial> = units.par_iter().map(|&(j, i)| jobs[j].unit(i)).collect();
    let mut at = 0;
    jobs.iter()
        .map(|job| {
            let out = merge(job, &parts[at..at + job.units()]);
            at += job.units();
            out
        })
        .collect()
}

fn merge(job: &Job, parts: &[Partial]) -> Vec<f32> {
    let (sh, chunks, per, ns) = (&job.sh, job.chunks, job.per, job.spans.len());
    let spans = &job.spans;
    let nq = sh.h / sh.kvh * sh.s;
    let mut out = vec![0f32; sh.h * sh.s * HEAD_DIM];
    out.par_chunks_mut(nq * HEAD_DIM).enumerate().for_each(|(g, out)| {
        for qi in 0..nq {
            let s = qi / per;
            let at = (qi - spans[s].0) * STRIDE;
            let ps = || (0..chunks).map(|c| &parts[(g * chunks + c) * ns + s].0);
            let m = ps().map(|p| p[at + HEAD_DIM]).fold(f32::NEG_INFINITY, f32::max);
            let o = &mut out[qi * HEAD_DIM..][..HEAD_DIM];
            let mut l = 0f32;
            for p in ps() {
                let pm = p[at + HEAD_DIM];
                if pm == f32::NEG_INFINITY {
                    continue;
                }
                let w = (pm - m).exp();
                l += w * p[at + HEAD_DIM + 1];
                for (o, a) in o.iter_mut().zip(&p[at..at + HEAD_DIM]) {
                    *o += w * a;
                }
            }
            let inv = 1.0 / l;
            o.iter_mut().for_each(|x| *x *= inv);
        }
    });
    out
}

/// Keys `a .. a + KEY_CHUNK` of KV head `g` for queries `qa .. qb` of the group (query `qi` = head `qi / s` of the
/// group, row `qi % s`), online softmax over blocks of `BK` keys. Row `r` sees keys `< kv_len - s + r + 1`; only
/// those enter its block, so its operations do not depend on the other rows.
fn unit(q: &[f32], k: &[u8], v: &[u8], sh: &Shape, g: usize, a: usize, (qa, qb): (usize, usize)) -> Partial {
    let n_rep = sh.h / sh.kvh;
    let nq = qb - qa;
    let past = sh.kv_len - sh.s;
    let b = (a + KEY_CHUNK).min(sh.kv_len);
    let mut part = vec![0f32; nq * STRIDE];
    let mut m = vec![f32::NEG_INFINITY; nq];
    let mut l = vec![0f32; nq];
    let mut kb = vec![0f32; BK * HEAD_DIM];
    let mut vb = vec![0f32; BK * HEAD_DIM];
    let mut sc = vec![0f32; nq * BK];
    let mut alpha = vec![0f32; nq];
    let mut vis = vec![0usize; nq];
    let qrow = |i: usize| {
        let qi = qa + i;
        &q[((g * n_rep + qi / sh.s) * sh.s + qi % sh.s) * HEAD_DIM..][..HEAD_DIM]
    };
    let mut j0 = a;
    while j0 < b {
        let n = BK.min(b - j0);
        for j in 0..n {
            dequant(&k[g * sh.k_hs + (j0 + j) * HEAD_DIM..][..HEAD_DIM], &mut kb[j * HEAD_DIM..][..HEAD_DIM]);
            dequant(&v[g * sh.v_hs + (j0 + j) * HEAD_DIM..][..HEAD_DIM], &mut vb[j * HEAD_DIM..][..HEAD_DIM]);
        }
        for i in 0..nq {
            vis[i] = (past + (qa + i) % sh.s + 1).saturating_sub(j0).min(n);
        }
        for i0 in (0..nq).step_by(QB) {
            let t = QB.min(nq - i0);
            let n_any = vis[i0..i0 + t].iter().copied().max().unwrap_or(0);
            let qs: [&[f32]; QB] = std::array::from_fn(|r| qrow(i0 + r.min(t - 1)));
            scores(&qs[..t], &kb, n_any, sh.scale, &mut sc[i0 * BK..][..t * BK]);
        }
        for i in 0..nq {
            let vis = vis[i];
            if vis == 0 {
                alpha[i] = 1.0;
                continue;
            }
            let p = &mut sc[i * BK..][..vis];
            let bm = p.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mn = m[i].max(bm);
            alpha[i] = if m[i] == f32::NEG_INFINITY { 0.0 } else { (m[i] - mn).exp() };
            let ps = exp_sub_sum(p, mn);
            l[i] = l[i] * alpha[i] + ps;
            m[i] = mn;
        }
        for i0 in (0..nq).step_by(QB) {
            let t = QB.min(nq - i0);
            if vis[i0..i0 + t].iter().any(|&n| n > 0) {
                let acc = &mut part[i0 * STRIDE..][..t * STRIDE];
                pv(acc, &alpha[i0..i0 + t], &vis[i0..i0 + t], &sc[i0 * BK..][..t * BK], &vb);
            }
        }
        j0 += n;
    }
    for i in 0..nq {
        part[i * STRIDE + HEAD_DIM] = m[i];
        part[i * STRIDE + HEAD_DIM + 1] = l[i];
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

/// `out[r * BK + j] = scale * qs[r] . kb[j]` for the first `n` rows of the block `kb`.
fn scores(qs: &[&[f32]], kb: &[f32], n: usize, scale: f32, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if simd() {
        let qp: [*const f32; QB] = std::array::from_fn(|r| qs[r.min(qs.len() - 1)].as_ptr());
        unsafe { avx::scores(qp, qs.len(), kb.as_ptr(), n, scale, out.as_mut_ptr()) };
        return;
    }
    for (r, q) in qs.iter().enumerate() {
        for j in 0..n {
            out[r * BK + j] = scale * q.iter().zip(&kb[j * HEAD_DIM..][..HEAD_DIM]).map(|(a, b)| a * b).sum::<f32>();
        }
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

/// Per query `r`: `acc_r = alpha[r] * acc_r + sum_{j < vis[r]} p_r[j] * vb[j]` (accumulators `STRIDE` apart,
/// weights `BK` apart).
fn pv(acc: &mut [f32], alpha: &[f32], vis: &[usize], p: &[f32], vb: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    if simd() {
        unsafe { avx::pv(acc.as_mut_ptr(), alpha, vis, p.as_ptr(), vb.as_ptr()) };
        return;
    }
    for (r, (&al, &n)) in alpha.iter().zip(vis).enumerate() {
        for d in 0..HEAD_DIM {
            let a = &mut acc[r * STRIDE + d];
            let mut x = *a * al;
            for j in 0..n {
                x += p[r * BK + j] * vb[j * HEAD_DIM + d];
            }
            *a = x;
        }
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

    use super::{BK, E4M3_AS_F16_SCALE, HEAD_DIM, QB, STRIDE};

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

    /// `t` (<= QB) queries times four keys per pass: each k load feeds `t` FMAs and each q load four, one
    /// horizontal sum per (query, key).
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn scores(q: [*const f32; QB], t: usize, kb: *const f32, n: usize, scale: f32, out: *mut f32) {
        let mut j = 0;
        while j + 4 <= n {
            let k0 = kb.add(j * HEAD_DIM);
            let mut a = [[_mm256_setzero_ps(); 4]; QB];
            for d in (0..HEAD_DIM).step_by(8) {
                let kv: [__m256; 4] = std::array::from_fn(|u| _mm256_loadu_ps(k0.add(u * HEAD_DIM + d)));
                for (r, a) in a.iter_mut().enumerate().take(t) {
                    let qv = _mm256_loadu_ps(q[r].add(d));
                    for (a, kv) in a.iter_mut().zip(&kv) {
                        *a = _mm256_fmadd_ps(qv, *kv, *a);
                    }
                }
            }
            for (r, a) in a.iter().enumerate().take(t) {
                for (u, a) in a.iter().enumerate() {
                    *out.add(r * BK + j + u) = scale * hsum(*a);
                }
            }
            j += 4;
        }
        while j < n {
            let k0 = kb.add(j * HEAD_DIM);
            for (r, q) in q.iter().enumerate().take(t) {
                let mut a = _mm256_setzero_ps();
                for d in (0..HEAD_DIM).step_by(8) {
                    a = _mm256_fmadd_ps(_mm256_loadu_ps(q.add(d)), _mm256_loadu_ps(k0.add(d)), a);
                }
                *out.add(r * BK + j) = scale * hsum(a);
            }
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

    /// `super::pv` 16 dims at a time held in registers over the block's keys: each v load feeds every query
    /// still within its `vis` keys.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn pv(acc: *mut f32, alpha: &[f32], vis: &[usize], p: *const f32, vb: *const f32) {
        let t = alpha.len();
        let n = vis.iter().copied().max().unwrap_or(0);
        for d in (0..HEAD_DIM).step_by(16) {
            let mut r = [[_mm256_setzero_ps(); 2]; QB];
            for (q, r) in r.iter_mut().enumerate().take(t) {
                let al = _mm256_set1_ps(alpha[q]);
                for (u, r) in r.iter_mut().enumerate() {
                    *r = _mm256_mul_ps(al, _mm256_loadu_ps(acc.add(q * STRIDE + d + 8 * u)));
                }
            }
            for j in 0..n {
                let row = vb.add(j * HEAD_DIM + d);
                let v = [_mm256_loadu_ps(row), _mm256_loadu_ps(row.add(8))];
                for (q, r) in r.iter_mut().enumerate().take(t) {
                    if j < vis[q] {
                        let w = _mm256_set1_ps(*p.add(q * BK + j));
                        for (r, v) in r.iter_mut().zip(&v) {
                            *r = _mm256_fmadd_ps(w, *v, *r);
                        }
                    }
                }
            }
            for (q, r) in r.iter().enumerate().take(t) {
                for (u, r) in r.iter().enumerate() {
                    _mm256_storeu_ps(acc.add(q * STRIDE + d + 8 * u), *r);
                }
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
        let (h, kvh) = (24, 4);
        let lens: Vec<usize> = std::env::var("BENCH_LENS").ok().map_or(vec![57600], |v| v.split(',').map(|x| x.parse().unwrap()).collect());
        for kv_len in lens {
        let kf: Vec<float8::F8E4M3> = (0..kvh * kv_len * HEAD_DIM).map(|i| float8::F8E4M3::from_bits((i % 113) as u8)).collect();
        for s in [1usize, 2] {
            let q: Vec<f32> = (0..h * s * HEAD_DIM).map(|i| (i % 7) as f32 * 0.01).collect();
            let sh = Shape { h, kvh, s, kv_len, k_hs: kv_len * HEAD_DIM, v_hs: kv_len * HEAD_DIM, scale: 0.0625 };
            let _ = POOL.pool.install(|| run(&q, as_bytes(&kf), as_bytes(&kf), &sh));
            let t0 = std::time::Instant::now();
            for _ in 0..50 {
                let _ = POOL.pool.install(|| run(&q, as_bytes(&kf), as_bytes(&kf), &sh));
            }
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / 50.0;
            let gb = (2 * kvh * kv_len * HEAD_DIM) as f64 / 1e9;
            println!("keys {kv_len} rows {s}: {ms:.3} ms per layer, {:.1} GB/s, {} threads", gb / ms * 1e3, POOL.pool.current_num_threads());
        }
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
    #[test]
    #[ignore]
    fn unit_bench() {
        let (h, kvh, kv_len) = (24, 4, 4096);
        let kf: Vec<float8::F8E4M3> = (0..kvh * kv_len * HEAD_DIM).map(|i| float8::F8E4M3::from_bits((i % 113) as u8)).collect();
        for s in [1usize, 2] {
            let q: Vec<f32> = (0..h * s * HEAD_DIM).map(|i| (i % 7) as f32 * 0.01).collect();
            let sh = Shape { h, kvh, s, kv_len, k_hs: kv_len * HEAD_DIM, v_hs: kv_len * HEAD_DIM, scale: 0.0625 };
            let nq = h / kvh * s;
            let _ = unit(&q, as_bytes(&kf), as_bytes(&kf), &sh, 0, 0, (0, nq));
            let t0 = std::time::Instant::now();
            for _ in 0..200 {
                std::hint::black_box(unit(&q, as_bytes(&kf), as_bytes(&kf), &sh, 0, 0, (0, nq)));
            }
            let us = t0.elapsed().as_secs_f64() * 1e6 / 200.0;
            let fma = (KEY_CHUNK * nq * HEAD_DIM * 2) as f64;
            println!("unit rows {s}: {us:.1} us, {:.1} GFMA/s one thread", fma / us / 1e3);
        }
    }

    #[test]
    fn batched_jobs_match_each_alone() {
        let (h, kvh) = (6, 2);
        let mut seed = 11u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let lens = [300usize, 5000];
        let data: Vec<(Vec<f32>, Vec<float8::F8E4M3>, Vec<float8::F8E4M3>)> = lens
            .iter()
            .map(|&n| {
                let q = (0..h * 2 * HEAD_DIM).map(|_| rnd()).collect();
                let k = (0..kvh * n * HEAD_DIM).map(|_| float8::F8E4M3::from_f32(4.0 * rnd())).collect();
                let v = (0..kvh * n * HEAD_DIM).map(|_| float8::F8E4M3::from_f32(4.0 * rnd())).collect();
                (q, k, v)
            })
            .collect();
        let shape = |n: usize| Shape { h, kvh, s: 2, kv_len: n, k_hs: n * HEAD_DIM, v_hs: n * HEAD_DIM, scale: 0.0625 };
        for alone_pool in [&*SMALL_POOL, &*POOL] {
            for batch_pool in [&*SMALL_POOL, &*POOL] {
                let alone: Vec<Vec<f32>> = data
                    .iter()
                    .zip(lens)
                    .map(|((q, k, v), n)| alone_pool.pool.install(|| run(q, as_bytes(k), as_bytes(v), &shape(n))))
                    .collect();
                let batched = batch_pool.pool.install(|| {
                    let jobs: Vec<Job> =
                        data.iter().zip(lens).map(|((q, k, v), n)| Job::new(q, as_bytes(k), as_bytes(v), shape(n))).collect();
                    run_jobs(&jobs)
                });
                assert_eq!(alone, batched);
            }
        }
    }
}
