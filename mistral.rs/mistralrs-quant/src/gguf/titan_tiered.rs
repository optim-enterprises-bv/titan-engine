//! titan-engine: GGUF MoE experts split between GPU slots and host RAM.
//! GPU experts run cuda-oxide kernels bit-identical to candle's `indexed_moe_forward`, the rest run the
//! bit-identical CPU twin, so any split produces the stock output exactly.
//! `TITAN_TIERED=1` enables it; `TITAN_TIERED_GPU_FRACTION` (default 1) sets the share of experts resident.
//! Placement: `TITAN_TIERED_PROFILE=<file>` (lines `layer expert count`, from m3/profile.py) seeds each
//! layer's slots with its hottest experts, else the lowest ids. `TITAN_TIERED_POLICY=lru` then admits
//! every decode-step miss into the least recently used slot (the CPU twin serves the miss itself, the
//! upload serves the next use); `static` (default) never moves an expert. `lfu` keeps a decayed routing
//! count per expert (`TITAN_TIERED_LFU_DECAY` per decode row, default 0.98) and admits a miss only if its
//! count beats the coldest resident's by `TITAN_TIERED_LFU_MARGIN` (default 4). `TITAN_TIERED_ASYNC=1`
//! makes `lfu` uploads asynchronous (see `titan_upload`). None of these changes the output.
//! `TITAN_TIERED_MMAP=1`: the host side is the GGUF's read-only mmap (page cache backed by the file)
//! instead of owned copies, so experts that do not fit in RAM are paged in from disk on a miss.
//! `TITAN_TIERED_LOOKAHEAD=N`: the model predicts the next N layers' experts (their routers on this layer's
//! residual) and a background thread pages their host bytes in ahead of use. `TITAN_TIERED_FINGERPRINT=M`:
//! at the first decode step after a prefill, the M coldest slots of each layer take the prefill's hottest
//! non-resident experts. Placement and prefetch never change the output.

use candle_core::{
    cuda::{
        cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg},
        CudaDevice, CudaStorage, WrapErr,
    },
    quantized::{GgmlDType, QTensor},
    DType, Result, Shape, Storage, Tensor,
};
use memmap2::{Advice, Mmap};
use rayon::prelude::*;
use std::sync::Arc;

use super::titan_cpu::{self, Format};
use super::titan_pfs;
use super::titan_upload;
use crate::utils::slice_ptr;

/// Prefill streaming of the non-resident experts (`titan_pfs`).
#[path = "titan_tiered_pfs.rs"]
mod pfs;
/// One CPU pass per missed expert (`TITAN_CPU_ONEPASS`).
#[path = "titan_onepass.rs"]
mod onepass;
/// Mapped-memory doorbell for the CPU misses (`TITAN_DOORBELL`).
#[path = "titan_doorbell.rs"]
mod doorbell;

const PTX: &str = include_str!("titan_kernels_oxide.ptx");
const MODULE: &str = "titan_kernels_oxide";
/// oxide-kernels/mmvq-moe: llama.cpp's `mul_mat_vec_q_moe` and b1 MUL_MAT_ID grid over the slot map.
const MOE_PTX: &str = include_str!("mmvq_moe_oxide.ptx");
const MOE_MODULE: &str = "mmvq_moe_oxide";
/// Largest batch (tokens) of the mmvq-moe kernels: one warp per token.
const MMVQ_MOE_MAX_BATCH: usize = 8;
const MATRIX_ROW_PADDING: usize = 512;
const Q8_1_BLOCK_BYTES: usize = 36;
/// Slot-map value for an expert not resident on the GPU; the kernels skip it.
pub const NOT_RESIDENT: u32 = u32::MAX;
const ROWS_PER_CPU_CHUNK: usize = 32;
/// Largest k the one-row-per-warp Q4_K / Q5_K expert kernels take (their reference uses warp 0 only).
const WARP_ROWS_MAX_K: usize = 512;
/// Not in every libc release; Linux 5.14+.
const MADV_POPULATE_READ: libc::c_int = 22;
const MADV_PAGEOUT: libc::c_int = 21;
const PAGE_BYTES: usize = 4096;
const HUGE_PAGE_BYTES: usize = 2 << 20;
/// Miss-pool workers under the doorbell (`TITAN_TIERED_THREADS` overrides).
const DOORBELL_WORKERS: usize = 14;
/// Lookahead jobs queued beyond this are dropped rather than block the decode thread.
const PREFETCH_QUEUE: usize = 256;
const MAX_LOOKAHEAD: usize = 4;
/// Largest batch counted as a decode step (MTP verification rows included) for `lfu` counts and admission.
const ADMIT_MAX_ROWS: usize = 8;

/// Rows per CPU work item (`TITAN_TIERED_CHUNK`, default ROWS_PER_CPU_CHUNK).
fn rows_per_chunk() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("TITAN_TIERED_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(ROWS_PER_CPU_CHUNK))
}

/// `TITAN_MMVQ_MOE`: 1 (default) runs decode (b = 1) and MTP-verify batches (b = 2..8) on the mmvq-moe
/// kernels: llama.cpp's `mul_mat_vec_q_moe` grid, except b = 1 with a large k (llama.cpp's b1 MUL_MAT_ID
/// grid); 2 uses the MoE grid at every b; 3 the b1 grid (small_k included) at b = 1; 0 keeps the
/// one-block-per-(row, task) kernels. The output bits are the same in every mode.
fn mmvq_moe_mode() -> u32 {
    static M: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *M.get_or_init(|| std::env::var("TITAN_MMVQ_MOE").ok().and_then(|v| v.parse().ok()).unwrap_or(1))
}

/// The pool that computes CPU misses: spinning workers (see `titan_spin`), or rayon's global pool
/// with `TITAN_TIERED_SPIN=0`.
fn miss_pool() -> Option<&'static super::titan_spin::SpinPool> {
    static POOL: std::sync::OnceLock<Option<super::titan_spin::SpinPool>> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        if std::env::var("TITAN_TIERED_SPIN").is_ok_and(|v| v == "0") {
            return None;
        }
        // the calling thread works too; with the doorbell it is not the decode thread, and the memory-bound pass
        // measured faster on fewer spinning workers (m4/idle: 22 -> 14 workers, MTP=2 decode +5%)
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let default = if doorbell::enabled() { DOORBELL_WORKERS.min(threads.saturating_sub(2)) } else { threads.saturating_sub(1) };
        let workers = env_usize("TITAN_TIERED_THREADS", default);
        Some(super::titan_spin::SpinPool::new(workers))
    })
    .as_ref()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Policy {
    Static,
    Lru,
    Lfu,
}

/// `TITAN_TIERED_LFU_DECAY` / `TITAN_TIERED_LFU_MARGIN`.
fn lfu_params() -> (f32, f32) {
    static P: std::sync::OnceLock<(f32, f32)> = std::sync::OnceLock::new();
    *P.get_or_init(|| {
        let f = |name: &str, default: f32| std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default);
        (f("TITAN_TIERED_LFU_DECAY", 0.98), f("TITAN_TIERED_LFU_MARGIN", 4.0))
    })
}

fn async_uploads() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| env_flag("TITAN_TIERED_ASYNC"))
}

/// GPU residency: which expert sits in which slot, and when each slot was last used.
struct Cache {
    slots: CudaSlice<u8>,
    slot_map: CudaSlice<u32>,
    /// expert -> slot or NOT_RESIDENT (host mirror of `slot_map`)
    map: Vec<u32>,
    /// slot -> expert
    owner: Vec<u32>,
    last_use: Vec<u64>,
    tick: u64,
    /// Fingerprint: routing counts of the current request's prefill, and whether decode has started.
    prefill_hist: Vec<u32>,
    decoding: bool,
    /// `lfu`: decayed routing count per expert.
    lfu: Vec<f32>,
    /// Async admissions in flight: (slot, expert, copy completion). Their slots have owner NOT_RESIDENT.
    pending: Vec<(usize, u32, Arc<titan_upload::Done>)>,
    /// Device address of `slots`.
    slots_ptr: u64,
}

static AUTO_FRACTION: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
/// Bytes the auto plan must leave free on top of the reserve (the KV cache of the requested context).
static AUTO_EXTRA_RESERVE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

static STATS: [std::sync::atomic::AtomicU64; 7] = [const { std::sync::atomic::AtomicU64::new(0) }; 7];
/// Lookahead predictor scores per depth: [predicted & routed, predicted, routed] over all experts, then the
/// same three over non-resident experts only.
static PRED_STATS: [[std::sync::atomic::AtomicU64; 6]; MAX_LOOKAHEAD] =
    [const { [const { std::sync::atomic::AtomicU64::new(0) }; 6] }; MAX_LOOKAHEAD];
/// (layer, depth) -> predicted ids waiting for that layer's decode step.
static PENDING: std::sync::Mutex<Vec<(usize, usize, Vec<u32>)>> = std::sync::Mutex::new(Vec::new());
/// Prefetch worker counters: [jobs, ranges, bytes, busy ns, dropped jobs].
static PF_STATS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];
const HITS: usize = 0;
const MISSES: usize = 1;
const UPLOADS: usize = 2;
const CALLS: usize = 3;
/// Hits and misses of 2..=ADMIT_MAX_ROWS-row forwards (MTP verification), and async uploads landed.
const VHITS: usize = 4;
const VMISSES: usize = 5;
const LANDED: usize = 6;
/// TITAN_TIERED_TIMING=1: nanoseconds in the miss path, [ids fetch, xq fetch, cpu compute, row upload, calls]
static TIMING: [std::sync::atomic::AtomicU64; 12] = [const { std::sync::atomic::AtomicU64::new(0) }; 12];
/// TIMING slots 5..8: CPU passes, tasks and expert groups they computed.
const PASSES: usize = 5;
/// TIMING slot 8: GPU expert gemv (TITAN_TIERED_TIMING=2 syncs after each launch); 9/10: major/minor faults
/// taken during CPU miss passes.
const GPU_GEMV: usize = 8;
const MAJFLT: usize = 9;
const MINFLT: usize = 10;
/// TITAN_TIERED_TIMING: 1 = wall time of the miss path, 2 = also sync after each GPU gemv (serializes).
fn timing_level() -> u32 {
    static L: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *L.get_or_init(|| std::env::var("TITAN_TIERED_TIMING").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
}
fn timing_on() -> bool {
    timing_level() > 0
}

fn faults() -> (u64, u64) {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    (ru.ru_majflt as u64, ru.ru_minflt as u64)
}

/// Process fault and storage-read counters, for the hit-rate log line.
fn proc_counters() -> String {
    let (maj, min) = faults();
    let read = std::fs::read_to_string("/proc/self/io")
        .ok()
        .and_then(|t| t.lines().find_map(|l| l.strip_prefix("read_bytes: ").and_then(|v| v.trim().parse::<u64>().ok())))
        .unwrap_or(0);
    format!("majflt {maj}, minflt {min}, storage read {} MiB", read >> 20)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// `TITAN_TIERED_WILLNEED=0` drops the within-layer MADV_WILLNEED hint on mapped misses.
fn willneed_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TITAN_TIERED_WILLNEED").map_or(true, |v| v != "0"))
}

/// `TITAN_TIERED_FINGERPRINT=M`: slots per tensor re-seeded from each request's prefill (0 = off).
fn fingerprint_slots() -> usize {
    static M: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *M.get_or_init(|| env_usize("TITAN_TIERED_FINGERPRINT", 0))
}

/// Byte ranges of one mapped file to page in, off the decode thread.
struct PrefetchJob {
    map: Arc<Mmap>,
    ranges: Vec<(usize, usize)>,
}

/// WILLNEED starts exact-range reads (no read-around amplification); `TITAN_TIERED_LOOKAHEAD_POPULATE=1`
/// then also maps the pages (MADV_POPULATE_READ) so the CPU twin takes no faults on them.
fn run_prefetch(job: &PrefetchJob, populate: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    let t0 = std::time::Instant::now();
    let base = job.map.as_ptr() as usize;
    let advise = |off: usize, len: usize, advice: libc::c_int| {
        let a = base + off;
        let a0 = a & !(PAGE_BYTES - 1);
        unsafe { libc::madvise(a0 as *mut libc::c_void, len + (a - a0), advice) };
    };
    for &(off, len) in &job.ranges {
        advise(off, len, libc::MADV_WILLNEED);
    }
    if populate {
        for &(off, len) in &job.ranges {
            advise(off, len, MADV_POPULATE_READ);
        }
    }
    PF_STATS[0].fetch_add(1, Relaxed);
    PF_STATS[1].fetch_add(job.ranges.len() as u64, Relaxed);
    PF_STATS[2].fetch_add(job.ranges.iter().map(|r| r.1 as u64).sum(), Relaxed);
    PF_STATS[3].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
}

/// `TITAN_TIERED_LOOKAHEAD_THREADS` (default 2) workers draining a bounded queue.
fn prefetcher() -> &'static std::sync::mpsc::SyncSender<PrefetchJob> {
    static TX: std::sync::OnceLock<std::sync::mpsc::SyncSender<PrefetchJob>> = std::sync::OnceLock::new();
    TX.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::sync_channel::<PrefetchJob>(PREFETCH_QUEUE);
        let rx = Arc::new(std::sync::Mutex::new(rx));
        let populate = env_flag("TITAN_TIERED_LOOKAHEAD_POPULATE");
        for i in 0..env_usize("TITAN_TIERED_LOOKAHEAD_THREADS", 2).max(1) {
            let rx = rx.clone();
            let _ = std::thread::Builder::new().name(format!("titan-prefetch-{i}")).spawn(move || loop {
                let job = match rx.lock().unwrap().recv() {
                    Ok(j) => j,
                    Err(_) => return,
                };
                run_prefetch(&job, populate);
            });
        }
        tx
    })
}

fn pred_report() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let pct = |a: u64, b: u64| 100.0 * a as f64 / b.max(1) as f64;
    let mut parts = Vec::new();
    for (d, s) in PRED_STATS.iter().enumerate() {
        let v: Vec<u64> = s.iter().map(|x| x.load(Relaxed)).collect();
        if v[1] > 0 {
            parts.push(format!(
                "L+{}: precision {:.1}% recall {:.1}% (non-resident: precision {:.1}% recall {:.1}%, {} routed)",
                d + 1,
                pct(v[0], v[1]),
                pct(v[0], v[2]),
                pct(v[3], v[4]),
                pct(v[3], v[5]),
                v[5]
            ));
        }
    }
    if parts.is_empty() {
        return String::new();
    }
    let pf: Vec<u64> = PF_STATS.iter().map(|x| x.load(Relaxed)).collect();
    format!(
        "; lookahead {}; prefetch {} jobs, {} ranges, {} MiB, worker busy {:.0} ms, {} dropped",
        parts.join(", "),
        pf[0],
        pf[1],
        pf[2] >> 20,
        pf[3] as f64 / 1e6,
        pf[4]
    )
}
/// [decode hits, decode misses, verify-row hits, verify-row misses, expert uploads], for the monitor page.
pub(crate) fn monitor_hits() -> [u64; 5] {
    [HITS, MISSES, VHITS, VMISSES, UPLOADS].map(|i| STATS[i].load(std::sync::atomic::Ordering::Relaxed))
}

fn add_time(slot: usize, since: std::time::Instant) {
    if timing_on() {
        TIMING[slot].fetch_add(since.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    }
}

/// A q8_1-quantized expert input on the GPU, optionally with its host copy.
pub struct QInput {
    xq: CudaSlice<u8>,
    host: Option<Vec<u8>>,
    batch: usize,
    input_dim1: usize,
    k: usize,
    q8_row_bytes: usize,
}

impl QInput {
    /// Copy the quantized input to the host once, for every projection's CPU misses.
    pub fn fetch(&mut self, dev: &CudaDevice) -> Result<()> {
        if self.host.is_none() {
            let t0 = std::time::Instant::now();
            self.host = Some(dev.clone_dtoh(&self.xq)?);
            add_time(1, t0);
        }
        Ok(())
    }
}

/// Where the CPU twin and LRU uploads read expert weights from.
enum HostStore {
    /// Copies: every expert under `lru`, only the non-resident ones under `static`.
    Owned(Vec<u8>),
    /// The stacked tensor inside the GGUF mmap, at byte `offset`; every expert, never copied.
    Mapped { map: Arc<Mmap>, offset: usize },
}

/// Stacked `[experts, n, k]` quantized weights: all in host RAM, a subset in GPU slots.
pub struct TieredExperts {
    /// Prefill streaming of the non-resident experts (`titan_pfs`); first, so it drops before `host`.
    pfs: Option<Arc<titan_pfs::Source>>,
    cache: std::sync::Mutex<Cache>,
    policy: Policy,
    host: HostStore,
    /// expert -> index of its copy in `host`, or NOT_RESIDENT when it has none
    host_index: Vec<u32>,
    expert_bytes: usize,
    dtype: GgmlDType,
    num_experts: usize,
    n: usize,
    k: usize,
    layer: usize,
    proj: &'static str,
    dev: CudaDevice,
    /// Placement order (hottest first): the first `num_resident - flex` are always resident, the next
    /// `flex` slots are re-seeded per request (fingerprint).
    order: Vec<usize>,
    flex: usize,
    /// `onepass_ok` of this layer when this is its gate projection, decided at the first forward.
    fast_ok: std::sync::OnceLock<bool>,
}

static TRACE: std::sync::OnceLock<Option<std::sync::Mutex<std::fs::File>>> = std::sync::OnceLock::new();

/// `TITAN_TIERED_TRACE=<file>`: one line per forward, `layer batch topk id id ...`.
fn trace_path() -> Option<&'static std::sync::Mutex<std::fs::File>> {
    TRACE
        .get_or_init(|| {
            let path = std::env::var("TITAN_TIERED_TRACE").ok()?;
            std::fs::OpenOptions::new().create(true).append(true).open(path).ok().map(std::sync::Mutex::new)
        })
        .as_ref()
}

fn trace_ids(f: &std::sync::Mutex<std::fs::File>, layer: usize, batch: usize, topk: usize, ids: &[u32]) {
    use std::io::Write;
    let mut line = format!("{layer} {batch} {topk}");
    for id in ids {
        line.push(' ');
        line.push_str(&id.to_string());
    }
    line.push('\n');
    if let Ok(mut f) = f.lock() {
        let _ = f.write_all(line.as_bytes());
    }
}

fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "1")
}

impl TieredExperts {
    pub fn enabled() -> bool {
        env_flag("TITAN_TIERED")
    }

    /// `TITAN_TIERED_MMAP=1`: host experts stay in the GGUF mmap (see `from_mapped`).
    pub fn mmap_enabled() -> bool {
        Self::enabled() && env_flag("TITAN_TIERED_MMAP")
    }

    pub fn supports(dtype: GgmlDType) -> bool {
        matches!(
            dtype,
            GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
                | GgmlDType::Q1_0
                | GgmlDType::IQ4NL
                | GgmlDType::MXFP4
                | GgmlDType::NVFP4
        )
    }

    fn policy() -> Policy {
        match std::env::var("TITAN_TIERED_POLICY").as_deref() {
            Ok("lru") => Policy::Lru,
            Ok("lfu") => Policy::Lfu,
            _ => Policy::Static,
        }
    }

    /// Experts of `layer` ordered hottest first, from `TITAN_TIERED_PROFILE` (`layer expert count` lines).
    fn profile_order(layer: usize, num_experts: usize) -> Option<Vec<usize>> {
        let path = std::env::var("TITAN_TIERED_PROFILE").ok()?;
        let text = std::fs::read_to_string(&path).ok()?;
        let mut counts = vec![0u64; num_experts];
        for line in text.lines() {
            let f: Vec<u64> = line.split_whitespace().filter_map(|v| v.parse().ok()).collect();
            if f.len() == 3 && f[0] as usize == layer && (f[1] as usize) < num_experts {
                counts[f[1] as usize] += f[2];
            }
        }
        let mut order: Vec<usize> = (0..num_experts).collect();
        order.sort_by(|&a, &b| counts[b].cmp(&counts[a]).then(a.cmp(&b)));
        Some(order)
    }

    /// Share of a stacked expert tensor of `dtype` that will live on the GPU (for device-map sizing).
    pub fn gpu_share(dtype: GgmlDType) -> f64 {
        if Self::enabled() && Self::supports(dtype) {
            Self::gpu_fraction()
        } else {
            1.0
        }
    }

    fn gpu_fraction() -> f64 {
        match std::env::var("TITAN_TIERED_GPU_FRACTION").as_deref() {
            Ok("auto") => *AUTO_FRACTION.get().unwrap_or(&1.0),
            Ok(v) => v.parse::<f64>().map_or(1.0, |f| f.clamp(0.0, 1.0)),
            Err(_) => 1.0,
        }
    }

    /// Whether `TITAN_TIERED_GPU_FRACTION=auto` still needs a plan (see `plan_auto`).
    pub fn wants_auto_plan() -> bool {
        Self::enabled()
            && std::env::var("TITAN_TIERED_GPU_FRACTION").as_deref() == Ok("auto")
            && AUTO_FRACTION.get().is_none()
    }

    /// `TITAN_TIERED_GPU_FRACTION=auto`: give tiered experts whatever VRAM is left after everything
    /// else the device must hold and a reserve for KV cache, activations and workspaces
    /// (`TITAN_TIERED_RESERVE_MIB`, default 1024: 0.68 ran at 4096 context with ~770 MiB spare on a 16 GB card). Decided once, before any weight is loaded, so every
    /// layer gets the same share and device-map sizing sees the same number.
    /// KV-cache bytes (from the context length) that `plan_auto` keeps free in addition to the reserve.
    pub fn set_auto_extra_reserve(bytes: usize) {
        AUTO_EXTRA_RESERVE.store(bytes, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn plan_auto(tiered_bytes: usize, other_bytes: usize, free_bytes: usize) -> f64 {
        let reserve = std::env::var("TITAN_TIERED_RESERVE_MIB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1024)
            << 20;
        let kv = AUTO_EXTRA_RESERVE.load(std::sync::atomic::Ordering::Relaxed);
        let reserve = reserve + kv;
        let budget = free_bytes.saturating_sub(other_bytes).saturating_sub(reserve);
        let fraction = if tiered_bytes == 0 { 1.0 } else { (budget as f64 / tiered_bytes as f64).clamp(0.0, 1.0) };
        // Round down to a whole percent so repeated runs pick the same slot counts.
        let fraction = (fraction * 100.0).floor() / 100.0;
        let fraction = *AUTO_FRACTION.get_or_init(|| fraction);
        tracing::info!(
            "titan tiered auto: {} MiB free, {} MiB other weights, {} MiB reserve, {} MiB tierable experts -> GPU fraction {fraction:.2}",
            free_bytes >> 20,
            other_bytes >> 20,
            reserve >> 20,
            tiered_bytes >> 20
        );
        fraction
    }

    /// Host experts copied out of a host `QTensor`.
    pub fn from_qtensor(q: &QTensor, dev: &CudaDevice, layer: usize, proj: &'static str) -> Result<Self> {
        let data = q.data()?;
        Self::build(&data, None, q.shape().dims3()?, q.dtype(), dev, layer, proj)
    }

    /// Host experts read in place from `map` (a GGUF mmap), the stacked tensor starting at byte
    /// `offset`: nothing is copied to owned memory; only the GPU slots are uploaded from it.
    pub fn from_mapped(
        map: Arc<Mmap>,
        offset: usize,
        shape: (usize, usize, usize),
        dtype: GgmlDType,
        dev: &CudaDevice,
        layer: usize,
        proj: &'static str,
    ) -> Result<Self> {
        let (num_experts, n, k) = shape;
        let len = num_experts * n * k / dtype.block_size() * dtype.type_size();
        if offset + len > map.len() {
            candle_core::bail!("titan tiered experts: mapped tensor {offset}+{len} past the file end {}", map.len());
        }
        let data = &map[offset..offset + len];
        Self::build(data, Some((map.clone(), offset)), shape, dtype, dev, layer, proj)
    }

    /// Residents are the profile's hottest experts, else the lowest ids.
    fn build(
        data: &[u8],
        mapped: Option<(Arc<Mmap>, usize)>,
        (num_experts, n, k): (usize, usize, usize),
        dtype: GgmlDType,
        dev: &CudaDevice,
        layer: usize,
        proj: &'static str,
    ) -> Result<Self> {
        if !Self::supports(dtype) {
            candle_core::bail!("titan tiered experts: unsupported dtype {dtype:?}");
        }
        let expert_bytes = n * k / dtype.block_size() * dtype.type_size();
        if data.len() < num_experts * expert_bytes {
            candle_core::bail!("titan tiered experts: {} bytes for {num_experts} x {expert_bytes}", data.len());
        }
        let num_resident = (Self::gpu_fraction() * num_experts as f64).round() as usize;
        let profiled = Self::profile_order(layer, num_experts);
        let order = profiled.clone().unwrap_or_else(|| (0..num_experts).collect());

        // TITAN_TIERED_PROBE=1 stores residents in shuffled slots: output must not change.
        let probe = env_flag("TITAN_TIERED_PROBE");
        let mut map = vec![NOT_RESIDENT; num_experts];
        let mut owner = vec![NOT_RESIDENT; num_resident];
        for (i, &e) in order.iter().take(num_resident).enumerate() {
            let slot = if probe { (i * 37 + 11) % num_resident } else { i };
            map[e] = slot as u32;
            owner[slot] = e as u32;
        }
        let mut slotted = vec![0u8; num_resident.max(1) * expert_bytes];
        for (slot, &e) in owner.iter().enumerate() {
            let e = e as usize;
            slotted[slot * expert_bytes..(slot + 1) * expert_bytes]
                .copy_from_slice(&data[e * expert_bytes..(e + 1) * expert_bytes]);
        }
        let slots = dev.clone_htod(&slotted)?;
        let gpu_mib = slotted.len() >> 20;
        drop(slotted);
        let slot_map = dev.clone_htod(&map)?;
        let policy = Self::policy();
        let flex = fingerprint_slots().min(num_resident);
        let (host, host_index, host_mib, host_kind) = match mapped {
            Some((mm, offset)) => {
                // TITAN_TIERED_POPULATE=f: page in (and map) the hottest f of the non-resident experts now.
                let share = std::env::var("TITAN_TIERED_POPULATE").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
                if share > 0.0 {
                    let take = ((num_experts - num_resident) as f64 * share.clamp(0.0, 1.0)).round() as usize;
                    let ranges = order[num_resident..].iter().take(take).map(|&e| (offset + e * expert_bytes, expert_bytes)).collect();
                    run_prefetch(&PrefetchJob { map: mm.clone(), ranges }, true);
                }
                // TITAN_TIERED_PAGEOUT=1: the GPU-resident experts' bytes were read once for the upload; drop
                // them from the page cache so they do not crowd out host experts (the cgroup charges both).
                if env_flag("TITAN_TIERED_PAGEOUT") {
                    let base = mm.as_ptr() as usize;
                    for &e in &owner {
                        let a = base + offset + e as usize * expert_bytes;
                        let a0 = a & !(PAGE_BYTES - 1);
                        unsafe { libc::madvise(a0 as *mut libc::c_void, expert_bytes + (a - a0), MADV_PAGEOUT) };
                    }
                }
                let mapped_mib = ((num_experts - num_resident) * expert_bytes) >> 20;
                (HostStore::Mapped { map: mm, offset }, (0..num_experts as u32).collect(), mapped_mib, "mmap")
            }
            None => {
                // A static placement never needs a resident expert on the host again.
                let mut host_index = vec![NOT_RESIDENT; num_experts];
                let flex_set = &order[num_resident - flex..num_resident];
                let keep: Vec<usize> = (0..num_experts)
                    .filter(|&e| policy != Policy::Static || map[e] == NOT_RESIDENT || flex_set.contains(&e))
                    .collect();
                let mut host = Vec::with_capacity(keep.len() * expert_bytes);
                if env_flag("TITAN_TIERED_HUGEPAGE") {
                    // before first touch: THP is madvise-only on titan; a cold CPU miss then walks 2 MiB pages
                    let (a, len) = (host.as_ptr() as usize, host.capacity());
                    let a0 = a.next_multiple_of(HUGE_PAGE_BYTES);
                    if a + len > a0 {
                        unsafe { libc::madvise(a0 as *mut libc::c_void, a + len - a0, libc::MADV_HUGEPAGE) };
                    }
                }
                for (i, &e) in keep.iter().enumerate() {
                    host_index[e] = i as u32;
                    host.extend_from_slice(&data[e * expert_bytes..(e + 1) * expert_bytes]);
                }
                let mib = host.len() >> 20;
                (HostStore::Owned(host), host_index, mib, "owned")
            }
        };
        tracing::info!(
            "titan tiered experts: {num_experts} x [{n}, {k}] {dtype:?}, {num_resident} on GPU ({gpu_mib} MiB), {host_mib} MiB on host ({host_kind}), {} placement{}, {} policy",
            if profiled.is_some() { "profiled" } else { "lowest-id" },
            if probe { ", permuted slots (probe)" } else { "" },
            match policy {
                Policy::Lru => "lru",
                Policy::Lfu if async_uploads() => "lfu (async uploads)",
                Policy::Lfu => "lfu",
                Policy::Static => "static",
            }
        );
        let host_bytes = match &host {
            HostStore::Owned(v) => v.len(),
            HostStore::Mapped { .. } => (num_experts - num_resident) * expert_bytes,
        };
        crate::titan_monitor::tiered_placement(proj == "gate", num_experts, num_resident, num_resident * expert_bytes, host_bytes);
        let slots_ptr = slice_ptr(&slots, 0).0;
        let cache = Cache {
            slots,
            slot_map,
            map,
            owner,
            last_use: vec![0; num_resident],
            tick: 0,
            prefill_hist: vec![0; num_experts],
            decoding: true,
            lfu: vec![0.0; num_experts],
            pending: Vec::new(),
            slots_ptr,
        };
        let mut me = Self {
            pfs: None,
            cache: std::sync::Mutex::new(cache),
            policy,
            host,
            host_index,
            expert_bytes,
            dtype,
            num_experts,
            n,
            k,
            layer,
            proj,
            dev: dev.clone(),
            order,
            flex,
            fast_ok: std::sync::OnceLock::new(),
        };
        if titan_pfs::enabled() && policy == Policy::Static && me.flex == 0 {
            let cold = &me.order[num_resident..];
            let staged = &cold[..(cold.len() as f64 * titan_pfs::share()).ceil() as usize];
            let owned = match &me.host {
                HostStore::Owned(v) => Some(&v[..]),
                HostStore::Mapped { .. } => None,
            };
            me.pfs = titan_pfs::register(dev, layer, proj, num_experts, staged, |e| me.host_expert(e), owned)?;
        }
        Ok(me)
    }

    fn host_expert(&self, e: usize) -> &[u8] {
        let i = self.host_index[e] as usize;
        let eb = self.expert_bytes;
        match &self.host {
            HostStore::Owned(v) => &v[i * eb..(i + 1) * eb],
            HostStore::Mapped { map, offset } => &map[offset + i * eb..offset + (i + 1) * eb],
        }
    }

    /// Mapped host store: start reading the non-resident experts among `ids` from disk now
    /// (MADV_WILLNEED), so their page faults overlap the GPU work before the CPU twin needs them.
    pub fn prefetch(&self, ids: &[u32]) {
        let HostStore::Mapped { map, offset } = &self.host else {
            return;
        };
        if !willneed_on() {
            return;
        }
        let cache = self.cache.lock().unwrap();
        if cache.owner.len() == self.num_experts {
            return;
        }
        let mut seen = vec![false; self.num_experts];
        for &e in ids {
            let e = e as usize;
            if cache.map[e] == NOT_RESIDENT && !std::mem::replace(&mut seen[e], true) {
                let _ = map.advise_range(Advice::WillNeed, offset + e * self.expert_bytes, self.expert_bytes);
            }
        }
    }

    /// P1: `ids` are the predicted routing of this (gate, up, down) layer, made `depth` layers ahead. Queue
    /// their non-resident host bytes for the prefetch workers and keep the prediction for scoring.
    pub fn lookahead(projs: &[&TieredExperts], depth: usize, ids: &[u32]) {
        let Some(first) = projs.first() else {
            return;
        };
        if (1..=MAX_LOOKAHEAD).contains(&depth) {
            let mut p = PENDING.lock().unwrap();
            p.retain(|(l, d, _)| !(*l == first.layer && *d == depth));
            p.push((first.layer, depth, ids.to_vec()));
        }
        let t0 = std::time::Instant::now();
        let mut uniq: Vec<u32> = ids.to_vec();
        uniq.sort_unstable();
        uniq.dedup();
        let resident: Vec<bool> = {
            let cache = first.cache.lock().unwrap();
            uniq.iter().map(|&e| cache.map[e as usize] != NOT_RESIDENT).collect()
        };
        let mut map_ranges: Option<(Arc<Mmap>, Vec<(usize, usize)>)> = None;
        for p in projs {
            let HostStore::Mapped { map, offset } = &p.host else {
                continue;
            };
            let entry = map_ranges.get_or_insert_with(|| (map.clone(), Vec::new()));
            for (&e, &r) in uniq.iter().zip(&resident) {
                if !r {
                    entry.1.push((offset + e as usize * p.expert_bytes, p.expert_bytes));
                }
            }
        }
        if let Some((map, ranges)) = map_ranges {
            if !ranges.is_empty() && prefetcher().try_send(PrefetchJob { map, ranges }).is_err() {
                PF_STATS[4].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        add_time(11, t0);
    }

    /// Score the predictions made for this layer against its actual decode routing.
    fn score_predictions(&self, map: &[u32], ids: &[u32]) {
        use std::sync::atomic::Ordering::Relaxed;
        let preds: Vec<(usize, Vec<u32>)> = {
            let mut p = PENDING.lock().unwrap();
            if p.is_empty() {
                return;
            }
            let (mine, rest): (Vec<_>, Vec<_>) = p.drain(..).partition(|(l, _, _)| *l == self.layer);
            *p = rest;
            mine.into_iter().map(|(_, d, v)| (d, v)).collect()
        };
        let mut actual: Vec<u32> = ids.to_vec();
        actual.sort_unstable();
        actual.dedup();
        let nr = |e: u32| map[e as usize] == NOT_RESIDENT;
        for (d, mut pred) in preds {
            pred.sort_unstable();
            pred.dedup();
            let s = &PRED_STATS[d - 1];
            let tp: Vec<u32> = pred.iter().copied().filter(|e| actual.binary_search(e).is_ok()).collect();
            s[0].fetch_add(tp.len() as u64, Relaxed);
            s[1].fetch_add(pred.len() as u64, Relaxed);
            s[2].fetch_add(actual.len() as u64, Relaxed);
            s[3].fetch_add(tp.iter().filter(|&&e| nr(e)).count() as u64, Relaxed);
            s[4].fetch_add(pred.iter().filter(|&&e| nr(e)).count() as u64, Relaxed);
            s[5].fetch_add(actual.iter().filter(|&&e| nr(e)).count() as u64, Relaxed);
        }
    }

    /// Fingerprint: re-seed the flex slots with the request's hottest prefill experts (ties and empty
    /// counts fall back to the placement order), keeping flex experts that stay wanted where they are.
    fn reseed(&self, cache: &mut Cache) -> Result<()> {
        let n_res = cache.owner.len();
        let core = n_res - self.flex;
        let hist = &cache.prefill_hist;
        let rank: Vec<usize> = {
            let mut r = vec![0; self.num_experts];
            for (i, &e) in self.order.iter().enumerate() {
                r[e] = i;
            }
            r
        };
        let mut cand: Vec<usize> = self.order[core..].to_vec();
        cand.sort_by(|&a, &b| hist[b].cmp(&hist[a]).then(rank[a].cmp(&rank[b])));
        let want: Vec<u32> = cand.iter().take(self.flex).map(|&e| e as u32).collect();
        let flex_slots: Vec<usize> = (0..n_res).filter(|&s| rank[cache.owner[s] as usize] >= core).collect();
        let mut free: Vec<usize> = flex_slots.iter().copied().filter(|&s| !want.contains(&cache.owner[s])).collect();
        let eb = self.expert_bytes;
        let mut uploads = 0u64;
        for &e in &want {
            if cache.map[e as usize] != NOT_RESIDENT {
                continue;
            }
            let Some(slot) = free.pop() else {
                break;
            };
            let old = cache.owner[slot];
            cache.map[old as usize] = NOT_RESIDENT;
            let src = self.host_expert(e as usize);
            let mut dst = cache.slots.slice_mut(slot * eb..(slot + 1) * eb);
            self.dev.memcpy_htod(src, &mut dst)?;
            crate::titan_monitor::tiered_upload_bytes(eb);
            cache.owner[slot] = e;
            cache.map[e as usize] = slot as u32;
            uploads += 1;
        }
        if uploads > 0 {
            let Cache { slot_map, map, .. } = cache;
            self.dev.memcpy_htod(map.as_slice(), slot_map)?;
            if self.proj == "gate" {
                STATS[UPLOADS].fetch_add(uploads, std::sync::atomic::Ordering::Relaxed);
            }
        }
        Ok(())
    }

    pub fn device(&self) -> &CudaDevice {
        &self.dev
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    fn format(&self) -> Format {
        match self.dtype {
            GgmlDType::Q5K => Format::Q5K,
            GgmlDType::Q6K => Format::Q6K,
            GgmlDType::Q1_0 => Format::Q1_0,
            GgmlDType::IQ4NL => Format::IQ4NL,
            GgmlDType::MXFP4 => Format::MXFP4,
            GgmlDType::NVFP4 => Format::NVFP4,
            _ => Format::Q4K,
        }
    }

    /// `x`: `[batch, topk or 1, k]` f32, `ids`: `[batch, topk]` u32 -> `[batch, topk, n]` f32,
    /// the contract of `QMatMul::indexed_moe_forward`.
    pub fn forward(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        self.forward_with_ids(x, ids, None)
    }

    /// Routing ids on the host, fetched once per MoE layer and shared by its gate, up and down
    /// projections (each fetch is a device sync).
    pub fn routing_to_host(ids: &Tensor) -> Result<Vec<u32>> {
        let t0 = std::time::Instant::now();
        let v = ids.flatten_all()?.to_vec1::<u32>();
        add_time(0, t0);
        v
    }

    /// `forward` with the routing ids already on the host (from `routing_to_host`).
    pub fn forward_with_ids(&self, x: &Tensor, ids: &Tensor, ids_host: Option<&[u32]>) -> Result<Tensor> {
        let q = self.quantize_input(x)?;
        self.forward_q(&q, ids, ids_host)
    }

    /// Quantize `x` (`[batch, topk or 1, k]` f32) to q8_1 on the GPU. Projections that share an input
    /// (gate and up) share one `QInput`, and with it one host copy for the CPU misses.
    pub fn quantize_input(&self, x: &Tensor) -> Result<QInput> {
        let (batch, input_dim1, k) = x.dims3()?;
        if k != self.k {
            candle_core::bail!("titan tiered experts: x {:?} vs k={}", x.dims(), self.k);
        }
        if x.dtype() != DType::F32 {
            candle_core::bail!("titan tiered experts: input must be f32, got {:?}", x.dtype());
        }
        let x = x.contiguous()?;
        let (x_storage, x_layout) = x.storage_and_layout();
        let Storage::Cuda(x_cuda) = &*x_storage else {
            candle_core::bail!("titan tiered experts: input not on CUDA");
        };
        let x_slice = x_cuda.as_cuda_slice::<f32>()?;
        let rows = batch * input_dim1;
        let k_padded = k.div_ceil(MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
        let q8_row_bytes = k_padded / 32 * Q8_1_BLOCK_BYTES;
        let xq = self.dev.alloc_zeros::<u8>(rows * q8_row_bytes)?;
        let (x_ptr, g0) = slice_ptr(x_slice, x_layout.start_offset());
        let (xq_ptr, g1) = slice_ptr(&xq, 0);
        // Slices cross the oxide ABI as (pointer, element count).
        let x_len = (rows * k) as u64;
        let (kx, kp) = (k as u32, k_padded as u32);
        let quant = self.dev.get_or_load_custom_func("quantize_q8_1", MODULE, PTX)?;
        let mut b = quant.builder();
        b.arg(&x_ptr).arg(&x_len).arg(&xq_ptr).arg(&kx).arg(&kp);
        let threads = (rows * k_padded) as u32;
        unsafe {
            b.launch(LaunchConfig { grid_dim: (threads / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
        }
        .w()?;
        drop((g0, g1));
        Ok(QInput { xq, host: None, batch, input_dim1, k, q8_row_bytes })
    }

    /// Whether any of these routed experts is served by the CPU (so the input must be on the host).
    pub fn any_miss(&self, ids_host: &[u32]) -> bool {
        let cache = self.cache.lock().unwrap();
        cache.owner.len() != self.num_experts && ids_host.iter().any(|&e| cache.map[e as usize] == NOT_RESIDENT)
    }

    /// The gemv (and CPU misses) on an input quantized by `quantize_input`.
    pub fn forward_q(&self, q: &QInput, ids: &Tensor, ids_host: Option<&[u32]>) -> Result<Tensor> {
        let (out, _, batch, topk) = self.forward_q_parts(q, ids, ids_host, false)?;
        Ok(self.wrap(out, batch, topk))
    }

    fn wrap(&self, out: CudaSlice<f32>, batch: usize, topk: usize) -> Tensor {
        let storage = CudaStorage::wrap_cuda_slice(out, self.dev.clone());
        Tensor::from((Storage::Cuda(storage), Shape::from((batch, topk, self.n))))
    }

    /// Kernel-name prefix of this dtype in mmvq_moe_oxide.ptx.
    fn moe_prefix(&self) -> &'static str {
        match self.dtype {
            GgmlDType::Q5K => "q5k",
            GgmlDType::Q6K => "q6k",
            GgmlDType::Q1_0 => "q1_0",
            GgmlDType::IQ4NL => "iq4_nl",
            GgmlDType::MXFP4 => "mxfp4",
            GgmlDType::NVFP4 => "nvfp4",
            _ => "q4k",
        }
    }

    /// llama.cpp `should_use_small_k` at b = 1: fewer quant blocks per row than 4 warps cover in one
    /// iteration (4 * 32 / (qi / vdr)), so a block computes 4 rows.
    fn b1_small_k(&self) -> bool {
        let (qk, threads_per_block) = match self.dtype {
            GgmlDType::Q4K | GgmlDType::Q5K => (256, 16),
            GgmlDType::Q6K => (256, 32),
            GgmlDType::Q1_0 => (128, 4),
            GgmlDType::NVFP4 => (64, 2),
            _ => (32, 2),
        };
        self.k / qk < 4 * (32 / threads_per_block)
    }

    /// The slot-map expert GEMV for this dtype, and whether it is the one-row-per-warp variant.
    fn gemv_kernel(&self) -> (&'static str, bool) {
        // k <= 512 (two K-quant blocks per row): one output row per warp, 4 per CUDA block
        let warp_rows = self.k <= WARP_ROWS_MAX_K && matches!(self.dtype, GgmlDType::Q4K | GgmlDType::Q5K);
        let name = match self.dtype {
            GgmlDType::Q4K if warp_rows => "q4k_q8_1_moe_gemv_w",
            GgmlDType::Q5K if warp_rows => "q5k_q8_1_moe_gemv_w",
            GgmlDType::Q4K => "q4k_q8_1_moe_gemv",
            GgmlDType::Q5K => "q5k_q8_1_moe_gemv",
            GgmlDType::Q6K => "q6k_q8_1_moe_gemv",
            // llama.cpp's mul_mat_vec_q<Q1_0> float program (fast-math; there is no candle Q1_0 MoE kernel)
            GgmlDType::Q1_0 => "q1_0_q8_1_moe_gemv",
            GgmlDType::IQ4NL => "iq4_nl_q8_1_moe_gemv",
            GgmlDType::MXFP4 => "mxfp4_q8_1_moe_gemv",
            GgmlDType::NVFP4 => "nvfp4_q8_1_moe_gemv",
            _ => unreachable!("checked in from_qtensor"),
        };
        (name, warp_rows)
    }

    /// Gate and up share their input and routing: one GPU launch each, then ONE parallel CPU pass over
    /// both projections' misses (instead of two pool round trips). Bit-identical to two `forward_q`s.
    pub fn forward_gate_up(
        gate: &TieredExperts,
        up: &TieredExperts,
        q: &QInput,
        ids: &Tensor,
        ids_host: &[u32],
    ) -> Result<(Tensor, Tensor)> {
        let (mut og, mg, batch, topk) = gate.forward_q_parts(q, ids, Some(ids_host), true)?;
        let (mut ou, mu, _, _) = up.forward_q_parts(q, ids, Some(ids_host), true)?;
        if !mg.is_empty() || !mu.is_empty() {
            let fetched;
            let xq_host = match q.host.as_deref() {
                Some(h) => h,
                None => {
                    fetched = gate.dev.clone_dtoh(&q.xq)?;
                    &fetched[..]
                }
            };
            let t1 = std::time::Instant::now();
            let results = Self::compute_misses(&[gate, up], &[&mg, &mu], ids_host, xq_host, q.input_dim1, topk, q.q8_row_bytes);
            add_time(2, t1);
            let t2 = std::time::Instant::now();
            let (rg, ru): (Vec<_>, Vec<_>) = results.into_iter().partition(|r| r.0 == 0);
            gate.upload_rows(&mut og, rg.into_iter().map(|(_, t, r0, v)| (t, r0, v)))?;
            up.upload_rows(&mut ou, ru.into_iter().map(|(_, t, r0, v)| (t, r0, v)))?;
            add_time(3, t2);
        }
        Ok((gate.wrap(og, batch, topk), up.wrap(ou, batch, topk)))
    }

    /// The routed experts of one SiLU-gated MoE layer through the new miss paths: the doorbell (`TITAN_DOORBELL=1`)
    /// and/or one CPU pass per miss (`TITAN_CPU_ONEPASS`). `None` when neither is on or the layer does not qualify
    /// (no misses possible, a non-static placement, a routing trace): the caller runs `forward_gate_up` + `forward_q`.
    #[allow(clippy::type_complexity)]
    pub fn forward_fast(
        gate: &TieredExperts,
        up: &TieredExperts,
        down: &TieredExperts,
        xs: &Tensor,
        indices: &Tensor,
        preds: &[([&TieredExperts; 3], usize, Tensor)],
        act: &dyn Fn(&Tensor, &Tensor) -> Result<Tensor>,
    ) -> Result<Option<Tensor>> {
        if !doorbell::enabled() && onepass::mode() == 0 {
            return Ok(None);
        }
        if trace_path().is_some() || gate.all_resident() || !*gate.fast_ok.get_or_init(|| Self::onepass_ok(gate, up, down)) {
            return Ok(None);
        }
        if doorbell::enabled() {
            if let Some(y) = Self::forward_doorbell(gate, up, down, xs, indices, preds, act)? {
                return Ok(Some(y));
            }
            if onepass::mode() == 0 {
                return Ok(None);
            }
        }
        Self::forward_onepass_sync(gate, up, down, xs, indices, preds, act).map(Some)
    }

    /// GPU part of a projection. With `defer` (static placement only), CPU misses are returned as task
    /// indices instead of computed, so the caller can batch them with another projection's.
    fn forward_q_parts(
        &self,
        q: &QInput,
        ids: &Tensor,
        ids_host: Option<&[u32]>,
        defer: bool,
    ) -> Result<(CudaSlice<f32>, Vec<usize>, usize, usize)> {
        let (batch, input_dim1, k) = (q.batch, q.input_dim1, q.k);
        let (id_batch, topk) = ids.dims2()?;
        if k != self.k || id_batch != batch {
            candle_core::bail!("titan tiered experts: input [{batch}, {input_dim1}, {k}] / ids {:?} vs k={}", ids.dims(), self.k);
        }
        let ids = ids.contiguous()?;
        let (ids_storage, ids_layout) = ids.storage_and_layout();
        let Storage::Cuda(ids_cuda) = &*ids_storage else {
            candle_core::bail!("titan tiered experts: ids not on CUDA");
        };
        let ids_slice = ids_cuda.as_cuda_slice::<u32>()?;
        if self.pfs.is_some() {
            titan_pfs::idle(&self.dev, batch)?;
        }
        let mut cache = self.cache.lock().unwrap();
        self.land(&mut cache)?;
        if self.flex > 0 && self.policy == Policy::Static {
            if let Some(h) = ids_host {
                if batch > 1 {
                    if cache.decoding {
                        cache.decoding = false;
                        cache.prefill_hist.iter_mut().for_each(|c| *c = 0);
                    }
                    for &e in h {
                        cache.prefill_hist[e as usize] += 1;
                    }
                } else if !cache.decoding {
                    cache.decoding = true;
                    self.reseed(&mut cache)?;
                }
            }
        }
        if batch == 1 && self.proj == "gate" {
            if let Some(h) = ids_host {
                self.score_predictions(&cache.map, h);
            }
        }
        let q8_row_bytes = q.q8_row_bytes;
        let xq = &q.xq;
        let tasks = batch * topk;
        let out = {
            let (ids_ptr, g1) = slice_ptr(ids_slice, ids_layout.start_offset());
            let out = self.gemv_launch(&cache, q, ids_ptr, topk)?;
            drop(g1);
            out
        };

        let all_resident = cache.owner.len() == self.num_experts;
        let trace = if self.proj == "gate" { trace_path() } else { None };
        let ids_host: std::borrow::Cow<[u32]> = match ids_host {
            Some(h) => std::borrow::Cow::Borrowed(h),
            None if all_resident && trace.is_none() => std::borrow::Cow::Owned(Vec::new()),
            None => std::borrow::Cow::Owned(
                self.dev.clone_dtoh(&ids_slice.slice(ids_layout.start_offset()..ids_layout.start_offset() + tasks))?,
            ),
        };
        if let Some(path) = trace {
            trace_ids(path, self.layer, batch, topk, &ids_host);
        }
        let mut out = out;
        let mut deferred = Vec::new();
        // TITAN_TIERED_DEBUG_SKIP_MISSES=1: timing experiment only, output is WRONG (misses stay zero).
        static SKIP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let skip = *SKIP.get_or_init(|| env_flag("TITAN_TIERED_DEBUG_SKIP_MISSES"));
        if !all_resident && !skip {
            if defer && self.policy != Policy::Lru {
                deferred = (0..tasks).filter(|&t| cache.map[ids_host[t] as usize] == NOT_RESIDENT).collect();
            } else {
                self.cpu_misses(&mut out, &cache.map, &ids_host, xq, q.host.as_deref(), input_dim1, topk, q8_row_bytes)?;
            }
            self.touch_and_admit(&mut cache, &ids_host, batch)?;
        }
        Ok((out, deferred, batch, topk))
    }

    /// The slot-map expert GEMV over the GPU-resident experts (misses stay zero): `[tasks, n]` f32.
    fn gemv_launch(&self, cache: &Cache, q: &QInput, ids_ptr: u64, topk: usize) -> Result<CudaSlice<f32>> {
        let (batch, input_dim1, k) = (q.batch, q.input_dim1, q.k);
        let k_padded = k.div_ceil(MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
        let xq = &q.xq;
        let tasks = batch * topk;
        let out = self.dev.alloc_zeros::<f32>(tasks * self.n)?;
        let (xq_ptr, g2) = slice_ptr(xq, 0);
        let (w_ptr, g3) = slice_ptr(&cache.slots, 0);
        let (map_ptr, g4) = slice_ptr(&cache.slot_map, 0);
        let (out_ptr, g5) = slice_ptr(&out, 0);
        let (kx, kp) = (k as u32, k_padded as u32);

        let (n, topk_u, dim1) = (self.n as u32, topk as u32, input_dim1 as u32);
        let moe = mmvq_moe_mode();
        if moe != 0 && batch <= MMVQ_MOE_MAX_BATCH {
            // llama.cpp-shaped kernels (oxide-kernels/mmvq-moe), each column reduced exactly as the
            // kernel below reduces it: same output bits, so the CPU twin is unchanged.
            // b = 1: llama.cpp's b1 grid where k is large (gate/up); for small k (the down projections)
            // its small_k variant measured slower than the MoE grid (m4/mmvqmoe), so those take the MoE grid.
            let (name, grid, block) = if batch == 1 && (moe == 3 || (moe == 1 && !self.b1_small_k())) {
                let rpb = if self.b1_small_k() { 4 } else { 1 };
                let sk = if rpb == 4 { "_sk" } else { "" };
                (format!("{}_moe_b1{sk}", self.moe_prefix()), (self.n.div_ceil(rpb) as u32, tasks as u32, 1), (32, 4, 1))
            } else {
                (format!("{}_moe_mmvq", self.moe_prefix()), (self.n.div_ceil(2) as u32, topk_u, 1), (32, batch as u32, 1))
            };
            let gemv = self.dev.get_or_load_custom_func(&name, MOE_MODULE, MOE_PTX)?;
            let mut b = gemv.builder();
            b.arg(&w_ptr).arg(&xq_ptr).arg(&ids_ptr).arg(&map_ptr).arg(&out_ptr).arg(&n).arg(&kx).arg(&kp).arg(&topk_u).arg(&dim1);
            unsafe { b.launch(LaunchConfig { grid_dim: grid, block_dim: block, shared_mem_bytes: 0 }) }.w()?;
        } else {
            let (name, warp_rows) = self.gemv_kernel();
            let w_len = (cache.slots.len() / 4) as u64;
            let xq_len = (xq.len() / 4) as u64;
            let ids_len = tasks as u64;
            let map_len = self.num_experts as u64;
            let gemv = self.dev.get_or_load_custom_func(name, MODULE, PTX)?;
            let mut b = gemv.builder();
            b.arg(&w_ptr).arg(&w_len).arg(&xq_ptr).arg(&xq_len).arg(&ids_ptr).arg(&ids_len)
                .arg(&map_ptr).arg(&map_len).arg(&out_ptr).arg(&n).arg(&kx).arg(&kp).arg(&topk_u).arg(&dim1);
            unsafe {
                let blocks = if warp_rows { self.n.div_ceil(4) * tasks } else { self.n * tasks };
                b.launch(LaunchConfig { grid_dim: (blocks as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 })
            }
            .w()?;
        }
        drop((g2, g3, g4, g5));
        if timing_level() >= 2 {
            let t = std::time::Instant::now();
            candle_core::backend::BackendDevice::synchronize(&self.dev)?;
            add_time(GPU_GEMV, t);
        }
        Ok(out)
    }

    /// CPU misses of projections sharing one input and routing (`misses[p]`: task indices of
    /// `projs[p]`), grouped by expert: one work item per (projection, expert, row chunk) computes that
    /// chunk for every task routed to the expert, so the chunk's weight rows are streamed from RAM once
    /// per layer however many verification rows (tokens) picked the expert. Returns (proj, task, r0, values).
    #[allow(clippy::too_many_arguments)]
    fn compute_misses(
        projs: &[&TieredExperts],
        misses: &[&Vec<usize>],
        ids_host: &[u32],
        xq_host: &[u8],
        input_dim1: usize,
        topk: usize,
        q8_row_bytes: usize,
    ) -> Vec<(usize, usize, usize, Vec<f32>)> {
        let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
        for (p, tasks) in misses.iter().enumerate() {
            let first = groups.len();
            for &t in tasks.iter() {
                let e = ids_host[t];
                static NOGROUP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                let nogroup = *NOGROUP.get_or_init(|| env_flag("TITAN_TIERED_NOGROUP"));
                match groups[first..].iter_mut().find(|(gp, g)| !nogroup && *gp == p && ids_host[g[0]] == e) {
                    Some((_, g)) => g.push(t),
                    None => groups.push((p, vec![t])),
                }
            }
        }
        if timing_on() {
            use std::sync::atomic::Ordering::Relaxed;
            TIMING[PASSES].fetch_add(1, Relaxed);
            TIMING[PASSES + 1].fetch_add(misses.iter().map(|m| m.len() as u64).sum(), Relaxed);
            TIMING[PASSES + 2].fetch_add(groups.len() as u64, Relaxed);
        }
        let work: Vec<(usize, usize)> = groups
            .iter()
            .enumerate()
            .flat_map(|(g, (p, _))| (0..projs[*p].n).step_by(rows_per_chunk()).map(move |r0| (g, r0)))
            .collect();
        let flt0 = if timing_on() { Some(faults()) } else { None };
        let t_mon = std::time::Instant::now();
        let item = |&(g, r0): &(usize, usize)| {
            let (p, tasks) = &groups[g];
            tasks.iter().map(move |&t| (*p, t, r0, projs[*p].chunk(t, r0, ids_host, xq_host, input_dim1, topk, q8_row_bytes)))
        };
        let out = match miss_pool() {
            Some(pool) => {
                let slots: Vec<std::sync::OnceLock<Vec<(usize, usize, usize, Vec<f32>)>>> =
                    (0..work.len()).map(|_| std::sync::OnceLock::new()).collect();
                pool.run(work.len(), &|i| {
                    let _ = slots[i].set(item(&work[i]).collect());
                });
                slots.into_iter().flat_map(|s| s.into_inner().unwrap_or_default()).collect()
            }
            None => work.par_iter().flat_map_iter(item).collect(),
        };
        crate::titan_monitor::tiered_miss_ns(t_mon.elapsed().as_nanos() as u64);
        if let Some((maj, min)) = flt0 {
            use std::sync::atomic::Ordering::Relaxed;
            let (maj1, min1) = faults();
            TIMING[MAJFLT].fetch_add(maj1 - maj, Relaxed);
            TIMING[MINFLT].fetch_add(min1 - min, Relaxed);
        }
        out
    }

    /// One CPU chunk of a task's output rows `[r0, r0 + ROWS_PER_CPU_CHUNK)`.
    #[allow(clippy::too_many_arguments)]
    fn chunk(&self, t: usize, r0: usize, ids_host: &[u32], xq_host: &[u8], input_dim1: usize, topk: usize, q8_row_bytes: usize) -> Vec<f32> {
        let fmt = self.format();
        let row_bytes = self.expert_bytes / self.n;
        let n = self.n;
        let simd = std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma");
        let e = ids_host[t] as usize;
        let in_row = if input_dim1 == 1 { t / topk } else { t };
        let xq_row = &xq_host[in_row * q8_row_bytes..(in_row + 1) * q8_row_bytes];
        let expert = self.host_expert(e);
        let row_of = |r: usize| &expert[r * row_bytes..][..row_bytes];
        let end = (r0 + rows_per_chunk()).min(n);
        if simd && matches!(fmt, Format::Q4K | Format::Q5K | Format::Q6K | Format::IQ4NL | Format::MXFP4 | Format::NVFP4) {
            let rows: Vec<&[u8]> = (r0..end).map(row_of).collect();
            let mut vals = vec![0f32; end - r0];
            unsafe { titan_cpu::rows_lanes(fmt, &rows, xq_row, self.k, &mut vals) };
            return vals;
        }
        let mut vals = Vec::with_capacity(end - r0);
        let mut r = r0;
        if simd {
            while r + 8 <= end {
                let rows: [&[u8]; 8] = std::array::from_fn(|i| row_of(r + i));
                vals.extend_from_slice(&unsafe { titan_cpu::rows8(fmt, &rows, xq_row, self.k) });
                r += 8;
            }
        }
        vals.extend((r..end).map(|r| titan_cpu::row(fmt, row_of(r), xq_row, self.k)));
        vals
    }

    /// Write CPU-computed chunks into their task rows of `out`: one upload of every row (with its task
    /// index) and one scatter launch, stream-ordered after the GEMV that left those rows zero.
    fn upload_rows(&self, out: &mut CudaSlice<f32>, chunks: impl Iterator<Item = (usize, usize, Vec<f32>)>) -> Result<()> {
        let n = self.n;
        let mut rows: std::collections::BTreeMap<usize, Vec<f32>> = std::collections::BTreeMap::new();
        for (t, r0, vals) in chunks {
            rows.entry(t).or_insert_with(|| vec![0f32; n])[r0..r0 + vals.len()].copy_from_slice(&vals);
        }
        let m = rows.len();
        if m == 0 {
            return Ok(());
        }
        let mut packed: Vec<u32> = Vec::with_capacity(m * (n + 1));
        packed.extend(rows.keys().map(|&t| t as u32));
        for row in rows.values() {
            packed.extend(row.iter().map(|v| v.to_bits()));
        }
        let buf = self.dev.clone_htod(&packed)?;
        crate::titan_monitor::tiered_upload_bytes(packed.len() * 4);
        let (buf_ptr, g0) = slice_ptr(&buf, 0);
        let (out_ptr, g1) = slice_ptr(out, 0);
        let (buf_len, m_u, n_u) = (packed.len() as u64, m as u32, n as u32);
        let scatter = self.dev.get_or_load_custom_func("scatter_rows", MODULE, PTX)?;
        let mut b = scatter.builder();
        b.arg(&buf_ptr).arg(&buf_len).arg(&m_u).arg(&out_ptr).arg(&n_u);
        unsafe {
            b.launch(LaunchConfig { grid_dim: ((m * n).div_ceil(256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
        }
        .w()?;
        drop((g0, g1));
        Ok(())
    }

    /// Record slot use; under `lru`, upload this decode step's missed experts over the least recently
    /// used slots; under `lfu`, see `admit_lfu`. Stream order puts each upload after the kernels that read
    /// the old contents.
    fn touch_and_admit(&self, cache: &mut Cache, ids: &[u32], batch: usize) -> Result<()> {
        use std::sync::atomic::Ordering::Relaxed;
        let decode = batch == 1;
        cache.tick += 1;
        let tick = cache.tick;
        let mut missed: Vec<u32> = Vec::new();
        for &e in ids {
            let slot = cache.map[e as usize];
            if slot == NOT_RESIDENT {
                if !missed.contains(&e) {
                    missed.push(e);
                }
            } else {
                cache.last_use[slot as usize] = tick;
            }
        }
        if self.proj == "gate" && batch <= ADMIT_MAX_ROWS {
            let m = ids.iter().filter(|e| missed.contains(e)).count() as u64;
            let (hs, ms) = if decode { (HITS, MISSES) } else { (VHITS, VMISSES) };
            STATS[hs].fetch_add(ids.len() as u64 - m, Relaxed);
            STATS[ms].fetch_add(m, Relaxed);
            if self.layer == 0 && STATS[CALLS].fetch_add(1, Relaxed) % 256 == 255 {
                let (h, m, u) = (STATS[HITS].load(Relaxed), STATS[MISSES].load(Relaxed), STATS[UPLOADS].load(Relaxed));
                let (vh, vm) = (STATS[VHITS].load(Relaxed), STATS[VMISSES].load(Relaxed));
                tracing::info!(
                    "titan tiered: decode hit rate {:.1}% ({h} hits, {m} misses, {u} uploads, {} landed); verify rows hit rate {:.1}% ({vh} hits, {vm} misses); {}{}",
                    100.0 * h as f64 / (h + m).max(1) as f64,
                    STATS[LANDED].load(Relaxed),
                    100.0 * vh as f64 / (vh + vm).max(1) as f64,
                    proc_counters(),
                    pred_report()
                );
            }
        }
        if self.policy == Policy::Lfu && batch <= ADMIT_MAX_ROWS {
            return self.admit_lfu(cache, ids, &missed, batch);
        }
        if self.policy != Policy::Lru || !decode || cache.owner.is_empty() {
            return Ok(());
        }
        let eb = self.expert_bytes;
        for e in missed {
            // victim: least recently used slot not touched by this step
            let Some(slot) = (0..cache.last_use.len()).filter(|&s| cache.last_use[s] != tick).min_by_key(|&s| cache.last_use[s]) else {
                break;
            };
            let old = cache.owner[slot];
            if old != NOT_RESIDENT {
                cache.map[old as usize] = NOT_RESIDENT;
            }
            let src = self.host_expert(e as usize);
            let mut dst = cache.slots.slice_mut(slot * eb..(slot + 1) * eb);
            self.dev.memcpy_htod(src, &mut dst)?;
            crate::titan_monitor::tiered_upload_bytes(eb);
            cache.owner[slot] = e;
            cache.map[e as usize] = slot as u32;
            cache.last_use[slot] = tick;
            if self.proj == "gate" {
                STATS[UPLOADS].fetch_add(1, Relaxed);
            }
        }
        let Cache { slot_map, map, .. } = cache;
        self.dev.memcpy_htod(map.as_slice(), slot_map)?;
        Ok(())
    }

    /// LFU with margin: decay every count by `decay^rows`, count this step's routing, then give each missed
    /// expert the slot of the coldest resident not used this step if its count beats that one's by the
    /// margin. Sync uploads land before the next kernel; async ones are pending until `land`.
    fn admit_lfu(&self, cache: &mut Cache, ids: &[u32], missed: &[u32], rows: usize) -> Result<()> {
        let (decay, margin) = lfu_params();
        let f = decay.powi(rows as i32);
        cache.lfu.iter_mut().for_each(|c| *c *= f);
        for &e in ids {
            cache.lfu[e as usize] += 1.0;
        }
        if cache.owner.is_empty() || missed.is_empty() {
            return Ok(());
        }
        let tick = cache.tick;
        let eb = self.expert_bytes;
        let up = if async_uploads() { Some(titan_upload::uploader(&self.dev.cuda_stream())) } else { None };
        let mut room = up.map_or(usize::MAX, |u| u.room());
        let mut admitted: Vec<(usize, u32)> = Vec::new();
        for &e in missed {
            if room == 0 {
                break;
            }
            if cache.pending.iter().any(|p| p.1 == e) {
                continue;
            }
            let Cache { owner, last_use, lfu, .. } = &*cache;
            let Some(slot) = (0..owner.len())
                .filter(|&s| owner[s] != NOT_RESIDENT && last_use[s] != tick)
                .min_by(|&a, &b| lfu[owner[a] as usize].total_cmp(&lfu[owner[b] as usize]).then(a.cmp(&b)))
            else {
                break;
            };
            if lfu[e as usize] <= lfu[owner[slot] as usize] + margin {
                continue;
            }
            let old = cache.owner[slot];
            cache.map[old as usize] = NOT_RESIDENT;
            cache.last_use[slot] = tick;
            if up.is_some() {
                cache.owner[slot] = NOT_RESIDENT;
            } else {
                let src = self.host_expert(e as usize);
                let mut dst = cache.slots.slice_mut(slot * eb..(slot + 1) * eb);
                self.dev.memcpy_htod(src, &mut dst)?;
                cache.owner[slot] = e;
                cache.map[e as usize] = slot as u32;
            }
            crate::titan_monitor::tiered_upload_bytes(eb);
            admitted.push((slot, e));
            room -= 1;
        }
        if admitted.is_empty() {
            return Ok(());
        }
        if self.proj == "gate" {
            STATS[UPLOADS].fetch_add(admitted.len() as u64, std::sync::atomic::Ordering::Relaxed);
        }
        {
            let Cache { slot_map, map, .. } = &mut *cache;
            self.dev.memcpy_htod(map.as_slice(), slot_map)?;
        }
        if let Some(up) = up {
            // The copy may overwrite a victim's slot only after every kernel queued so far (the ones that
            // could still read it) and the slot-map update that stops later ones reading it.
            let after = Arc::new(self.dev.cuda_stream().record_event(None).w()?);
            for (slot, e) in admitted {
                let dst = cache.slots_ptr + (slot * eb) as u64;
                let done = up.submit(self.host_expert(e as usize), dst, after.clone());
                cache.pending.push((slot, e, done));
            }
        }
        Ok(())
    }

    /// Map async uploads whose copies have completed into their slots; the decode stream also waits on
    /// each copy's event before the slot-map update that exposes the slot.
    fn land(&self, cache: &mut Cache) -> Result<()> {
        if cache.pending.is_empty() {
            return Ok(());
        }
        let stream = self.dev.cuda_stream();
        let mut landed = 0u64;
        let mut i = 0;
        while i < cache.pending.len() {
            let Some(ev) = cache.pending[i].2.landed() else {
                i += 1;
                continue;
            };
            stream.wait(ev).w()?;
            let (slot, e, _) = cache.pending.swap_remove(i);
            cache.owner[slot] = e;
            cache.map[e as usize] = slot as u32;
            landed += 1;
        }
        if landed > 0 {
            let Cache { slot_map, map, .. } = cache;
            self.dev.memcpy_htod(map.as_slice(), slot_map)?;
            if self.proj == "gate" {
                STATS[LANDED].fetch_add(landed, std::sync::atomic::Ordering::Relaxed);
            }
        }
        Ok(())
    }

    /// Tasks routed to non-resident experts: compute on the CPU and overwrite the GPU's zeros,
    /// one task row at a time (stream-ordered after the gemv that left them zero).
    #[allow(clippy::too_many_arguments)]
    fn cpu_misses(
        &self,
        out: &mut CudaSlice<f32>,
        map: &[u32],
        ids_host: &[u32],
        xq: &CudaSlice<u8>,
        xq_host: Option<&[u8]>,
        input_dim1: usize,
        topk: usize,
        q8_row_bytes: usize,
    ) -> Result<()> {
        let tasks = out.len() / self.n;
        let misses: Vec<usize> = (0..tasks).filter(|&t| map[ids_host[t] as usize] == NOT_RESIDENT).collect();
        if misses.is_empty() {
            return Ok(());
        }
        let t0 = std::time::Instant::now();
        let fetched;
        let xq_host = match xq_host {
            Some(h) => h,
            None => {
                fetched = self.dev.clone_dtoh(xq)?;
                &fetched[..]
            }
        };
        add_time(1, t0);
        let t1 = std::time::Instant::now();
        let results: Vec<(usize, usize, Vec<f32>)> = Self::compute_misses(&[self], &[&misses], ids_host, xq_host, input_dim1, topk, q8_row_bytes)
            .into_iter()
            .map(|(_, t, r0, v)| (t, r0, v))
            .collect();
        add_time(2, t1);
        let t2 = std::time::Instant::now();
        self.upload_rows(out, results.into_iter())?;
        add_time(3, t2);
        if timing_on() {
            use std::sync::atomic::Ordering::Relaxed;
            if TIMING[4].fetch_add(1, Relaxed) % 2048 == 2047 {
                let ms = |i: usize| TIMING[i].load(Relaxed) as f64 / 1e6;
                tracing::info!(
                    "titan tiered timing: {} miss calls; ids fetch {:.0} ms, xq fetch {:.0} ms, cpu {:.0} ms ({} passes, {} tasks, {} expert groups, {} majflt, {} minflt), row upload {:.0} ms, gpu gemv {:.0} ms, lookahead submit {:.0} ms",
                    TIMING[4].load(Relaxed), ms(0), ms(1), ms(2), TIMING[PASSES].load(Relaxed), TIMING[PASSES + 1].load(Relaxed), TIMING[PASSES + 2].load(Relaxed),
                    TIMING[MAJFLT].load(Relaxed), TIMING[MINFLT].load(Relaxed), ms(3), ms(GPU_GEMV), ms(11)
                );
            }
        }
        Ok(())
    }
}
