//! titan-engine: prefill expert streaming (`TITAN_PFS=1`). A tiered MoE layer of a prompt chunk with at
//! least `TITAN_PFS_MIN_ROWS` rows (default 128) computes every routed expert on the GPU: residents from
//! their slots, the rest from a staging ring the non-resident experts stream through in a fixed order
//! (layer 0..L, gate/up/down, expert id ascending) on a copy stream, so the next layers' experts arrive
//! while this layer computes. With the slot-map GEMV both halves give the CPU twin's output bit for bit.
//! The kernel is llama.cpp's grouped MMQ by default (`TITAN_PFS_KERNEL=gemv` for the slot-map GEMV).
//! Ring: `TITAN_PFS_RING_MIB` (default 400) of device buffers, one projection's non-resident experts each,
//! allocated at the first streamed layer from what is free above `TITAN_PFS_HEADROOM_MIB` (default 1024)
//! plus `TITAN_PFS_TASK_KIB` (default 32) per routed (token, expert) task, and freed by the next non-streamed tiered forward, so decode sees exactly the static placement.
//! Owned host experts are page-locked in place (cuMemHostRegister); mapped ones (`TITAN_TIERED_MMAP=1`) are
//! copied through pinned bounce buffers by helper threads first. `TITAN_PFS_SHARE=f` streams only the
//! hottest share f of each tensor's non-resident experts (profile order); the CPU twin serves the rest.

use candle_core::{
    cuda::{
        cudarc::driver::{result, sys, CudaContext, CudaEvent, CudaSlice, CudaStream},
        CudaDevice, WrapErr,
    },
    Result,
};
use candle_core::cuda::cudarc::driver::DevicePtr;
use std::sync::{mpsc, Arc, Condvar, Mutex, OnceLock, Weak};

use super::titan_tiered::NOT_RESIDENT;

const DEFAULT_MIN_ROWS: usize = 128;
const DEFAULT_RING_MIB: usize = 400;
const DEFAULT_HEADROOM_MIB: usize = 1024;
/// Headroom per (token, expert) task on top (`TITAN_PFS_TASK_KIB`, 256 KiB per token at top-8): the
/// chunk's activations still to be allocated.
const DEFAULT_TASK_KIB: usize = 32;
/// Headroom per token of the prompt's KV room on top (`TITAN_PFS_CTX_KIB`): long-context attention
/// temporaries (a 28k prompt ran with 1 staging buffer where 4 ran out of memory).
const DEFAULT_CTX_KIB: usize = 10;
static CONTEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// The KV room (tokens) of the prompt pass about to run, for the ring's headroom.
pub(crate) fn set_context(tokens: usize) {
    CONTEXT.store(tokens, std::sync::atomic::Ordering::Relaxed);
}
const MAX_ENTRIES: usize = 64;
/// Pinned bounce buffers for unpinned (mapped) host experts, and the bytes each carries per copy.
const BOUNCE_BUFS: usize = 8;
const BOUNCE_BYTES: usize = 16 << 20;
/// Helper threads per bounce copy.
const DEFAULT_COPY_THREADS: usize = 4;
const PAGE: usize = 4096;
/// Slack after each staging buffer: MMQ tiles may read past a buffer's last expert.
const ENTRY_PAD: usize = 64 << 10;
/// Projection order within a layer: the order the MoE forward calls them in.
const PROJS: [&str; 3] = ["gate", "up", "down"];

fn env_usize(name: &str, default: usize) -> usize {
    crate::titan_cfg::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

pub(crate) fn enabled() -> bool {
    static ON: crate::titan_cfg::GenCell<bool> = crate::titan_cfg::GenCell::new();
    *ON.get_or_init(|| crate::titan_cfg::var("TITAN_PFS").is_ok_and(|v| v == "1"))
}

pub(crate) fn min_rows() -> usize {
    static N: crate::titan_cfg::GenCell<usize> = crate::titan_cfg::GenCell::new();
    *N.get_or_init(|| env_usize("TITAN_PFS_MIN_ROWS", DEFAULT_MIN_ROWS).max(1))
}

/// `TITAN_PFS_SHARE`: share of each tensor's non-resident experts that streams (default 1).
pub(crate) fn share() -> f64 {
    static S: crate::titan_cfg::GenCell<f64> = crate::titan_cfg::GenCell::new();
    *S.get_or_init(|| crate::titan_cfg::var("TITAN_PFS_SHARE").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(1.0).clamp(0.0, 1.0))
}

/// Streamed layers run llama.cpp's grouped MMQ (Q4_K/Q5_K/Q6_K; tensors of other dtypes take the GEMV).
/// `TITAN_PFS_KERNEL=gemv`: the slot-map GEMV instead, bit-identical to the CPU twin but ~3x slower; MMQ's
/// int8 tiles round differently from both.
pub(crate) fn mmq() -> bool {
    static ON: crate::titan_cfg::GenCell<bool> = crate::titan_cfg::GenCell::new();
    *ON.get_or_init(|| crate::titan_cfg::var("TITAN_PFS_KERNEL").map_or(true, |v| v != "gemv"))
}

fn timing_on() -> bool {
    static ON: crate::titan_cfg::GenCell<bool> = crate::titan_cfg::GenCell::new();
    *ON.get_or_init(|| crate::titan_cfg::var("TITAN_PFS_TIMING").is_ok_and(|v| v == "1"))
}

/// One tensor's streamed experts: their host bytes in staging order and the device map to their
/// staging index.
pub struct Source {
    layer: usize,
    proj: usize,
    /// (host address, length), adjacent experts merged.
    ranges: Vec<(usize, usize)>,
    bytes: usize,
    /// The ranges lie in page-locked memory (registered in place): DMA straight from them.
    pinned: bool,
    /// Registered host range to unregister on drop, and the context it was registered in.
    registered: Option<usize>,
    ctx: Arc<CudaContext>,
    /// expert -> staging index, NOT_RESIDENT for residents and for experts left to the CPU twin.
    pub(crate) stage_map: CudaSlice<u32>,
    /// Host mirror of `stage_map`.
    pub(crate) stage_index: Vec<u32>,
    pub(crate) staged: usize,
}

// The host addresses point into a store (owned buffer or GGUF mmap) that outlives the Source: the
// TieredExperts holding both drops the Source first.
unsafe impl Send for Source {}
unsafe impl Sync for Source {}

impl Drop for Source {
    fn drop(&mut self) {
        if let Some(addr) = self.registered {
            // the dropping thread (a model unload) may have no current context
            let _ = self.ctx.bind_to_thread();
            let r = unsafe { sys::cuMemHostUnregister(addr as *mut std::ffi::c_void) };
            if r != sys::CUresult::CUDA_SUCCESS {
                tracing::warn!("titan pfs: cuMemHostUnregister failed ({r:?})");
            }
        }
    }
}

static REGISTRY: Mutex<Vec<Weak<Source>>> = Mutex::new(Vec::new());

/// Register a tensor for streaming. `staged` lists the experts to stream (non-resident, hottest first),
/// `expert` their host bytes; `owned` is the whole owned host store to page-lock in place, if any.
pub(crate) fn register<'a>(
    dev: &CudaDevice,
    layer: usize,
    proj: &str,
    num_experts: usize,
    staged: &[usize],
    expert: impl Fn(usize) -> &'a [u8],
    owned: Option<&'a [u8]>,
) -> Result<Option<Arc<Source>>> {
    let Some(proj) = PROJS.iter().position(|p| *p == proj) else {
        return Ok(None);
    };
    // With MMQ an all-resident tensor streams nothing but still runs its prefill as grouped MMQ.
    if staged.is_empty() && !mmq() {
        return Ok(None);
    }
    let mut order = staged.to_vec();
    order.sort_unstable();
    let mut stage_index = vec![NOT_RESIDENT; num_experts];
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    for (i, &e) in order.iter().enumerate() {
        stage_index[e] = i as u32;
        let b = expert(e);
        let a = b.as_ptr() as usize;
        match ranges.last_mut() {
            Some(r) if r.0 + r.1 == a => r.1 += b.len(),
            _ => ranges.push((a, b.len())),
        }
    }
    let bytes = ranges.iter().map(|r| r.1).sum();
    let mut registered = None;
    // `TITAN_PFS_PIN=0`: leave the owned host store pageable (bounce copies).
    let pin = crate::titan_cfg::var("TITAN_PFS_PIN").map_or(true, |v| v != "0");
    if let Some(buf) = owned.filter(|_| pin && !staged.is_empty()) {
        // The enclosing pages: a large Vec is its own mapping, so they belong to it.
        let a0 = buf.as_ptr() as usize & !(PAGE - 1);
        let a1 = (buf.as_ptr() as usize + buf.len()).next_multiple_of(PAGE);
        dev.cuda_stream().context().bind_to_thread().w()?;
        let r = unsafe { sys::cuMemHostRegister_v2(a0 as *mut std::ffi::c_void, a1 - a0, 0) };
        if r == sys::CUresult::CUDA_SUCCESS {
            registered = Some(a0);
        } else {
            static WARNED: OnceLock<()> = OnceLock::new();
            WARNED.get_or_init(|| tracing::warn!("titan pfs: cuMemHostRegister failed ({r:?}); host experts go through bounce buffers"));
        }
    }
    let stage_map = dev.clone_htod(&stage_index)?;
    let src = Arc::new(Source {
        layer,
        proj,
        ranges,
        bytes,
        pinned: registered.is_some(),
        registered,
        ctx: dev.cuda_stream().context().clone(),
        stage_map,
        stage_index,
        staged: order.len(),
    });
    let mut reg = REGISTRY.lock().unwrap();
    reg.retain(|w| w.strong_count() > 0);
    reg.push(Arc::downgrade(&src));
    Ok(Some(src))
}

/// Set once the worker has issued a job's copies: the event recorded after them on the copy stream.
struct Done {
    ev: Mutex<Option<Arc<CudaEvent>>>,
    cv: Condvar,
}

impl Done {
    fn wait(&self) -> Arc<CudaEvent> {
        let mut g = self.ev.lock().unwrap();
        while g.is_none() {
            g = self.cv.wait(g).unwrap();
        }
        g.clone().expect("set")
    }
}

struct Job {
    src: Arc<Source>,
    dst: u64,
    after: Arc<CudaEvent>,
    done: Arc<Done>,
}

struct Worker {
    tx: mpsc::Sender<Job>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    copy: Arc<CudaStream>,
    /// TITAN_PFS_TIMING=1: (start, end) timing events of each job's copies.
    timed: Arc<Mutex<Vec<(CudaEvent, CudaEvent)>>>,
}

/// The copy worker; dropped (thread joined, bounce buffers freed) by `release`.
static WORKER: Mutex<Option<Arc<Worker>>> = Mutex::new(None);

fn worker(ctx: &Arc<CudaContext>) -> Result<Arc<Worker>> {
    let mut g = WORKER.lock().unwrap();
    if let Some(w) = g.as_ref() {
        return Ok(w.clone());
    }
    ctx.bind_to_thread().w()?;
    let copy = ctx.new_stream().w()?;
    let (tx, rx) = mpsc::channel::<Job>();
    let timed = Arc::new(Mutex::new(Vec::new()));
    let (c, s, t) = (ctx.clone(), copy.clone(), timed.clone());
    let thread = std::thread::Builder::new()
        .name("titan-pfs".into())
        .spawn(move || run_worker(c, s, rx, t))
        .map_err(candle_core::Error::wrap)?;
    let w = Arc::new(Worker { tx, thread: Mutex::new(Some(thread)), copy, timed });
    *g = Some(w.clone());
    Ok(w)
}

/// Model unload: drop the staging ring (its buffers and its references to the host stores) once its copies
/// are done, and stop the copy worker, which frees its pinned bounce buffers.
pub(crate) fn release_all() {
    let ring = RING.lock().unwrap().take();
    STATE.store(NONE, std::sync::atomic::Ordering::Release);
    if let Some(ring) = ring {
        for e in &ring.entries {
            if let Some(d) = &e.done {
                let _ = d.wait();
            }
        }
        if let Some(w) = WORKER.lock().unwrap().as_ref() {
            let _ = w.copy.synchronize();
        }
        drop(ring);
    }
    let w = WORKER.lock().unwrap().take();
    if let Some(w) = w {
        let _ = w.copy.synchronize();
        let thread = w.thread.lock().unwrap().take();
        drop(w);
        if let Some(t) = thread {
            let _ = t.join();
        }
    }
    CONTEXT.store(0, std::sync::atomic::Ordering::Relaxed);
}

struct Bounce {
    ptr: *mut u8,
    copied: Option<CudaEvent>,
}

fn run_worker(ctx: Arc<CudaContext>, copy: Arc<CudaStream>, rx: mpsc::Receiver<Job>, timed: Arc<Mutex<Vec<(CudaEvent, CudaEvent)>>>) {
    if let Err(e) = ctx.bind_to_thread() {
        tracing::warn!("titan pfs: copy worker cannot bind the context ({e})");
        return;
    }
    let mut bounce: Vec<Bounce> = Vec::new();
    let mut next = 0usize;
    let threads = env_usize("TITAN_PFS_COPY_THREADS", DEFAULT_COPY_THREADS).max(1);
    while let Ok(job) = rx.recv() {
        let r = issue(&ctx, &copy, &job, &mut bounce, &mut next, threads, &timed);
        let ev = match r {
            Ok(ev) => ev,
            Err(e) => {
                // The compute stream still waits on a recorded event; the output is then wrong, so say so.
                tracing::error!("titan pfs: expert copy failed ({e:?})");
                Arc::new(copy.record_event(None).expect("record event"))
            }
        };
        *job.done.ev.lock().unwrap() = Some(ev);
        job.done.cv.notify_all();
    }
    for b in bounce {
        if let Some(ev) = b.copied {
            let _ = ev.synchronize();
        }
        unsafe {
            let _ = result::free_host(b.ptr as _);
        }
    }
}

fn issue(
    ctx: &Arc<CudaContext>,
    copy: &Arc<CudaStream>,
    job: &Job,
    bounce: &mut Vec<Bounce>,
    next: &mut usize,
    threads: usize,
    timed: &Mutex<Vec<(CudaEvent, CudaEvent)>>,
) -> std::result::Result<Arc<CudaEvent>, sys::CUresult> {
    let d = |e: result::DriverError| e.0;
    copy.wait(&job.after).map_err(d)?;
    let t0 = if timing_on() { Some(copy.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).map_err(d)?) } else { None };
    let mut off = 0u64;
    if job.src.pinned {
        for &(a, len) in &job.src.ranges {
            let src = unsafe { std::slice::from_raw_parts(a as *const u8, len) };
            unsafe { result::memcpy_htod_async(job.dst + off, src, copy.cu_stream()) }.map_err(d)?;
            off += len as u64;
        }
    } else {
        // Pieces of at most BOUNCE_BYTES, each gathered into a pinned bounce buffer by helper threads.
        let mut pieces: Vec<Vec<(usize, usize)>> = vec![Vec::new()];
        let mut fill = 0usize;
        for &(a, len) in &job.src.ranges {
            let mut o = 0;
            while o < len {
                if fill == BOUNCE_BYTES {
                    pieces.push(Vec::new());
                    fill = 0;
                }
                let take = (len - o).min(BOUNCE_BYTES - fill);
                pieces.last_mut().expect("piece").push((a + o, take));
                fill += take;
                o += take;
            }
        }
        for piece in pieces {
            if bounce.len() < BOUNCE_BUFS {
                let ptr = unsafe { result::malloc_host(BOUNCE_BYTES, 0) }.map_err(d)? as *mut u8;
                bounce.push(Bounce { ptr, copied: None });
            }
            let nb = bounce.len();
            let b = &mut bounce[*next % nb];
            *next += 1;
            if let Some(ev) = b.copied.take() {
                ev.synchronize().map_err(d)?;
            }
            let len: usize = piece.iter().map(|p| p.1).sum();
            gather(&piece, b.ptr, threads);
            let staged = unsafe { std::slice::from_raw_parts(b.ptr as *const u8, len) };
            unsafe { result::memcpy_htod_async(job.dst + off, staged, copy.cu_stream()) }.map_err(d)?;
            b.copied = Some(copy.record_event(None).map_err(d)?);
            off += len as u64;
        }
    }
    let ev = copy.record_event(None).map_err(d)?;
    if let Some(t0) = t0 {
        let t1 = copy.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).map_err(d)?;
        timed.lock().unwrap().push((t0, t1));
    }
    let _ = ctx;
    Ok(Arc::new(ev))
}

/// Copy `ranges` back to back into `dst`, split across `threads` helper threads (page faults of a mapped
/// store are taken there, off the launching thread).
fn gather(ranges: &[(usize, usize)], dst: *mut u8, threads: usize) {
    let total: usize = ranges.iter().map(|r| r.1).sum();
    let per = total.div_ceil(threads).next_multiple_of(PAGE).max(PAGE);
    let mut spans: Vec<(usize, usize, usize)> = Vec::new(); // (src, dst offset, len)
    let mut o = 0usize;
    for &(a, len) in ranges {
        spans.push((a, o, len));
        o += len;
    }
    let dst = dst as usize;
    std::thread::scope(|s| {
        for t in 0..threads {
            let (lo, hi) = (t * per, ((t + 1) * per).min(total));
            if lo >= hi {
                break;
            }
            let spans = &spans;
            s.spawn(move || {
                for &(a, d0, len) in spans {
                    let (s0, s1) = (d0.max(lo), (d0 + len).min(hi));
                    if s0 < s1 {
                        unsafe { std::ptr::copy_nonoverlapping((a + s0 - d0) as *const u8, (dst + s0) as *mut u8, s1 - s0) };
                    }
                }
            });
        }
    });
}

struct Entry {
    buf: CudaSlice<u8>,
    ptr: u64,
    /// Sequence number of the item copied into it, and that copy.
    seq: usize,
    done: Option<Arc<Done>>,
    /// Recorded on the compute stream after the last kernel that read it.
    used: Arc<CudaEvent>,
}

struct Ring {
    order: Vec<Arc<Source>>,
    entries: Vec<Entry>,
    /// Next item the compute side takes, and next item to copy (sequence numbers; item = seq % order.len()).
    cursor: usize,
    issued: usize,
    started: std::time::Instant,
    items: u64,
    bytes: u64,
    resyncs: u64,
    wait_ns: u64,
    /// Routed / staged experts over the streamed forwards of gate projections.
    routed: u64,
    staged: u64,
}

static RING: Mutex<Option<Ring>> = Mutex::new(None);
/// NONE until a streamed forward allocates the ring (LIVE) or finds no room for it (FAILED, until the next
/// non-streamed forward: the rest of that prompt takes the CPU twin path).
static STATE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(NONE);
const NONE: u8 = 0;
const LIVE: u8 = 1;
const FAILED: u8 = 2;

/// Memory the device can still hand out: free memory plus what the stream-ordered pool holds unused.
fn free_bytes(ctx: &Arc<CudaContext>) -> usize {
    let (free, _) = result::mem_get_info().unwrap_or((0, 0));
    let mut pool = std::ptr::null_mut();
    if !ctx.has_async_alloc() || unsafe { sys::cuDeviceGetMemPool(&mut pool, ctx.cu_device()) } != sys::CUresult::CUDA_SUCCESS {
        return free;
    }
    let get = |attr| {
        let mut v = 0u64;
        unsafe { sys::cuMemPoolGetAttribute(pool, attr, (&mut v as *mut u64).cast()) };
        v as usize
    };
    use sys::CUmemPool_attribute::*;
    free + get(CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT).saturating_sub(get(CU_MEMPOOL_ATTR_USED_MEM_CURRENT))
}

fn allocate(dev: &CudaDevice, tasks: usize) -> Result<Option<Ring>> {
    let mut order: Vec<Arc<Source>> =
        REGISTRY.lock().unwrap().iter().filter_map(|w| w.upgrade()).filter(|s| s.bytes > 0).collect();
    order.sort_by_key(|s| (s.layer, s.proj));
    let Some(entry_bytes) = order.iter().map(|s| s.bytes).max() else {
        return Ok(None);
    };
    let stream = dev.cuda_stream();
    let ctx = stream.context().clone();
    ctx.bind_to_thread().w()?;
    let want = (env_usize("TITAN_PFS_RING_MIB", DEFAULT_RING_MIB) << 20) / entry_bytes;
    let headroom = (env_usize("TITAN_PFS_HEADROOM_MIB", DEFAULT_HEADROOM_MIB) << 20) + tasks * (env_usize("TITAN_PFS_TASK_KIB", DEFAULT_TASK_KIB) << 10)
        + CONTEXT.load(std::sync::atomic::Ordering::Relaxed) * (env_usize("TITAN_PFS_CTX_KIB", DEFAULT_CTX_KIB) << 10);
    let free = free_bytes(&ctx);
    let fit = free.saturating_sub(headroom) / (entry_bytes + ENTRY_PAD);
    let n = want.min(fit).clamp(0, MAX_ENTRIES);
    if n == 0 {
        tracing::warn!(
            "titan pfs: no room for a staging buffer ({} MiB free, {} MiB headroom, {} MiB per buffer); CPU twin prefill",
            free >> 20,
            headroom >> 20,
            entry_bytes >> 20
        );
        return Ok(None);
    }
    let mut bufs = Vec::with_capacity(n);
    for _ in 0..n {
        match unsafe { dev.alloc::<u8>(entry_bytes + ENTRY_PAD) } {
            Ok(b) => bufs.push(b),
            Err(_) => break,
        }
    }
    if bufs.is_empty() {
        return Ok(None);
    }
    worker(&ctx)?;
    // The copy stream may write a buffer only after its stream-ordered allocation.
    let alloc = Arc::new(stream.record_event(None).w()?);
    static LOGGED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
        tracing::info!(
            "titan pfs: ring of {} x {} MiB ({} tensors, {} MiB per pass, {} pinned), {} MiB was free",
            bufs.len(),
            entry_bytes >> 20,
            order.len(),
            order.iter().map(|s| s.bytes).sum::<usize>() >> 20,
            order.iter().filter(|s| s.pinned).count(),
            free >> 20
        );
    }
    let entries = bufs
        .into_iter()
        .map(|buf| {
            let ptr = buf.device_ptr(&stream).0;
            Entry { buf, ptr, seq: usize::MAX, done: None, used: alloc.clone() }
        })
        .collect();
    Ok(Some(Ring {
        order,
        entries,
        cursor: 0,
        issued: 0,
        started: std::time::Instant::now(),
        items: 0,
        bytes: 0,
        resyncs: 0,
        wait_ns: 0,
        routed: 0,
        staged: 0,
    }))
}

impl Ring {
    fn issue_until(&mut self, end: usize, w: &Worker) -> Result<()> {
        while self.issued < end {
            let seq = self.issued;
            let src = self.order[seq % self.order.len()].clone();
            let r = self.entries.len();
            let e = &mut self.entries[seq % r];
            let dst = e.ptr;
            let done = Arc::new(Done { ev: Mutex::new(None), cv: Condvar::new() });
            self.bytes += src.bytes as u64;
            w.tx.send(Job { src, dst, after: e.used.clone(), done: done.clone() }).map_err(|_| candle_core::Error::msg("titan pfs: copy worker gone"))?;
            e.seq = seq;
            e.done = Some(done);
            self.issued += 1;
        }
        Ok(())
    }
}

/// A staged tensor's experts on the device, lent until `release`.
pub(crate) struct Staged {
    pub(crate) ptr: u64,
    pub(crate) len: usize,
    entry: usize,
}

/// The staging buffer holding `src`'s experts, with the compute stream ordered after its copy; `None` if
/// no ring could be allocated (the caller then takes the CPU twin path).
pub(crate) fn acquire(dev: &CudaDevice, src: &Arc<Source>, tasks: usize) -> Result<Option<Staged>> {
    let mut g = RING.lock().unwrap();
    if g.is_none() {
        if STATE.load(std::sync::atomic::Ordering::Acquire) == FAILED {
            return Ok(None);
        }
        *g = allocate(dev, tasks)?;
        STATE.store(if g.is_some() { LIVE } else { FAILED }, std::sync::atomic::Ordering::Release);
    }
    let Some(ring) = g.as_mut() else {
        return Ok(None);
    };
    let len = ring.order.len();
    let Some(idx) = ring.order.iter().position(|s| Arc::ptr_eq(s, src)) else {
        return Ok(None);
    };
    let stream = dev.cuda_stream();
    let w = worker(stream.context())?;
    let r = ring.entries.len();
    if ring.cursor % len != idx {
        // Out of the fixed order (a new pass, or a layer that did not stream): restart the sequence at
        // this item. Copies already queued into an entry stay ordered before new ones on the copy stream.
        ring.resyncs += u64::from(ring.issued > 0);
        let base = (ring.issued.div_ceil(len) + 1) * len + idx;
        ring.cursor = base;
        ring.issued = base;
    }
    ring.issue_until(ring.cursor + r, &w)?;
    let e = ring.cursor % r;
    let entry = &ring.entries[e];
    debug_assert_eq!(entry.seq, ring.cursor);
    let t0 = std::time::Instant::now();
    let ev = entry.done.as_ref().expect("issued").wait();
    ring.wait_ns += t0.elapsed().as_nanos() as u64;
    stream.wait(&ev).w()?;
    Ok(Some(Staged { ptr: entry.ptr, len: entry.buf.len(), entry: e }))
}

/// The kernels reading `st` are queued: the buffer may take the next copy once they have run.
pub(crate) fn release(dev: &CudaDevice, st: Staged, routed: Option<(usize, usize)>) -> Result<()> {
    let mut g = RING.lock().unwrap();
    let Some(ring) = g.as_mut() else {
        return Ok(());
    };
    let stream = dev.cuda_stream();
    ring.entries[st.entry].used = Arc::new(stream.record_event(None).w()?);
    ring.cursor += 1;
    ring.items += 1;
    if let Some((r, s)) = routed {
        ring.routed += r as u64;
        ring.staged += s as u64;
    }
    let w = worker(stream.context())?;
    let end = ring.cursor + ring.entries.len();
    ring.issue_until(end, &w)
}

/// A tiered forward of `rows` tokens that does not stream: free the ring (once its copies are done) so
/// decode runs with the static placement's memory. A prompt chunk that could not get a ring leaves it so.
pub(crate) fn idle(dev: &CudaDevice, rows: usize) -> Result<()> {
    let state = STATE.load(std::sync::atomic::Ordering::Acquire);
    if state == NONE || (state == FAILED && rows >= min_rows()) {
        return Ok(());
    }
    let mut g = RING.lock().unwrap();
    STATE.store(NONE, std::sync::atomic::Ordering::Release);
    let Some(ring) = g.take() else {
        return Ok(());
    };
    let stream = dev.cuda_stream();
    let w = worker(stream.context())?;
    for e in &ring.entries {
        if let Some(d) = &e.done {
            let _ = d.wait();
        }
    }
    // Frees are ordered on the compute stream: after every copy into the buffers.
    stream.wait(&w.copy.record_event(None).w()?).w()?;
    let mut copy_ms = 0f32;
    let mut timed = w.timed.lock().unwrap();
    for (a, b) in timed.drain(..) {
        copy_ms += a.elapsed_ms(&b).unwrap_or(0.0);
    }
    drop(timed);
    let secs = ring.started.elapsed().as_secs_f64();
    tracing::info!(
        "titan pfs: pass of {:.2} s: {} streamed forwards, {} MiB copied ({:.1} GB/s over the pass{}), host waited {:.0} ms for copies to be issued, {} resyncs, gate staged experts routed {:.1}%",
        secs,
        ring.items,
        ring.bytes >> 20,
        ring.bytes as f64 / secs.max(1e-9) / 1e9,
        if copy_ms > 0.0 { format!(", copy engine busy {copy_ms:.0} ms = {:.1} GB/s", ring.bytes as f64 / (copy_ms as f64 * 1e6)) } else { String::new() },
        ring.wait_ns as f64 / 1e6,
        ring.resyncs,
        100.0 * ring.routed as f64 / ring.staged.max(1) as f64
    );
    drop(ring);
    Ok(())
}
