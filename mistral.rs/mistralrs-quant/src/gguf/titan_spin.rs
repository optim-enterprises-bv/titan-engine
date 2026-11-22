//! Fork-join pool for the tiered experts' CPU misses. A decode step runs one or two short passes
//! (50-200 us) per MoE layer with GPU work in between; rayon parks its workers in those gaps and
//! re-waking them costs tens of microseconds per pass. These workers spin (`pause`) for a while after
//! each pass, then park until the next one.

use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering::SeqCst};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// How long an idle worker spins before parking (`TITAN_TIERED_SPIN_US`).
const DEFAULT_SPIN_US: u64 = 2000;

struct Job<'a> {
    f: &'a (dyn Fn(usize) + Sync),
    n: usize,
    next: AtomicUsize,
}

impl Job<'_> {
    fn work(&self) {
        loop {
            let i = self.next.fetch_add(1, SeqCst);
            if i >= self.n {
                return;
            }
            (self.f)(i);
        }
    }
}

struct Shared {
    /// Generation of the published job; odd values never occur (bumped by 2 per job).
    gen: AtomicU64,
    /// The job of generation `gen` (a `*const Job` whose lifetime `run` guarantees).
    job: AtomicPtr<()>,
    /// Generation whose job may no longer be entered.
    retired: AtomicU64,
    /// Workers inside a job.
    active: AtomicUsize,
    sleepers: AtomicUsize,
    /// Set by `shutdown`: workers exit.
    stop: AtomicBool,
    park: Mutex<()>,
    wake: Condvar,
    spin: Duration,
}

pub struct SpinPool {
    shared: Arc<Shared>,
    /// One job at a time: the decode thread and the doorbell thread may both run passes.
    busy: Mutex<()>,
}

impl SpinPool {
    /// `cpus`: pin worker i to `cpus[(i + 1) % len]` (the caller of `run` takes `cpus[0]`, see `pin_current_thread`).
    pub fn new(workers: usize, cpus: Option<&'static [usize]>) -> Self {
        let spin_us = crate::titan_cfg::var("TITAN_TIERED_SPIN_US").ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_SPIN_US);
        let shared = Arc::new(Shared {
            gen: AtomicU64::new(0),
            job: AtomicPtr::new(std::ptr::null_mut()),
            retired: AtomicU64::new(0),
            active: AtomicUsize::new(0),
            sleepers: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            park: Mutex::new(()),
            wake: Condvar::new(),
            spin: Duration::from_micros(spin_us),
        });
        for i in 0..workers {
            let s = shared.clone();
            std::thread::Builder::new()
                .name(format!("titan-miss-{i}"))
                .spawn(move || {
                    if let Some(c) = cpus {
                        pin_current_thread(c[(i + 1) % c.len()]);
                    }
                    worker(&s)
                })
                .expect("spawn titan miss worker");
        }
        Self { shared, busy: Mutex::new(()) }
    }

    /// Stop the workers (model unload); `run` must not be called afterwards.
    pub fn shutdown(&self) {
        let _one = self.busy.lock().unwrap();
        self.shared.stop.store(true, SeqCst);
        let _l = self.shared.park.lock().unwrap();
        self.shared.wake.notify_all();
    }

    /// `f(0..n)` on the workers and the calling thread; returns once every call has finished.
    pub fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        let _one = self.busy.lock().unwrap();
        let s = &*self.shared;
        let job = Job { f, n, next: AtomicUsize::new(0) };
        s.job.store(&job as *const Job as *mut (), SeqCst);
        let g = s.gen.fetch_add(2, SeqCst) + 2;
        if s.sleepers.load(SeqCst) > 0 {
            let _l = s.park.lock().unwrap();
            s.wake.notify_all();
        }
        job.work();
        // No worker may enter `job` after this; wait for the ones inside it.
        s.retired.store(g, SeqCst);
        while s.active.load(SeqCst) != 0 {
            std::hint::spin_loop();
        }
    }
}

fn worker(s: &Shared) {
    let mut seen = 0u64;
    loop {
        let t0 = Instant::now();
        let mut spins = 0u32;
        let g = loop {
            if s.stop.load(SeqCst) {
                return;
            }
            let g = s.gen.load(SeqCst);
            if g != seen {
                break g;
            }
            spins += 1;
            if spins % 64 == 0 && t0.elapsed() > s.spin {
                let mut l = s.park.lock().unwrap();
                s.sleepers.fetch_add(1, SeqCst);
                while s.gen.load(SeqCst) == seen && !s.stop.load(SeqCst) {
                    l = s.wake.wait(l).unwrap();
                }
                s.sleepers.fetch_sub(1, SeqCst);
                break s.gen.load(SeqCst);
            }
            std::hint::spin_loop();
        };
        seen = g;
        s.active.fetch_add(1, SeqCst);
        if s.retired.load(SeqCst) < g && s.gen.load(SeqCst) == g {
            // SAFETY: generation `g` is not retired, so `run` is still waiting for us to leave its job.
            let job = unsafe { &*(s.job.load(SeqCst) as *const Job) };
            job.work();
        }
        s.active.fetch_sub(1, SeqCst);
    }
}

/// Pin the calling thread to one CPU (best effort).
pub fn pin_current_thread(cpu: usize) {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) != 0 {
            tracing::warn!("titan: could not pin a thread to CPU {cpu}");
        }
    }
}

/// `0-7,12,14-15` -> the CPUs listed.
fn parse_cpus(s: &str) -> Option<Vec<usize>> {
    let mut v = Vec::new();
    for part in s.trim().split(',').filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((a, b)) => v.extend(a.trim().parse::<usize>().ok()?..=b.trim().parse::<usize>().ok()?),
            None => v.push(part.trim().parse().ok()?),
        }
    }
    (!v.is_empty()).then_some(v)
}

/// `TITAN_TIERED_CPUS`: the CPUs the CPU-expert threads (the doorbell thread and the miss pool) are pinned to:
/// a list (`0-7,12`), `auto` (a hybrid Intel CPU's performance cores, `/sys/devices/cpu_core/cpus`) or `none`
/// (default). In isolation a decode-sized pass ran 0.150 ms on 14 mixed threads and 0.110 ms on the 8 P-cores of a
/// Core Ultra 9 285HX, but in the server the pass is bound by DRAM bandwidth (~40 GB/s of expert weights) and
/// pinning measured the same (0.071 against 0.068 ms per decode layer), so it stays off.
pub fn miss_cpus() -> Option<&'static [usize]> {
    static C: crate::titan_cfg::GenCell<Option<Vec<usize>>> = crate::titan_cfg::GenCell::new();
    C.get_or_init(|| {
        let v = crate::titan_cfg::var("TITAN_TIERED_CPUS").unwrap_or_else(|_| "none".into());
        let cpus = match v.trim() {
            "none" | "0" | "" => None,
            "auto" => std::fs::read_to_string("/sys/devices/cpu_core/cpus").ok().and_then(|s| parse_cpus(&s)),
            list => parse_cpus(list),
        };
        let avail = std::thread::available_parallelism().map_or(1, |n| n.get());
        let cpus = cpus.filter(|c| c.len() >= 2 && c.iter().all(|&x| x < avail.max(1024)));
        if let Some(c) = &cpus {
            tracing::info!("titan cpu experts: pinned to CPUs {c:?}");
        }
        cpus
    })
    .as_deref()
}
