//! Fork-join pool for the tiered experts' CPU misses. A decode step runs one or two short passes
//! (50-200 us) per MoE layer with GPU work in between; rayon parks its workers in those gaps and
//! re-waking them costs tens of microseconds per pass. These workers spin (`pause`) for a while after
//! each pass, then park until the next one.

use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering::SeqCst};
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
    pub fn new(workers: usize) -> Self {
        let spin_us = std::env::var("TITAN_TIERED_SPIN_US").ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_SPIN_US);
        let shared = Arc::new(Shared {
            gen: AtomicU64::new(0),
            job: AtomicPtr::new(std::ptr::null_mut()),
            retired: AtomicU64::new(0),
            active: AtomicUsize::new(0),
            sleepers: AtomicUsize::new(0),
            park: Mutex::new(()),
            wake: Condvar::new(),
            spin: Duration::from_micros(spin_us),
        });
        for i in 0..workers {
            let s = shared.clone();
            std::thread::Builder::new()
                .name(format!("titan-miss-{i}"))
                .spawn(move || worker(&s))
                .expect("spawn titan miss worker");
        }
        Self { shared, busy: Mutex::new(()) }
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
            let g = s.gen.load(SeqCst);
            if g != seen {
                break g;
            }
            spins += 1;
            if spins % 64 == 0 && t0.elapsed() > s.spin {
                let mut l = s.park.lock().unwrap();
                s.sleepers.fetch_add(1, SeqCst);
                while s.gen.load(SeqCst) == seen {
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
