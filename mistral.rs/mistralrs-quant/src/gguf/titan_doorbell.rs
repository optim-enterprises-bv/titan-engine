//! `TITAN_DOORBELL=1`: a MoE layer's CPU misses without a host sync. A one-block kernel copies the routing ids (and
//! any lookahead predictions) and the q8_1 expert input into pinned, mapped host memory and bumps a sequence word
//! there; the decode thread queues a job for the `titan-doorbell` thread and goes on launching. That thread spins on
//! the word, computes the misses (`titan_onepass`, or gate/up then down with the GPU's SwiGLU in between when
//! `TITAN_CPU_ONEPASS=0`), writes the rows into mapped memory and answers with the sequence number. On the GPU a
//! one-thread kernel waits for the answer and a scatter kernel copies the rows into the expert output. The GPU runs
//! the resident experts meanwhile, and the decode thread never waits on the GPU inside the layer.
//! One set of buffers serves every layer: the stream orders a layer's publish after the previous layer's scatter,
//! and the host thread takes jobs in queue order. Kernel arguments are fixed pointers (the sequence number lives in
//! device memory), so the launches are capture-safe.

use candle_core::{
    cuda::{
        cudarc::driver::{result, sys, CudaSlice, LaunchConfig, PushKernelArg},
        CudaDevice, WrapErr,
    },
    Result, Storage, Tensor,
};
use std::collections::VecDeque;
use std::sync::atomic::{fence, AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};

use super::onepass::{self, Misses, Out};
use super::{titan_pfs, TieredExperts, MAX_LOOKAHEAD, NOT_RESIDENT};
use crate::utils::slice_ptr;

/// oxide-kernels/titan-doorbell (cuda-oxide).
const PTX: &str = include_str!("titan_doorbell_oxide.ptx");
const MODULE: &str = "titan_doorbell_oxide";
/// Largest forward (tokens) the doorbell takes; larger ones run the synchronous path.
pub(super) const MAX_ROWS: usize = 64;
/// Word offsets in the control block (oxide-kernels/titan-doorbell), each on its own cache line.
const CTL_PUB: usize = 0;
const CTL_DONE: usize = 16;
const CTL_ERR: usize = 32;
const REGION_ALIGN: usize = 4096;
/// The GPU gives up waiting for the host after this long and reports it through the error word.
const WAIT_TIMEOUT_NS: u64 = 60_000_000_000;
/// The host thread drops a job whose publish never arrives after this long.
const PUB_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);
const PUBLISH_THREADS: u32 = 256;
const SCATTER_THREADS: u32 = 256;
const DEFAULT_COLLECT_BLOCKS: u32 = 1;
const DEFAULT_COLLECT_THREADS: u32 = 512;
/// The scatter's blocks stride over the rows; each reads the header from host memory once.
const SCATTER_MAX_BLOCKS: usize = 32;
/// MAX_TASKS in oxide-kernels/titan-doorbell: the scatter's shared task list.
const SCATTER_MAX_TASKS: usize = 1024;
/// How long the host thread spins for the next job before parking (`TITAN_TIERED_SPIN_US`, as the miss pool).
const DEFAULT_SPIN_US: u64 = 2000;

/// `TITAN_DOORBELL_COLLECT=blocks[,threads]` (default 1,512): the shape of the wait-and-copy kernel; 0 = a one-thread
/// wait kernel, then a separate scatter.
fn collect_shape() -> (u32, u32) {
    static S: crate::titan_cfg::GenCell<(u32, u32)> = crate::titan_cfg::GenCell::new();
    *S.get_or_init(|| {
        let v = crate::titan_cfg::var("TITAN_DOORBELL_COLLECT").unwrap_or_default();
        let mut it = v.split(',').map(|x| x.trim().parse::<u32>().ok());
        let blocks = it.next().flatten().unwrap_or(DEFAULT_COLLECT_BLOCKS);
        let threads = it.next().flatten().unwrap_or(DEFAULT_COLLECT_THREADS).clamp(32, 1024);
        (blocks, threads)
    })
}

pub(crate) fn enabled() -> bool {
    static ON: crate::titan_cfg::GenCell<bool> = crate::titan_cfg::GenCell::new();
    *ON.get_or_init(|| crate::titan_cfg::var("TITAN_DOORBELL").is_ok_and(|v| v == "1"))
}

/// Byte offsets of the mapped regions, sized for the first layer's shapes.
#[derive(Clone, Copy)]
struct Layout {
    topk: usize,
    xq_row: usize,
    qa_row: usize,
    ng: usize,
    nd: usize,
    ids: usize,
    xq: usize,
    qa: usize,
    hdr_gu: usize,
    rows_g: usize,
    rows_u: usize,
    hdr_d: usize,
    rows_d: usize,
    len: usize,
}

impl Layout {
    fn new(topk: usize, xq_row: usize, qa_row: usize, ng: usize, nd: usize) -> Self {
        let tasks = MAX_ROWS * topk;
        let mut off = REGION_ALIGN;
        let mut take = |bytes: usize| {
            let o = off;
            off += bytes.div_ceil(REGION_ALIGN) * REGION_ALIGN;
            o
        };
        let ids = take(4 * tasks * (1 + MAX_LOOKAHEAD));
        let xq = take(MAX_ROWS * xq_row);
        let qa = take(tasks * qa_row);
        let hdr_gu = take(4 * (1 + tasks));
        let rows_g = take(4 * tasks * ng);
        let rows_u = take(4 * tasks * ng);
        let hdr_d = take(4 * (1 + tasks));
        let rows_d = take(4 * tasks * nd);
        Self { topk, xq_row, qa_row, ng, nd, ids, xq, qa, hdr_gu, rows_g, rows_u, hdr_d, rows_d, len: off }
    }

    fn fits(&self, topk: usize, xq_row: usize, qa_row: usize, ng: usize, nd: usize) -> bool {
        topk <= self.topk && xq_row <= self.xq_row && qa_row <= self.qa_row && ng <= self.ng && nd <= self.nd
    }
}

/// One layer's host work: the three projections, the shapes, and the lookahead predictions riding with the ids.
struct Job {
    projs: [*const TieredExperts; 3],
    two_phase: bool,
    batch: usize,
    topk: usize,
    input_dim1: usize,
    xq_row: usize,
    qa_row: usize,
    preds: Vec<([*const TieredExperts; 3], usize, usize)>,
}

// The projections are model weights, alive for as long as the model runs forwards.
unsafe impl Send for Job {}

pub(super) struct Doorbell {
    host: *mut u8,
    dev_base: u64,
    layout: Layout,
    /// The device sequence counter, freed by `shutdown` (the struct itself stays leaked), and its address.
    seq: Mutex<Option<CudaSlice<u32>>>,
    seq_dev: u64,
    ctx: std::sync::Arc<candle_core::cuda::cudarc::driver::CudaContext>,
    queue: Mutex<VecDeque<Job>>,
    wake: Condvar,
    sleeping: AtomicBool,
    failed: AtomicBool,
    spin: std::time::Duration,
    /// Model unload (`release`): the host thread exits and the mapped memory is freed.
    stop: AtomicBool,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

unsafe impl Send for Doorbell {}
unsafe impl Sync for Doorbell {}

/// The doorbell of the current settings generation (one per loaded model: the layout is sized for its first layer's
/// shapes). `release` retires it; the next forward builds a new one.
#[allow(clippy::type_complexity)]
static DOORBELL: Mutex<Option<(u64, std::result::Result<&'static Doorbell, String>)>> = Mutex::new(None);

/// Model unload or OOM recovery: stop the doorbell's host thread and free its mapped memory. The (small) struct stays
/// leaked.
pub(crate) fn release() {
    let cur = DOORBELL.lock().unwrap().take();
    if let Some((_, Ok(db))) = cur {
        db.shutdown();
    }
}

impl Doorbell {
    /// The process's doorbell, created (buffers, device counter, host thread) by the first call.
    fn get(dev: &CudaDevice, lay: Layout) -> Result<&'static Doorbell> {
        let g = crate::titan_cfg::generation();
        let mut cur = DOORBELL.lock().unwrap();
        if !matches!(&*cur, Some((cg, _)) if *cg == g) {
            if let Some((_, Ok(old))) = cur.take() {
                old.shutdown();
            }
            *cur = Some((g, Self::create(dev, lay).map(|d| &*Box::leak(Box::new(d))).map_err(|e| e.to_string())));
        }
        match &cur.as_ref().expect("set above").1 {
            Ok(d) => Ok(*d),
            Err(e) => candle_core::bail!("titan doorbell: {e}"),
        }
    }

    fn create(dev: &CudaDevice, layout: Layout) -> Result<Self> {
        let stream = dev.cuda_stream();
        stream.context().bind_to_thread().w()?;
        let flags = sys::CU_MEMHOSTALLOC_PORTABLE | sys::CU_MEMHOSTALLOC_DEVICEMAP;
        let host = unsafe { result::malloc_host(layout.len, flags) }.w()? as *mut u8;
        unsafe { std::ptr::write_bytes(host, 0, layout.len) };
        let mut dev_base: sys::CUdeviceptr = 0;
        unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev_base, host as *mut std::ffi::c_void, 0) }.result().w()?;
        let seq = dev.alloc_zeros::<u32>(1)?;
        let seq_dev = slice_ptr(&seq, 0).0;
        candle_core::backend::BackendDevice::synchronize(dev)?;
        let spin_us = crate::titan_cfg::var("TITAN_TIERED_SPIN_US").ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_SPIN_US);
        tracing::info!(
            "titan doorbell: {} KiB mapped host memory, up to {MAX_ROWS} rows x top-{}, {}",
            layout.len >> 10,
            layout.topk,
            if onepass::mode() > 0 { "one CPU pass per miss" } else { "gate/up and down passes" }
        );
        Ok(Self {
            host,
            dev_base,
            layout,
            seq: Mutex::new(Some(seq)),
            seq_dev,
            ctx: stream.context().clone(),
            queue: Mutex::new(VecDeque::new()),
            wake: Condvar::new(),
            sleeping: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            spin: std::time::Duration::from_micros(spin_us),
            stop: AtomicBool::new(false),
            thread: Mutex::new(None),
        })
    }

    fn start(&'static self) {
        let mut t = self.thread.lock().unwrap();
        if t.is_none() && !self.stop.load(Ordering::Acquire) {
            *t = Some(std::thread::Builder::new().name("titan-doorbell".into()).spawn(move || self.serve()).expect("spawn titan-doorbell"));
        }
    }

    fn shutdown(&'static self) {
        if self.stop.swap(true, Ordering::AcqRel) {
            return;
        }
        {
            let _q = self.queue.lock().unwrap();
            self.wake.notify_all();
        }
        if let Some(t) = self.thread.lock().unwrap().take() {
            let _ = t.join();
        }
        let ctx = self.ctx.clone();
        if ctx.bind_to_thread().is_ok() {
            drop(self.seq.lock().unwrap().take());
            let _ = unsafe { sys::cuMemFreeHost(self.host as *mut std::ffi::c_void) };
        }
        tracing::info!("titan doorbell: released {} KiB mapped host memory", self.layout.len >> 10);
    }

    fn ctl(&self, word: usize) -> *mut u32 {
        unsafe { (self.host as *mut u32).add(word) }
    }

    fn at<T>(&self, off: usize) -> *mut T {
        unsafe { self.host.add(off) as *mut T }
    }

    fn check(&self) -> Result<()> {
        let err = unsafe { std::ptr::read_volatile(self.ctl(CTL_ERR)) };
        if err != 0 || self.failed.load(Ordering::Acquire) {
            candle_core::bail!("titan doorbell: the CPU expert pass failed or timed out (sequence {err})");
        }
        Ok(())
    }

    fn seq_ptr(&self) -> (u64, ()) {
        (self.seq_dev, ())
    }

    /// Copy `n_ids` ids from `ids_ptr` and `xq_bytes` from `xq_ptr` to the regions at `ids_off` / `xq_off`, then
    /// publish the next sequence number.
    #[allow(clippy::too_many_arguments)]
    fn publish(&self, dev: &CudaDevice, ids_ptr: u64, n_ids: usize, xq_ptr: u64, xq_bytes: usize, xq_off: usize) -> Result<()> {
        let f = dev.get_or_load_custom_func("db_publish", MODULE, PTX)?;
        let (seq, g) = self.seq_ptr();
        let (n_ids, xq_vec) = (n_ids as u32, (xq_bytes / 16) as u32);
        let (host_ids, host_xq, ctl) = (self.dev_base + self.layout.ids as u64, self.dev_base + xq_off as u64, self.dev_base);
        let mut b = f.builder();
        b.arg(&ids_ptr).arg(&n_ids).arg(&xq_ptr).arg(&xq_vec).arg(&host_ids).arg(&host_xq).arg(&seq).arg(&ctl);
        unsafe { b.launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (PUBLISH_THREADS, 1, 1), shared_mem_bytes: 0 }) }.w()?;
        drop(g);
        Ok(())
    }

    fn wait(&self, dev: &CudaDevice) -> Result<()> {
        let f = dev.get_or_load_custom_func("db_wait", MODULE, PTX)?;
        let (seq, g) = self.seq_ptr();
        let (ctl, timeout) = (self.dev_base, WAIT_TIMEOUT_NS);
        let mut b = f.builder();
        b.arg(&seq).arg(&ctl).arg(&timeout);
        unsafe { b.launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 }) }.w()?;
        drop(g);
        Ok(())
    }

    /// Wait for the host's answer, then copy its rows: every block's first thread spins, then the blocks copy.
    fn collect(&self, dev: &CudaDevice, hdr_off: usize, rows_off: usize, out: &mut CudaSlice<f32>, n: usize, tasks: usize) -> Result<()> {
        let (blocks, threads) = collect_shape();
        if blocks == 0 {
            self.wait(dev)?;
            return self.scatter(dev, hdr_off, rows_off, out, n, tasks);
        }
        let f = dev.get_or_load_custom_func("db_collect", MODULE, PTX)?;
        let (seq, g0) = self.seq_ptr();
        let (ctl, timeout) = (self.dev_base, WAIT_TIMEOUT_NS);
        let (hdr, rows) = (self.dev_base + hdr_off as u64, self.dev_base + rows_off as u64);
        let (out_ptr, g) = slice_ptr(out, 0);
        let (n_u, max_rows) = (n as u32, tasks as u32);
        let mut b = f.builder();
        b.arg(&seq).arg(&ctl).arg(&timeout).arg(&hdr).arg(&rows).arg(&out_ptr).arg(&n_u).arg(&max_rows);
        unsafe { b.launch(LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 }) }.w()?;
        drop((g0, g));
        Ok(())
    }

    /// Rows listed in the header at `hdr_off` (region `rows_off`, `n` floats each) into their task rows of `out`.
    fn scatter(&self, dev: &CudaDevice, hdr_off: usize, rows_off: usize, out: &mut CudaSlice<f32>, n: usize, tasks: usize) -> Result<()> {
        let f = dev.get_or_load_custom_func("db_scatter", MODULE, PTX)?;
        let (hdr, rows) = (self.dev_base + hdr_off as u64, self.dev_base + rows_off as u64);
        let (out_ptr, g) = slice_ptr(out, 0);
        let (n_u, max_rows) = (n as u32, tasks as u32);
        let blocks = (tasks * n / 4).div_ceil(SCATTER_THREADS as usize).min(SCATTER_MAX_BLOCKS) as u32;
        let mut b = f.builder();
        b.arg(&hdr).arg(&rows).arg(&out_ptr).arg(&n_u).arg(&max_rows);
        unsafe { b.launch(LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (SCATTER_THREADS, 1, 1), shared_mem_bytes: 0 }) }.w()?;
        drop(g);
        Ok(())
    }

    fn push(&self, job: Job) {
        self.queue.lock().unwrap().push_back(job);
        if self.sleeping.load(Ordering::SeqCst) {
            let _l = self.queue.lock().unwrap();
            self.wake.notify_one();
        }
    }

    fn next_job(&self) -> Option<Job> {
        let t0 = std::time::Instant::now();
        let mut spins = 0u32;
        loop {
            if let Some(j) = self.queue.lock().unwrap().pop_front() {
                return Some(j);
            }
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            spins += 1;
            if spins % 64 == 0 && t0.elapsed() > self.spin {
                let mut q = self.queue.lock().unwrap();
                self.sleeping.store(true, Ordering::SeqCst);
                loop {
                    if let Some(j) = q.pop_front() {
                        self.sleeping.store(false, Ordering::SeqCst);
                        return Some(j);
                    }
                    if self.stop.load(Ordering::Acquire) {
                        return None;
                    }
                    q = self.wake.wait(q).unwrap();
                }
            }
            for _ in 0..16 {
                std::hint::spin_loop();
            }
        }
    }

    /// The next published sequence number after `last`, or None if none arrives within PUB_TIMEOUT.
    fn wait_pub(&self, last: u32) -> Option<u32> {
        let t0 = std::time::Instant::now();
        let mut spins = 0u32;
        loop {
            let p = unsafe { std::ptr::read_volatile(self.ctl(CTL_PUB)) };
            if p != last {
                fence(Ordering::Acquire);
                return Some(p);
            }
            spins += 1;
            if spins % 1024 == 0 && (t0.elapsed() > PUB_TIMEOUT || self.stop.load(Ordering::Acquire)) {
                return None;
            }
            std::hint::spin_loop();
        }
    }

    fn answer(&self, s: u32) {
        fence(Ordering::Release);
        unsafe { std::ptr::write_volatile(self.ctl(CTL_DONE), s) };
    }

    fn write_hdr(&self, off: usize, tasks: &[usize]) {
        let h = self.at::<u32>(off);
        unsafe {
            for (i, &t) in tasks.iter().enumerate() {
                std::ptr::write_volatile(h.add(1 + i), t as u32);
            }
            std::ptr::write_volatile(h, tasks.len() as u32);
        }
    }

    fn serve(&'static self) {
        let mut last = 0u32;
        loop {
            let Some(job) = self.next_job() else {
                return;
            };
            let Some(s) = self.wait_pub(last) else {
                tracing::warn!("titan doorbell: no publish within {PUB_TIMEOUT:?}; job dropped");
                continue;
            };
            last = s;
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.run(&job, &mut last)));
            if r.is_err() {
                tracing::error!("titan doorbell: CPU expert pass panicked");
                self.failed.store(true, Ordering::Release);
                self.write_hdr(self.layout.hdr_gu, &[]);
                self.write_hdr(self.layout.hdr_d, &[]);
                self.answer(last);
            }
        }
    }

    /// One layer: `last` is the sequence number of its (first) publish, updated by the second one in two-phase mode.
    fn run(&self, job: &Job, last: &mut u32) {
        let lay = &self.layout;
        let [gate, up, down] = job.projs.map(|p| unsafe { &*p });
        let tasks = job.batch * job.topk;
        let n_pred: usize = job.preds.iter().map(|p| p.2).sum();
        let ids: Vec<u32> = (0..tasks + n_pred).map(|i| unsafe { std::ptr::read_volatile(self.at::<u32>(lay.ids).add(i)) }).collect();
        let (ids, mut rest) = ids.split_at(tasks);
        for (projs, depth, len) in &job.preds {
            let (p, r) = rest.split_at(*len);
            let [g, u, d] = (*projs).map(|p| unsafe { &*p });
            TieredExperts::lookahead(&[g, u, d], *depth, p);
            rest = r;
        }
        for p in [gate, up, down] {
            p.prefetch(ids);
        }
        let misses = {
            let mut cache = gate.cache.lock().unwrap();
            if job.batch == 1 {
                gate.score_predictions(&cache.map, ids);
            }
            let m = Misses::new(ids, |e| cache.map[e as usize] == NOT_RESIDENT);
            let _ = gate.touch_and_admit(&mut cache, ids, job.batch);
            m
        };
        let xq = unsafe { std::slice::from_raw_parts(self.at::<u8>(lay.xq) as *const u8, MAX_ROWS * job.xq_row) };
        let (ng, nd) = (gate.n, down.n);
        let (input_dim1, topk, xq_row, qa_row) = (job.input_dim1, job.topk, job.xq_row, job.qa_row);
        if job.two_phase {
            let x = |t: usize| {
                let r = if input_dim1 == 1 { t / topk } else { t };
                &xq[r * xq_row..(r + 1) * xq_row]
            };
            TieredExperts::gate_up_rows(gate, up, &misses, &x, Out(self.at(lay.rows_g)), Out(self.at(lay.rows_u)), ng);
            self.write_hdr(lay.hdr_gu, &misses.tasks);
            self.answer(*last);
            let Some(s2) = self.wait_pub(*last) else {
                tracing::warn!("titan doorbell: no activation publish within {PUB_TIMEOUT:?}");
                return;
            };
            *last = s2;
            let qa = unsafe { std::slice::from_raw_parts(self.at::<u8>(lay.qa) as *const u8, tasks * job.qa_row) };
            let a = |i: usize| {
                let t = misses.tasks[i];
                &qa[t * qa_row..(t + 1) * qa_row]
            };
            TieredExperts::down_rows(down, &misses, &a, Out(self.at(lay.rows_d)), nd);
        } else {
            let out = unsafe { std::slice::from_raw_parts_mut(self.at::<f32>(lay.rows_d), misses.tasks.len() * nd) };
            TieredExperts::onepass(gate, up, down, &misses, xq, job.xq_row, job.input_dim1, job.topk, out);
        }
        self.write_hdr(lay.hdr_d, &misses.tasks);
        self.answer(*last);
    }
}

impl TieredExperts {
    /// The routed experts of one MoE layer through the doorbell (`TITAN_DOORBELL=1`); `None` when this layer or batch
    /// does not qualify (the caller runs the synchronous path). `preds`: later layers' predicted routing (lookahead).
    #[allow(clippy::type_complexity)]
    pub(crate) fn forward_doorbell(
        gate: &TieredExperts,
        up: &TieredExperts,
        down: &TieredExperts,
        xs: &Tensor,
        indices: &Tensor,
        preds: &[([&TieredExperts; 3], usize, Tensor)],
        act: &dyn Fn(&Tensor, &Tensor) -> Result<Tensor>,
    ) -> Result<Option<Tensor>> {
        let (batch, topk) = indices.dims2()?;
        let tasks = batch * topk;
        let xq_row = onepass::q8_row_bytes(gate.k);
        let qa_row = onepass::q8_row_bytes(down.k);
        if batch > MAX_ROWS || tasks > SCATTER_MAX_TASKS || down.n % 4 != 0 || gate.n % 4 != 0 || preds.len() > MAX_LOOKAHEAD {
            return Ok(None);
        }
        let lay = Layout::new(topk, xq_row, qa_row, gate.n, down.n);
        let db = Doorbell::get(&gate.dev, lay)?;
        if !db.layout.fits(topk, xq_row, qa_row, gate.n, down.n) {
            return Ok(None);
        }
        db.start();
        db.check()?;
        let two_phase = onepass::mode() == 0;
        let dev = &gate.dev;
        if gate.pfs.is_some() {
            titan_pfs::idle(dev, batch)?;
        }
        let qx = gate.quantize_input(xs)?;
        let rows = qx.batch * qx.input_dim1;
        let ids = indices.contiguous()?;
        let all = if preds.is_empty() {
            ids.flatten_all()?
        } else {
            let mut v = vec![ids.flatten_all()?];
            for (_, _, p) in preds {
                v.push(p.flatten_all()?);
            }
            Tensor::cat(&v, 0)?
        };
        let n_ids = all.elem_count();
        {
            let (st, lay_all) = all.storage_and_layout();
            let Storage::Cuda(c) = &*st else {
                candle_core::bail!("titan doorbell: ids not on CUDA");
            };
            let (ids_ptr, g0) = slice_ptr(c.as_cuda_slice::<u32>()?, lay_all.start_offset());
            let (xq_ptr, g1) = slice_ptr(&qx.xq, 0);
            db.publish(dev, ids_ptr, n_ids, xq_ptr, rows * qx.q8_row_bytes, db.layout.xq)?;
            drop((g0, g1));
        }
        db.push(Job {
            projs: [gate, up, down],
            two_phase,
            batch,
            topk,
            input_dim1: qx.input_dim1,
            xq_row: qx.q8_row_bytes,
            qa_row,
            preds: preds.iter().map(|(p, d, t)| ((*p).map(|x| x as *const TieredExperts), *d, t.elem_count())).collect(),
        });
        let (st, lay_ids) = ids.storage_and_layout();
        let Storage::Cuda(c) = &*st else {
            candle_core::bail!("titan doorbell: ids not on CUDA");
        };
        let (ids_ptr, g) = slice_ptr(c.as_cuda_slice::<u32>()?, lay_ids.start_offset());
        let mut og = gate.gemv_launch(&gate.cache.lock().unwrap(), &qx, ids_ptr, topk)?;
        let mut ou = up.gemv_launch(&up.cache.lock().unwrap(), &qx, ids_ptr, topk)?;
        if two_phase {
            db.collect(dev, db.layout.hdr_gu, db.layout.rows_g, &mut og, gate.n, tasks)?;
            db.scatter(dev, db.layout.hdr_gu, db.layout.rows_u, &mut ou, up.n, tasks)?;
        }
        let a = act(&gate.wrap(og, batch, topk), &up.wrap(ou, batch, topk))?;
        let qa = down.quantize_input(&a)?;
        if two_phase {
            let (qa_ptr, g2) = slice_ptr(&qa.xq, 0);
            let (seq, g3) = db.seq_ptr();
            db.publish(dev, seq, 0, qa_ptr, tasks * qa.q8_row_bytes, db.layout.qa)?;
            drop((g2, g3));
        }
        let mut od = down.gemv_launch(&down.cache.lock().unwrap(), &qa, ids_ptr, topk)?;
        drop(g);
        db.collect(dev, db.layout.hdr_d, db.layout.rows_d, &mut od, down.n, tasks)?;
        Ok(Some(down.wrap(od, batch, topk)))
    }
}
