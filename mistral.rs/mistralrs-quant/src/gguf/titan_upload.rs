//! Asynchronous expert uploads for `TieredExperts` (`TITAN_TIERED_ASYNC=1`). A worker thread copies an
//! expert's host bytes into pinned staging (faulting mmap pages in off the decode thread), makes its own
//! stream wait on the decode stream's swap event, then issues the copy into the slot. The decode thread
//! maps the expert into its slot only after the copy's event has completed; until then the CPU twin
//! serves it, so the output never depends on when a copy lands.

use candle_core::cuda::cudarc::driver::{result, CudaContext, CudaEvent, CudaStream};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    mpsc, Arc, OnceLock,
};

/// Pinned staging buffers, reused round robin once their copy has completed.
const STAGING_BUFS: usize = 16;
/// Uploads submitted but not yet issued by the worker; admissions stop at this many.
const MAX_QUEUED: usize = 64;

/// Completion of one upload: set by the worker once the copy is issued.
pub struct Done(OnceLock<Arc<CudaEvent>>);

impl Done {
    /// The copy's event, once the copy has been issued and has completed on the GPU.
    pub fn landed(&self) -> Option<&CudaEvent> {
        self.0.get().map(|e| &**e).filter(|e| e.is_complete())
    }
}

struct Job {
    src: *const u8,
    len: usize,
    dst: u64,
    after: Arc<CudaEvent>,
    done: Arc<Done>,
}

// `src` points into a host store (owned buffer or GGUF mmap) that lives as long as the model.
unsafe impl Send for Job {}

pub struct Uploader {
    tx: mpsc::Sender<Job>,
    queued: Arc<AtomicUsize>,
}

struct Staging {
    ptr: *mut u8,
    cap: usize,
    copy: Option<Arc<CudaEvent>>,
}

/// The process-wide uploader on the context of `stream` (the decode stream).
pub fn uploader(stream: &Arc<CudaStream>) -> &'static Uploader {
    static U: OnceLock<Uploader> = OnceLock::new();
    U.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Job>();
        let queued = Arc::new(AtomicUsize::new(0));
        let (ctx, q) = (stream.context().clone(), queued.clone());
        std::thread::Builder::new()
            .name("titan-upload".into())
            .spawn(move || worker(ctx, rx, q))
            .expect("spawn titan-upload");
        Uploader { tx, queued }
    })
}

impl Uploader {
    /// How many more uploads may be submitted now.
    pub fn room(&self) -> usize {
        MAX_QUEUED.saturating_sub(self.queued.load(Ordering::Acquire))
    }

    /// Queue a copy of `src` to device address `dst`, ordered after `after` (recorded on the decode stream).
    pub fn submit(&self, src: &[u8], dst: u64, after: Arc<CudaEvent>) -> Arc<Done> {
        let done = Arc::new(Done(OnceLock::new()));
        self.queued.fetch_add(1, Ordering::AcqRel);
        let job = Job { src: src.as_ptr(), len: src.len(), dst, after, done: done.clone() };
        if self.tx.send(job).is_err() {
            self.queued.fetch_sub(1, Ordering::AcqRel);
        }
        done
    }
}

fn worker(ctx: Arc<CudaContext>, rx: mpsc::Receiver<Job>, queued: Arc<AtomicUsize>) {
    let stream = match ctx.bind_to_thread().and_then(|_| ctx.new_stream()) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("titan tiered async uploads: no upload stream ({e}); admitted experts stay on the CPU");
            return;
        }
    };
    let mut bufs: Vec<Staging> = Vec::with_capacity(STAGING_BUFS);
    let mut next = 0usize;
    let mut warned = false;
    while let Ok(job) = rx.recv() {
        if let Err(e) = issue(&ctx, &stream, &mut bufs, &mut next, &job) {
            if !std::mem::replace(&mut warned, true) {
                tracing::warn!("titan tiered async upload failed ({e}); that expert stays on the CPU");
            }
        }
        queued.fetch_sub(1, Ordering::AcqRel);
    }
}

fn issue(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    bufs: &mut Vec<Staging>,
    next: &mut usize,
    job: &Job,
) -> std::result::Result<(), candle_core::cuda::cudarc::driver::DriverError> {
    if bufs.len() < STAGING_BUFS {
        bufs.push(Staging { ptr: std::ptr::null_mut(), cap: 0, copy: None });
    }
    let i = *next % bufs.len();
    *next += 1;
    let b = &mut bufs[i];
    if let Some(ev) = b.copy.take() {
        ev.synchronize()?;
    }
    if b.cap < job.len {
        if !b.ptr.is_null() {
            unsafe { result::free_host(b.ptr as _) }?;
            b.ptr = std::ptr::null_mut();
            b.cap = 0;
        }
        b.ptr = unsafe { result::malloc_host(job.len, 0) }? as *mut u8;
        b.cap = job.len;
    }
    unsafe { std::ptr::copy_nonoverlapping(job.src, b.ptr, job.len) };
    let staged = unsafe { std::slice::from_raw_parts(b.ptr as *const u8, job.len) };
    stream.wait(&job.after)?;
    unsafe { result::memcpy_htod_async(job.dst, staged, stream.cu_stream()) }?;
    let ev = Arc::new(ctx.new_event(None)?);
    ev.record(stream)?;
    b.copy = Some(ev.clone());
    let _ = job.done.0.set(ev);
    Ok(())
}
