//! `TITAN_CUDA_GRAPHS=1`: piecewise CUDA graphs for batch-1 decode (vLLM-style "piecewise" capture).
//!
//! A decode step cannot be one graph: every tiered MoE layer copies its routing ids to the host and
//! runs its misses on the CPU. So the trunk between two host syncs is a *segment* (e.g. the previous
//! layer's combine + shared expert, this layer's norms, GDN mixer, router and top-k) captured as its
//! own graph and replayed with one launch, while the routed experts (and the KV-cache append plus
//! attention of full-attention layers, whose offsets and lengths change every step) stay eager.
//!
//! Each segment runs its closure eagerly on the first call (warming candle's capture-time HtoD cache
//! with the step-invariant dims / strides uploads), is captured on the second (the stream-ordered
//! allocator makes the closure's allocations graph memory at fixed addresses), and is replayed after
//! that. A graph reads its inputs from fixed buffers: an input that is another graph's output is read
//! in place (its address never changes), any other input is copied into a buffer of the segment before
//! each launch. Outputs are the graph's own tensors, overwritten by the next replay of the same
//! segment, so callers must not keep them past the step (every consumer here uses them within it).
//! Replay runs the same kernels on the same arguments as the eager path, so outputs are bit-identical.
//! A capture that fails (an op that cannot be captured) falls back to eager for that segment for good.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};

use candle_core::cuda_backend::cudarc::driver::{result, sys, CudaStream, DevicePtr};
use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
use candle_core::{Device, Result, Storage, Tensor};

use crate::pipeline::cuda_graph::{
    disable_event_tracking_for_capture, end_cuda_capture_discard, restore_event_tracking_after_capture, CudaGraphHandle,
};

pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("TITAN_CUDA_GRAPHS").is_ok_and(|v| v == "1"))
}

/// Identifies one captured segment: what it computes (`tag`, `layer`) plus every value the capture
/// bakes in that could change between steps (row count, recurrent slot, state-pool addresses).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SegKey {
    pub tag: &'static str,
    pub layer: usize,
    pub extra: Vec<u64>,
}

struct Seg {
    graph: CudaGraphHandle,
    /// what the graph reads for each input
    slots: Vec<Tensor>,
    outputs: Vec<Tensor>,
}

enum State {
    /// ran eagerly once; the shapes and dtypes of its outputs
    Warm(Vec<(candle_core::Shape, candle_core::DType)>),
    Ready(Seg),
    Failed,
}

#[derive(Default)]
struct Inner {
    segs: HashMap<SegKey, State>,
    /// device addresses of graph outputs (fixed for the graph's lifetime; graphs are never dropped)
    stable: HashSet<u64>,
    captured: usize,
    failed: usize,
    replays: u64,
    copies: u64,
}

#[derive(Default)]
pub(crate) struct SegmentGraphs {
    inner: Mutex<Inner>,
}

fn cuda_parts(t: &Tensor) -> Result<(u64, usize)> {
    let (storage, layout) = t.storage_and_layout();
    let Storage::Cuda(CudaStorage { slice, device }) = &*storage else {
        candle_core::bail!("titan graphs: tensor not on CUDA");
    };
    let stream = device.cuda_stream();
    macro_rules! base {
        ($s:expr) => {{
            let (p, _g) = $s.device_ptr(&stream);
            p
        }};
    }
    let base = match slice {
        CudaStorageSlice::U8(s)
        | CudaStorageSlice::F6E2M3(s)
        | CudaStorageSlice::F6E3M2(s)
        | CudaStorageSlice::F4(s)
        | CudaStorageSlice::F8E8M0(s) => base!(s),
        CudaStorageSlice::U32(s) => base!(s),
        CudaStorageSlice::I16(s) => base!(s),
        CudaStorageSlice::I32(s) => base!(s),
        CudaStorageSlice::I64(s) => base!(s),
        CudaStorageSlice::BF16(s) => base!(s),
        CudaStorageSlice::F16(s) => base!(s),
        CudaStorageSlice::F32(s) => base!(s),
        CudaStorageSlice::F64(s) => base!(s),
        CudaStorageSlice::F8E4M3(s) => base!(s),
    };
    let es = t.dtype().size_in_bytes();
    Ok((base + (layout.start_offset() * es) as u64, t.elem_count() * es))
}

/// Device address of a tensor's first element.
pub(crate) fn dptr(t: &Tensor) -> Result<u64> {
    Ok(cuda_parts(t)?.0)
}

fn check(r: sys::CUresult) -> Result<()> {
    r.result().map_err(candle_core::Error::wrap)
}

/// Free VRAM after syncing and trimming the stream-ordered pool: graph memory comes from its own pool,
/// so blocks the default pool keeps cached from the prompt pass would otherwise starve it.
fn trim_pool(stream: &Arc<CudaStream>) -> Result<usize> {
    stream.synchronize().map_err(candle_core::Error::wrap)?;
    let ctx = stream.context();
    ctx.bind_to_thread().map_err(candle_core::Error::wrap)?;
    let mut pool = std::ptr::null_mut();
    check(unsafe { sys::cuDeviceGetMemPool(&mut pool, ctx.cu_device()) })?;
    check(unsafe { sys::cuMemPoolTrimTo(pool, 0) })?;
    let (mut free, mut total) = (0usize, 0usize);
    check(unsafe { sys::cuMemGetInfo_v2(&mut free, &mut total) })?;
    Ok(free)
}

fn graph_mem(stream: &Arc<CudaStream>) -> (u64, u64) {
    let dev = stream.context().cu_device();
    let (mut used, mut reserved) = (0u64, 0u64);
    unsafe {
        let _ = sys::cuDeviceGetGraphMemAttribute(dev, sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT, (&mut used as *mut u64).cast());
        let _ = sys::cuDeviceGetGraphMemAttribute(dev, sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT, (&mut reserved as *mut u64).cast());
    }
    (used, reserved)
}

fn copy_into(stream: &Arc<CudaStream>, src: &Tensor, dst: &Tensor) -> Result<()> {
    if src.shape() != dst.shape() || src.dtype() != dst.dtype() {
        candle_core::bail!("titan graphs: input {:?} {:?} vs slot {:?} {:?}", src.shape(), src.dtype(), dst.shape(), dst.dtype());
    }
    let (s, n) = cuda_parts(src)?;
    let (d, _) = cuda_parts(dst)?;
    unsafe { result::memcpy_dtod_async(d, s, n, stream.cu_stream()) }.map_err(candle_core::Error::wrap)
}

impl SegmentGraphs {
    /// Run `f` on `inputs` (every input contiguous), as a graph once captured. `f` must not read
    /// device data on the host or depend on host state that changes between steps beyond `key`.
    pub(crate) fn run(
        &self,
        key: SegKey,
        inputs: &[&Tensor],
        mut f: impl FnMut(&[Tensor]) -> Result<Vec<Tensor>>,
    ) -> Result<Vec<Tensor>> {
        let owned: Vec<Tensor> = inputs.iter().map(|t| (*t).clone()).collect();
        let Some(Device::Cuda(dev)) = inputs.first().map(|t| t.device().clone()) else {
            return f(&owned);
        };
        if inputs.iter().any(|t| !t.is_contiguous()) {
            return f(&owned);
        }
        let stream = dev.cuda_stream();
        let mut inner = self.inner.lock().expect("titan graphs poisoned");
        match inner.segs.remove(&key) {
            None => {
                // warm-up: eager, recording the step-invariant HtoD uploads
                drop(inner);
                let out = {
                    let _htod = dev.enable_cuda_graph_htod_cache();
                    f(&owned)?.iter().map(|t| t.contiguous()).collect::<Result<Vec<_>>>()?
                };
                let shapes = out.iter().map(|t| (t.shape().clone(), t.dtype())).collect();
                self.inner.lock().expect("titan graphs poisoned").segs.insert(key, State::Warm(shapes));
                Ok(out)
            }
            Some(State::Failed) => {
                inner.segs.insert(key, State::Failed);
                drop(inner);
                f(&owned)
            }
            Some(State::Ready(seg)) => {
                for (t, slot) in owned.iter().zip(&seg.slots) {
                    if dptr(t)? != dptr(slot)? {
                        copy_into(&stream, t, slot)?;
                        inner.copies += 1;
                    }
                }
                seg.graph.launch()?;
                inner.replays += 1;
                let out = seg.outputs.clone();
                inner.segs.insert(key, State::Ready(seg));
                Ok(out)
            }
            Some(State::Warm(shapes)) => {
                let free = trim_pool(&stream)?;
                if inner.captured == 0 {
                    tracing::info!("titan graphs: first capture, {} MiB free", free >> 20);
                }
                let mut slots = Vec::with_capacity(owned.len());
                for t in &owned {
                    if inner.stable.contains(&dptr(t)?) {
                        slots.push(t.clone());
                    } else {
                        let s = Tensor::zeros(t.shape(), t.dtype(), t.device())?;
                        copy_into(&stream, t, &s)?;
                        slots.push(s);
                    }
                }
                // outputs land in buffers outside graph memory, so every graph allocation is freed
                // inside its graph and graphs launched on this stream can share physical memory
                let bufs = shapes
                    .iter()
                    .map(|(s, d)| Tensor::zeros(s, *d, &Device::Cuda(dev.clone())))
                    .collect::<Result<Vec<_>>>()?;
                let restore = disable_event_tracking_for_capture(&stream);
                let captured = {
                    let _htod = dev.enable_cuda_graph_htod_cache();
                    match stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED) {
                        Err(e) => Err(candle_core::Error::wrap(e)),
                        Ok(()) => {
                            let outs = f(&slots).and_then(|o| {
                                if o.len() != bufs.len() {
                                    candle_core::bail!("titan graphs: {} outputs, warm-up had {}", o.len(), bufs.len());
                                }
                                for (t, b) in o.iter().zip(&bufs) {
                                    copy_into(&stream, &t.contiguous()?, b)?;
                                }
                                Ok(())
                            });
                            match outs {
                                Err(e) => {
                                    end_cuda_capture_discard(&stream);
                                    Err(e)
                                }
                                Ok(()) => match CudaGraphHandle::end_capture(&stream) {
                                    Ok(Some(g)) => Ok((g, bufs)),
                                    Ok(None) => Err(candle_core::Error::msg("empty graph")),
                                    Err(e) => Err(e),
                                },
                            }
                        }
                    }
                };
                restore_event_tracking_after_capture(&stream, restore);
                // nothing ran during capture: any failure up to the first launch leaves eager as the fallback
                let captured = captured.and_then(|(graph, outputs)| {
                    graph.upload()?;
                    graph.launch()?;
                    Ok((graph, outputs))
                });
                match captured {
                    Ok((graph, outputs)) => {
                        inner.captured += 1;
                        // a segment re-keyed (its state pool regrew) replaces its old graph; the old outputs
                        // stop counting as stable (graphs reading them in place now get copies)
                        let stale: Vec<SegKey> = inner
                            .segs
                            .keys()
                            .filter(|k| k.tag == key.tag && k.layer == key.layer && k.extra.first() == key.extra.first())
                            .cloned()
                            .collect();
                        for k in stale {
                            if let Some(State::Ready(old)) = inner.segs.remove(&k) {
                                for o in &old.outputs {
                                    let p = dptr(o)?;
                                    inner.stable.remove(&p);
                                }
                            }
                        }
                        for o in &outputs {
                            let p = dptr(o)?;
                            inner.stable.insert(p);
                        }
                        let out = outputs.clone();
                        inner.segs.insert(key, State::Ready(Seg { graph, slots, outputs }));
                        Ok(out)
                    }
                    Err(e) => {
                        tracing::warn!("titan graphs: capture of {key:?} failed, segment stays eager: {e}");
                        inner.failed += 1;
                        inner.segs.insert(key, State::Failed);
                        drop(inner);
                        f(&owned)
                    }
                }
            }
        }
    }

    pub(crate) fn report(&self, dev: &Device) -> String {
        let i = self.inner.lock().expect("titan graphs poisoned");
        let mem = match dev {
            Device::Cuda(d) => {
                let (used, reserved) = graph_mem(&d.cuda_stream());
                format!(", graph memory {} MiB used / {} MiB reserved", used >> 20, reserved >> 20)
            }
            _ => String::new(),
        };
        format!(
            "{} segments captured, {} eager after a failed capture, {} replays, {} input copies{mem}",
            i.captured, i.failed, i.replays, i.copies
        )
    }
}
