//! titan-engine: Ollama-style model swapping in one process.
//!
//! Every configured model is registered with its loader config; only the default is loaded at start. A request
//! for an unloaded model takes the swap lock, evicts the least recently used resident models until fewer than
//! `max_resident` remain (each unload waits for its in-flight requests, joins the engine thread, drops the
//! pipeline, stops the titan threads, frees the titan buffers and trims the CUDA pools), installs the model's
//! TITAN_* settings (`titan_cfg`) and loads it. Requests arriving meanwhile wait on the lock instead of failing.
//! Unknown or absent model names go to the default model. A model with an idle TTL is unloaded after that long
//! without requests.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use tokio::sync::mpsc::Sender;
use tracing::{info, warn};

use crate::{EngineConfig, MistralRs, MistralRsError, Request, UnloadedModelState};

pub const DEFAULT_MAX_RESIDENT: usize = 1;
/// The binary's global allocator hook that returns freed memory to the kernel (mimalloc's `mi_collect`), run
/// after every unload; malloc_trim only covers glibc's heap.
pub static HEAP_COLLECT: std::sync::OnceLock<fn()> = std::sync::OnceLock::new();
const IDLE_POLL: Duration = Duration::from_millis(100);
/// Longest an eviction waits for the victim's in-flight requests before unloading anyway.
const IDLE_WAIT_MAX: Duration = Duration::from_secs(1800);
/// A model handed out to a request within this long is not evicted: the request may not have reached its engine.
const HANDOUT_GRACE: Duration = Duration::from_millis(1500);
const TTL_POLL: Duration = Duration::from_secs(5);
/// Polls (IDLE_POLL apart) for the pipeline's last reference to go after its engine stopped.
const PIPELINE_DROP_POLLS: usize = 50;

/// One model's titan settings: env-style TITAN_* values applied while it is loaded, its idle TTL, its prefix
/// cache size (sequences; `None`: the global `prefix_cache_n`), and its serving settings: PagedAttention mode, KV
/// cache type and pool size, prompt tokens per scheduler step and concurrent sequences. `None` means the global
/// value; the server builds the model's loader and scheduler configs from them, which the model keeps across swaps.
#[derive(Clone, Debug, Default)]
pub struct TitanModelSettings {
    pub env: HashMap<String, String>,
    pub idle_ttl: Option<Duration>,
    pub prefix_cache_n: Option<usize>,
    /// PagedAttention for this model: `Some(None)` auto (on CUDA), `Some(Some(on))` forced; `None`: the global mode.
    pub paged_attn: Option<Option<bool>>,
    pub pa_cache_type: Option<crate::PagedCacheType>,
    /// Pool size by context length or by MiB; either one replaces the global pool sizing for this model.
    pub pa_context_len: Option<usize>,
    pub pa_memory_mb: Option<usize>,
    pub max_num_batched_tokens: Option<usize>,
    /// Sequences scheduled together (the PagedAttention `max_num_seqs`, or the default scheduler's fixed batch).
    pub max_seqs: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct TitanSwapPolicy {
    pub default_model: String,
    pub max_resident: usize,
    pub models: HashMap<String, TitanModelSettings>,
}

impl TitanSwapPolicy {
    pub fn settings(&self, model_id: &str) -> HashMap<String, String> {
        self.models.get(model_id).map(|m| m.env.clone()).unwrap_or_default()
    }

    /// The model's prefix cache size: its own if set, else `global`.
    pub fn prefix_cache_n(&self, model_id: &str, global: usize) -> usize {
        self.models.get(model_id).and_then(|m| m.prefix_cache_n).unwrap_or(global)
    }
}

pub(crate) struct SwapState {
    policy: TitanSwapPolicy,
    lock: Mutex<()>,
    swapping: AtomicBool,
    last_used: Mutex<HashMap<String, Instant>>,
}

impl SwapState {
    fn touch(&self, id: &str) {
        self.last_used.lock().unwrap_or_else(|e| e.into_inner()).insert(id.to_string(), Instant::now());
    }

    fn since_used(&self, id: &str) -> Duration {
        self.last_used.lock().unwrap_or_else(|e| e.into_inner()).get(id).map_or(Duration::MAX, Instant::elapsed)
    }
}

/// Free VRAM of `device` (cuMemGetInfo), in bytes.
pub fn device_free(device: &candle_core::Device) -> usize {
    crate::MemoryUsage.query(device).map_or(0, |m| m.available())
}

/// This process's resident set, in bytes.
pub fn host_rss() -> usize {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("VmRSS:"))
                .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<usize>().ok())
        })
        .map_or(0, |kib| kib << 10)
}

/// After a model's engine and pipeline are gone: stop the titan threads, free the titan buffers, then sync and
/// hand the CUDA pools' cached memory (stream-ordered and graph) back to the driver, and malloc's to the kernel.
pub fn release_device_memory(device: &candle_core::Device) {
    mistralrs_quant::release_titan_state();
    #[cfg(feature = "oxide")]
    crate::titan_oxide::release_workspaces();
    #[cfg(feature = "cuda")]
    if let candle_core::Device::Cuda(dev) = device {
        trim_cuda(dev);
    }
    #[cfg(not(feature = "cuda"))]
    let _ = device;
    #[cfg(target_os = "linux")]
    unsafe {
        libc::malloc_trim(0);
    }
    if let Some(collect) = HEAP_COLLECT.get() {
        collect();
    }
}

/// Whether a failed step ran out of device memory.
pub(crate) fn is_oom(e: &impl std::fmt::Debug) -> bool {
    let s = format!("{e:?}");
    s.contains("OUT_OF_MEMORY") || s.contains("out of memory")
}

/// After a step failed with a CUDA OOM (its sequences already failed and their caches dropped): destroy the model's
/// captured graphs, the pfs staging ring and the doorbell, sync, and trim the pools, so the next request starts
/// from the memory of a fresh load instead of failing on what the pool kept.
pub(crate) fn recover_from_oom(p: &dyn crate::pipeline::Pipeline) {
    use crate::pipeline::MetadataMixin;
    let device = p.device();
    let before = device_free(&device);
    p.titan_reset_graphs();
    mistralrs_quant::titan_recover_oom();
    #[cfg(feature = "cuda")]
    if let candle_core::Device::Cuda(dev) = &device {
        trim_cuda(dev);
    }
    warn!(
        "titan: CUDA out of memory: the request failed; graphs, staging ring and doorbell reset, pools trimmed ({} -> {} MiB free)",
        before >> 20,
        device_free(&device) >> 20
    );
}

#[cfg(feature = "cuda")]
fn trim_cuda(dev: &candle_core::cuda::CudaDevice) {
    use candle_core::cuda::cudarc::driver::sys;
    let stream = dev.cuda_stream();
    let ctx = stream.context();
    if ctx.bind_to_thread().is_ok() {
        let _ = stream.synchronize();
        let _ = ctx.synchronize();
        let mut pool = std::ptr::null_mut();
        unsafe {
            if sys::cuDeviceGetMemPool(&mut pool, ctx.cu_device()) == sys::CUresult::CUDA_SUCCESS {
                let _ = sys::cuMemPoolTrimTo(pool, 0);
            }
            let _ = sys::cuDeviceGraphMemTrim(ctx.cu_device());
        }
    }
}

/// VRAM free, the stream-ordered pool's reserved / used, graph memory and host RSS, for the swap log.
fn memlog(device: &candle_core::Device, stage: &str) {
    #[cfg(feature = "cuda")]
    if let Ok(Some(a)) = crate::MemoryUsage.query_cuda_allocator(device) {
        let pool = a.async_pool.map_or(String::new(), |p| format!(", pool reserved {} MiB used {} MiB", p.current.reserved >> 20, p.current.used >> 20));
        let graph = a.graph_pool.map_or(String::new(), |g| format!(", graph reserved {} MiB used {} MiB", g.reserved >> 20, g.used >> 20));
        info!("titan swap mem [{stage}]: {} MiB free{pool}{graph}, host RSS {} MiB", a.available >> 20, host_rss() >> 20);
        return;
    }
    info!("titan swap mem [{stage}]: {} MiB free, host RSS {} MiB", device_free(device) >> 20, host_rss() >> 20);
}

fn idle() -> bool {
    let s = crate::titan_monitor::snapshot();
    s.queued == 0 && s.running == 0
}

/// Run `f`, which blocks, from a tokio worker without stalling the runtime.
fn run_blocking<R>(f: impl FnOnce() -> R) -> R {
    match tokio::runtime::Handle::try_current() {
        Ok(h) if h.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => tokio::task::block_in_place(f),
        _ => f(),
    }
}

impl MistralRs {
    /// Turn on swap mode: `policy.default_model` must be loaded; the other models registered unloaded.
    pub fn set_titan_swap(self: &Arc<Self>, policy: TitanSwapPolicy) {
        let ttl = policy.models.values().any(|m| m.idle_ttl.is_some());
        let default = policy.default_model.clone();
        let st = Arc::new(SwapState {
            policy,
            lock: Mutex::new(()),
            swapping: AtomicBool::new(false),
            last_used: Mutex::new(HashMap::new()),
        });
        st.touch(&default);
        if self.titan_swap.set(st).is_err() {
            warn!("titan swap: policy already set");
            return;
        }
        if let Ok(mut d) = self.default_engine_id.write() {
            *d = Some(default);
        }
        if ttl {
            let weak = Arc::downgrade(self);
            let _ = std::thread::Builder::new().name("titan-swap-ttl".into()).spawn(move || ttl_loop(weak));
        }
    }

    /// Register a model that is not loaded yet; the first request for it loads it.
    pub fn register_unloaded_model(&self, model_id: &str, state: UnloadedModelState) -> Result<(), MistralRsError> {
        self.unloaded_models
            .write()
            .map_err(|_| MistralRsError::EnginePoisoned)?
            .insert(model_id.to_string(), state);
        Ok(())
    }

    /// Whether swap mode is on (unknown model names then go to the default model).
    pub fn titan_swap_active(&self) -> bool {
        self.titan_swap.get().is_some()
    }

    pub(crate) fn titan_swap_state(&self) -> Option<&Arc<SwapState>> {
        self.titan_swap.get()
    }

    fn titan_known(&self, id: &str) -> Result<bool, MistralRsError> {
        Ok(self.engines.read().map_err(|_| MistralRsError::EnginePoisoned)?.contains_key(id)
            || self.unloaded_models.read().map_err(|_| MistralRsError::EnginePoisoned)?.contains_key(id)
            || self.reloading_models.read().map_err(|_| MistralRsError::EnginePoisoned)?.contains(id))
    }

    /// The model a request names: an alias's target, else the default model when the name is absent or unknown.
    pub(crate) fn titan_resolve(&self, st: &SwapState, model_id: Option<&str>) -> Result<String, MistralRsError> {
        let Some(name) = model_id else {
            return Ok(st.policy.default_model.clone());
        };
        let id = self.resolve_alias(name)?;
        if self.titan_known(&id)? {
            Ok(id)
        } else {
            Ok(st.policy.default_model.clone())
        }
    }

    fn titan_loaded_sender(&self, id: &str) -> Result<Option<Sender<Request>>, MistralRsError> {
        let engines = self.engines.read().map_err(|_| MistralRsError::EnginePoisoned)?;
        Ok(engines.get(id).filter(|e| !e.is_finished()).map(|e| e.sender.clone()))
    }

    /// `get_sender` in swap mode: the named (or default) model's sender, swapping it in first if needed.
    pub(crate) fn titan_get_sender(&self, st: &SwapState, model_id: Option<&str>) -> Result<Sender<Request>, MistralRsError> {
        let id = self.titan_resolve(st, model_id)?;
        if !st.swapping.load(Ordering::Acquire) {
            if let Some(tx) = self.titan_loaded_sender(&id)? {
                st.touch(&id);
                return Ok(tx);
            }
        }
        run_blocking(|| self.titan_swap_to(st, &id))
    }

    /// Make `id` resident (evicting LRU models first) and return its sender; serialized by the swap lock.
    fn titan_swap_to(&self, st: &SwapState, id: &str) -> Result<Sender<Request>, MistralRsError> {
        let _g = st.lock.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(tx) = self.titan_loaded_sender(id)? {
            st.touch(id);
            return Ok(tx);
        }
        st.swapping.store(true, Ordering::Release);
        let r = self.titan_swap_locked(st, id);
        st.touch(id);
        st.swapping.store(false, Ordering::Release);
        r
    }

    fn titan_swap_locked(&self, st: &SwapState, id: &str) -> Result<Sender<Request>, MistralRsError> {
        let t0 = Instant::now();
        let mut evicted = Vec::new();
        loop {
            let loaded: Vec<String> = self.engines.read().map_err(|_| MistralRsError::EnginePoisoned)?.keys().cloned().collect();
            if loaded.len() < st.policy.max_resident.max(1) {
                break;
            }
            let victim = loaded
                .into_iter()
                .max_by_key(|m| st.since_used(m))
                .expect("at least one resident model");
            self.titan_unload_locked(st, &victim)?;
            evicted.push(victim);
        }
        let t1 = Instant::now();
        self.titan_load_locked(st, id)?;
        info!(
            "titan swap: [{}] -> {id} in {:.1} s (unload {:.1} s, load {:.1} s)",
            evicted.join(", "),
            t0.elapsed().as_secs_f64(),
            (t1 - t0).as_secs_f64(),
            t1.elapsed().as_secs_f64()
        );
        self.titan_loaded_sender(id)?.ok_or_else(|| MistralRsError::ModelNotFound(id.to_string()))
    }

    /// Unload a resident model with its memory released; the swap lock must be held.
    fn titan_unload_locked(&self, st: &SwapState, id: &str) -> Result<(), MistralRsError> {
        let t0 = Instant::now();
        let mut quiet = 0;
        while quiet < 2 {
            quiet = if st.since_used(id) >= HANDOUT_GRACE && idle() { quiet + 1 } else { 0 };
            if t0.elapsed() > IDLE_WAIT_MAX {
                warn!("titan swap: {id} still busy after {IDLE_WAIT_MAX:?}; unloading anyway");
                break;
            }
            std::thread::sleep(IDLE_POLL);
        }
        let waited = t0.elapsed();
        let mut inst = self
            .engines
            .write()
            .map_err(|_| MistralRsError::EnginePoisoned)?
            .remove(id)
            .ok_or_else(|| MistralRsError::ModelNotFound(id.to_string()))?;
        let Some(loader_config) = inst.reboot_state.loader_config.clone() else {
            self.engines.write().map_err(|_| MistralRsError::EnginePoisoned)?.insert(id.to_string(), inst);
            return Err(MistralRsError::NoLoaderConfig(id.to_string()));
        };
        let device = loader_config.device.clone();
        let state = UnloadedModelState {
            loader_config,
            scheduler_config: inst.reboot_state.method.clone(),
            engine_config: EngineConfig {
                no_kv_cache: inst.reboot_state.no_kv_cache,
                no_prefix_cache: inst.reboot_state.no_prefix_cache,
                prefix_cache_n: inst.reboot_state.prefix_cache_n,
                disable_eos_stop: inst.reboot_state.disable_eos_stop,
                throughput_logging_enabled: inst.reboot_state.throughput_logging_enabled,
                search_embedding_model: inst.reboot_state.search_embedding_model,
                search_callback: inst.reboot_state.search_callback.clone(),
                tool_callbacks: inst.reboot_state.tool_callbacks.clone(),
            },
            mcp_client_config: inst.reboot_state.mcp_client_config.clone(),
            category: inst.category.clone(),
            mistralrs_config: inst.config.clone(),
        };
        inst.terminate();
        inst.join();
        // the staging ring holds page-locked views of host stores the model owns: release it before the model drops
        mistralrs_quant::titan_recover_oom();
        let pipeline = Arc::downgrade(&inst.reboot_state.pipeline);
        memlog(&device, &format!("{id} engine stopped"));
        drop(inst);
        for _ in 0..PIPELINE_DROP_POLLS {
            if pipeline.strong_count() == 0 {
                break;
            }
            std::thread::sleep(IDLE_POLL);
        }
        if pipeline.strong_count() > 0 {
            warn!("titan swap: {} references to {id}'s pipeline remain after its engine stopped", pipeline.strong_count());
        }
        memlog(&device, &format!("{id} pipeline dropped"));
        self.unloaded_models.write().map_err(|_| MistralRsError::EnginePoisoned)?.insert(id.to_string(), state);
        release_device_memory(&device);
        memlog(&device, &format!("{id} titan state released, pools trimmed"));
        mistralrs_quant::titan_cfg::end_model();
        let free = device_free(&device);
        info!(
            "titan swap: unloaded {id} in {:.1} s (waited {:.1} s for requests): {} MiB VRAM free, host RSS {} MiB",
            t0.elapsed().as_secs_f64(),
            waited.as_secs_f64(),
            free >> 20,
            host_rss() >> 20
        );
        Ok(())
    }

    /// Load an unloaded model under its titan settings; the swap lock must be held.
    fn titan_load_locked(&self, st: &SwapState, id: &str) -> Result<(), MistralRsError> {
        let device = self
            .unloaded_models
            .read()
            .map_err(|_| MistralRsError::EnginePoisoned)?
            .get(id)
            .map(|s| s.loader_config.device.clone())
            .ok_or_else(|| MistralRsError::ModelNotFound(id.to_string()))?;
        let t0 = Instant::now();
        let free0 = device_free(&device);
        mistralrs_quant::titan_cfg::begin_model(st.policy.settings(id));
        let r = match tokio::runtime::Handle::try_current() {
            Ok(h) if h.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => h.block_on(self.reload_model_raw(id)),
            Ok(_) => Err(MistralRsError::ReloadFailed("swap needs a multi-threaded tokio runtime".to_string())),
            Err(_) => tokio::runtime::Runtime::new()
                .map_err(|e| MistralRsError::ReloadFailed(e.to_string()))
                .and_then(|rt| rt.block_on(self.reload_model_raw(id))),
        };
        if let Err(e) = &r {
            warn!("titan swap: loading {id} failed: {e}");
            release_device_memory(&device);
            mistralrs_quant::titan_cfg::end_model();
            return r;
        }
        let free1 = device_free(&device);
        info!(
            "titan swap: loaded {id} in {:.1} s: {} MiB VRAM free before, {} MiB after, host RSS {} MiB",
            t0.elapsed().as_secs_f64(),
            free0 >> 20,
            free1 >> 20,
            host_rss() >> 20
        );
        Ok(())
    }

    /// `/v1/models/reload` in swap mode: swap the model in.
    pub(crate) fn titan_reload(&self, st: &SwapState, id: &str) -> Result<(), MistralRsError> {
        if !self.titan_known(id)? {
            return Err(MistralRsError::ModelNotFound(id.to_string()));
        }
        run_blocking(|| self.titan_swap_to(st, id)).map(|_| ())
    }

    /// `/v1/models/unload` in swap mode: unload with the memory released.
    pub(crate) fn titan_unload(&self, st: &SwapState, id: &str) -> Result<(), MistralRsError> {
        run_blocking(|| {
            let _g = st.lock.lock().unwrap_or_else(|e| e.into_inner());
            if !self.engines.read().map_err(|_| MistralRsError::EnginePoisoned)?.contains_key(id) {
                return Err(MistralRsError::ModelAlreadyUnloaded(id.to_string()));
            }
            st.swapping.store(true, Ordering::Release);
            let r = self.titan_unload_locked(st, id);
            st.swapping.store(false, Ordering::Release);
            r
        })
    }
}

fn ttl_loop(weak: Weak<MistralRs>) {
    loop {
        std::thread::sleep(TTL_POLL);
        let Some(m) = weak.upgrade() else {
            return;
        };
        let Some(st) = m.titan_swap.get().cloned() else {
            return;
        };
        let loaded: Vec<String> = match m.engines.read() {
            Ok(e) => e.keys().cloned().collect(),
            Err(_) => continue,
        };
        let busy = !idle();
        for id in loaded {
            if busy {
                st.touch(&id);
                continue;
            }
            let Some(ttl) = st.policy.models.get(&id).and_then(|s| s.idle_ttl) else {
                continue;
            };
            if st.since_used(&id) < ttl {
                continue;
            }
            let Ok(_g) = st.lock.try_lock() else {
                continue;
            };
            info!("titan swap: {id} idle for {ttl:?}, unloading");
            st.swapping.store(true, Ordering::Release);
            if let Err(e) = m.titan_unload_locked(&st, &id) {
                warn!("titan swap: idle unload of {id} failed: {e}");
            }
            st.swapping.store(false, Ordering::Release);
        }
    }
}
