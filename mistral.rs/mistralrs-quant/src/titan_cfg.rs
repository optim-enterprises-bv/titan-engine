//! titan-engine: per-model TITAN_* settings.
//!
//! The titan code reads its knobs through [`var`] instead of `std::env::var`. A model server that swaps
//! models in one process installs the settings of the model it is about to load with [`begin_model`]
//! (the `[models.titan.env]` table of the swap config); a name the table does not carry falls back to the
//! process environment, so env vars remain the defaults. An empty value in the table means "unset" (it
//! hides an env default). [`end_model`] drops the table after an unload.
//!
//! Values the code caches are held in [`GenCell`]s instead of `OnceLock`s: every `begin_model` /
//! `end_model` starts a new generation, and a cell computed under an older generation is computed
//! again on its next read. A replaced value is leaked, not dropped (readers may still hold `&'static`
//! references into it), so cells hold small settings only; buffers and threads have explicit resets.

use std::collections::HashMap;
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};
use std::sync::RwLock;

static OVERLAY: RwLock<Option<HashMap<String, String>>> = RwLock::new(None);
static GEN: AtomicU64 = AtomicU64::new(1);

/// `std::env::var` with the current model's settings on top.
pub fn var(name: &str) -> Result<String, std::env::VarError> {
    if let Some(m) = OVERLAY.read().unwrap_or_else(|e| e.into_inner()).as_ref() {
        if let Some(v) = m.get(name) {
            return if v.is_empty() { Err(std::env::VarError::NotPresent) } else { Ok(v.clone()) };
        }
    }
    std::env::var(name)
}

/// `std::env::var_os` with the current model's settings on top.
pub fn var_os(name: &str) -> Option<std::ffi::OsString> {
    if let Some(m) = OVERLAY.read().unwrap_or_else(|e| e.into_inner()).as_ref() {
        if let Some(v) = m.get(name) {
            return (!v.is_empty()).then(|| std::ffi::OsString::from(v));
        }
    }
    std::env::var_os(name)
}

/// An on/off setting: `1`/`true`/`on`/`yes`/`enable(d)` or `0`/`false`/`off`/`no`/`disable(d)` (any case,
/// trimmed); unset means `default`. Anything else keeps `default` and warns once per setting. The settings
/// used to disagree: some were on only for exactly "1" (so "true" meant off), others for anything but "0"
/// (so "false" meant on), and neither said so.
pub fn flag(name: &str, default: bool) -> bool {
    match var(name) {
        Err(_) => default,
        Ok(v) => parse_flag(&v).unwrap_or_else(|| {
            warn_once(name, &v, &default.to_string());
            default
        }),
    }
}

fn parse_flag(v: &str) -> Option<bool> {
    match v.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" | "enable" | "enabled" => Some(true),
        "0" | "false" | "off" | "no" | "disable" | "disabled" => Some(false),
        _ => None,
    }
}

/// A numeric setting; unset means `default`, and a value that does not parse keeps `default` and warns once
/// per setting (it used to fall back silently).
pub fn num<T: std::str::FromStr + std::fmt::Display + Copy>(name: &str, default: T) -> T {
    match var(name) {
        Err(_) => default,
        Ok(v) => v.trim().parse::<T>().unwrap_or_else(|_| {
            warn_once(name, &v, &default.to_string());
            default
        }),
    }
}

fn warn_once(name: &str, value: &str, used: &str) {
    static SEEN: std::sync::Mutex<Option<std::collections::HashSet<String>>> = std::sync::Mutex::new(None);
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if seen.get_or_insert_with(Default::default).insert(format!("{name}={value}")) {
        tracing::warn!("titan setting {name}={value:?} is not understood; using {used}");
    }
}

/// Log `msg` once per process for `key` (a setting fell back, or turned another one off).
pub fn note_once(key: &str, msg: impl FnOnce() -> String) {
    static SEEN: std::sync::Mutex<Option<std::collections::HashSet<String>>> = std::sync::Mutex::new(None);
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if seen.get_or_insert_with(Default::default).insert(key.to_string()) {
        tracing::warn!("{}", msg());
    }
}

/// `~/x` -> `$HOME/x` (models.toml paths are not shell-expanded).
pub fn expand_home(path: &str) -> String {
    match (path.strip_prefix("~/"), std::env::var("HOME")) {
        (Some(rest), Ok(home)) => format!("{home}/{rest}"),
        _ => path.to_string(),
    }
}

/// The settings generation; bumped by [`begin_model`] and [`end_model`].
pub fn generation() -> u64 {
    GEN.load(Ordering::Acquire)
}

/// Install `settings` for the model about to load; every [`GenCell`] recomputes on its next read.
pub fn begin_model(settings: HashMap<String, String>) {
    let mut keys: Vec<_> = settings.iter().map(|(k, v)| format!("{k}={v}")).collect();
    keys.sort();
    tracing::info!("titan settings for the next model: {}", keys.join(" "));
    *OVERLAY.write().unwrap_or_else(|e| e.into_inner()) = Some(settings);
    GEN.fetch_add(1, Ordering::AcqRel);
}

/// Drop the current model's settings (after its unload): back to the process environment.
pub fn end_model() {
    *OVERLAY.write().unwrap_or_else(|e| e.into_inner()) = None;
    GEN.fetch_add(1, Ordering::AcqRel);
}

/// A `OnceLock` that is computed again once per settings generation (see the module docs).
pub struct GenCell<T> {
    gen: AtomicU64,
    ptr: AtomicPtr<T>,
}

unsafe impl<T: Send + Sync> Sync for GenCell<T> {}
unsafe impl<T: Send> Send for GenCell<T> {}

impl<T> Default for GenCell<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> GenCell<T> {
    pub const fn new() -> Self {
        Self { gen: AtomicU64::new(0), ptr: AtomicPtr::new(std::ptr::null_mut()) }
    }

    /// The value of the current generation, if computed.
    pub fn get(&self) -> Option<&T> {
        if self.gen.load(Ordering::Acquire) != generation() {
            return None;
        }
        let p = self.ptr.load(Ordering::Acquire);
        // SAFETY: values are leaked, never freed, so a published pointer stays valid.
        (!p.is_null()).then(|| unsafe { &*p })
    }

    pub fn get_or_init(&self, f: impl FnOnce() -> T) -> &T {
        let g = generation();
        if self.gen.load(Ordering::Acquire) == g {
            let p = self.ptr.load(Ordering::Acquire);
            if !p.is_null() {
                // SAFETY: as in `get`.
                return unsafe { &*p };
            }
        }
        let p = Box::into_raw(Box::new(f()));
        self.ptr.store(p, Ordering::Release);
        self.gen.store(g, Ordering::Release);
        // SAFETY: just leaked.
        unsafe { &*p }
    }

    /// Set the value of the current generation unless one is already set (`OnceLock::get_or_init` semantics).
    pub fn set_if_unset(&self, v: T) -> &T {
        self.get_or_init(|| v)
    }
}


/// Compute capability of the CUDA device behind `dev`, as the route table writes it (`sm_120`).
/// Empty on a non-CUDA device. Cached: the attribute query costs a driver call.
#[cfg(feature = "cuda")]
fn cuda_arch(dev: &candle_core::Device) -> &'static str {
    use candle_core::cuda::cudarc::driver::sys::CUdevice_attribute;
    static ARCH: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    ARCH.get_or_init(|| {
        let candle_core::Device::Cuda(cd) = dev else {
            return String::new();
        };
        let stream = cd.cuda_stream();
        let ctx = stream.context();
        let major = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .unwrap_or(0);
        let minor = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .unwrap_or(0);
        if major == 0 && minor == 0 {
            String::new()
        } else {
            format!("sm_{major}{minor}")
        }
    })
}

#[cfg(not(feature = "cuda"))]
fn cuda_arch(_dev: &candle_core::Device) -> &'static str {
    ""
}

/// Whether Q3_K single-row decode matvecs take the in-tree cuda-oxide rows kernel.
///
/// One decision for one knob. This used to be two divergent defaults (`TITAN_Q3K_ROWS` defaulted on
/// in `gguf::fast_mmvq` and off in the qwen35 MoE path, each with its own recorded measurement).
/// Order: an explicit `TITAN_Q3K_ROWS` wins; else the measured route row for this device arch; else
/// the generic path -- never a guess.
///
/// Keyed on compute capability rather than the marketing device name: it is what the row's kernel was
/// built for and it survives a SKU rename. The row does not depend on key depth, so `keys` only has to
/// fall inside its range.
pub fn q3k_rows_enabled(dev: &candle_core::Device) -> bool {
    match var("TITAN_Q3K_ROWS") {
        Ok(v) => v != "0",
        Err(_) => {
            let key = crate::titan_routes::RouteKey {
                device: "",
                arch: cuda_arch(dev),
                quant: "Q3_K",
                role: "mmvq_decode_row",
                m: 1,
                keys: 8192,
            };
            crate::titan_routes::route(&key).is_some_and(|row| row.kernel == "mmvq_rows_q3k")
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn flags_and_numbers_read_every_usual_spelling() {
        for (v, want) in [("1", true), ("true", true), ("ON", true), (" yes ", true), ("0", false), ("false", false), ("Off", false), ("no", false)] {
            assert_eq!(parse_flag(v), Some(want), "{v}");
        }
        assert_eq!(parse_flag("maybe"), None);
        std::env::set_var("TITAN_CFG_TEST_FLAG", "true");
        assert!(flag("TITAN_CFG_TEST_FLAG", false), "\"true\" used to mean off for exact-\"1\" settings");
        std::env::set_var("TITAN_CFG_TEST_FLAG", "false");
        assert!(!flag("TITAN_CFG_TEST_FLAG", true), "\"false\" used to mean on for anything-but-\"0\" settings");
        std::env::set_var("TITAN_CFG_TEST_FLAG", "sometimes");
        assert!(flag("TITAN_CFG_TEST_FLAG", true));
        std::env::remove_var("TITAN_CFG_TEST_FLAG");
        assert!(!flag("TITAN_CFG_TEST_FLAG", false));
        std::env::set_var("TITAN_CFG_TEST_NUM", " 64 ");
        assert_eq!(num("TITAN_CFG_TEST_NUM", 8usize), 64);
        std::env::set_var("TITAN_CFG_TEST_NUM", "64x");
        assert_eq!(num("TITAN_CFG_TEST_NUM", 8usize), 8);
        std::env::remove_var("TITAN_CFG_TEST_NUM");
    }

    use super::*;

    #[test]
    fn overlay_and_generations() {
        static C: GenCell<usize> = GenCell::new();
        #[allow(unused_unsafe)]
        unsafe {
            std::env::set_var("TITAN_CFG_TEST_A", "env")
        };
        assert_eq!(var("TITAN_CFG_TEST_A").as_deref(), Ok("env"));
        assert_eq!(*C.get_or_init(|| 1), 1);
        assert_eq!(*C.get_or_init(|| 2), 1);
        begin_model(HashMap::from([
            ("TITAN_CFG_TEST_A".to_string(), String::new()),
            ("TITAN_CFG_TEST_B".to_string(), "model".to_string()),
        ]));
        assert!(var("TITAN_CFG_TEST_A").is_err());
        assert_eq!(var("TITAN_CFG_TEST_B").as_deref(), Ok("model"));
        assert!(C.get().is_none());
        assert_eq!(*C.get_or_init(|| 3), 3);
        end_model();
        assert_eq!(var("TITAN_CFG_TEST_A").as_deref(), Ok("env"));
        assert!(var("TITAN_CFG_TEST_B").is_err());
        assert_eq!(*C.get_or_init(|| 4), 4);
    }
}
