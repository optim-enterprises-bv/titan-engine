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

#[cfg(test)]
mod tests {
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
