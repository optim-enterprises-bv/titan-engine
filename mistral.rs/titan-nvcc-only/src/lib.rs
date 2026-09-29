//! titan `oxide` build (nvcc-free): stubs for every CUDA launcher that mistral.rs v0.9.4 declares
//! but titan-oxide-ffi does not export at the declared ABI, i.e. kernels that exist only as nvcc
//! CUDA C++ (new in v0.9.4, or changed ABI; the latter are declared with
//! `link_name = "titan_nvcc_only_<name>"` under the oxide feature). The list is nvcc_only.txt; it
//! is the port-later list for cuda-oxide.
//!
//! A stub never launches anything. By default it prints the launcher name and aborts (so a code
//! path that needs an unported kernel fails loudly instead of computing garbage). With
//! `TITAN_NVCC_ONLY=warn` it reports each launcher once and returns 0 (the caller then continues
//! on unwritten output): a discovery mode for listing every unported launcher a workload reaches
//! in one run. Never serve with it.
#![allow(clippy::missing_safety_doc)]

use std::collections::HashSet;
use std::sync::Mutex;

static SEEN: Mutex<Option<HashSet<&'static str>>> = Mutex::new(None);

#[cold]
fn hit(name: &'static str) -> i64 {
    let warn = std::env::var("TITAN_NVCC_ONLY").is_ok_and(|v| v == "warn");
    if !warn {
        eprintln!(
            "titan oxide build: CUDA launcher `{name}` is nvcc-only (not ported to cuda-oxide); \
             this code path needs an nvcc (--features cuda) build"
        );
        std::process::abort();
    }
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if seen.get_or_insert_with(HashSet::new).insert(name) {
        eprintln!("TITAN_NVCC_ONLY hit: {name} (output of this launch is garbage)");
    }
    0
}

include!(concat!(env!("OUT_DIR"), "/stubs.rs"));
