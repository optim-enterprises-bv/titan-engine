//! cuda-oxide replacements for every CUDA host launcher mistral.rs links from nvcc-built static
//! libraries (candle-kernels' libmoe.a, mistralrs-quant's libmistralrsquant.a, mistralrs-core's
//! libmistralrscuda.a, mistralrs-paged-attn's libmistralrspagedattention.a), exported under their
//! exact C symbol names. Symbols defined by more than one of those libraries are exported once, as
//! the linked binary resolves them: the libmistralrsquant.a versions of the GGUF mmvq/mmq launchers
//! (quant_c) and libmistralrscuda.a's `moe_gemm_wmma` (core_cuda).
//!
//! The launcher modules are generated from the port crates' `src/launch.rs` by `gen.py`.
//!
//! Linking: add `titan-oxide-ffi = { path = ".../oxide-kernels/titan-oxide-ffi" }` to the crate that
//! declares the `extern "C"` launchers and reference the crate once (`use titan_oxide_ffi as _;`),
//! otherwise rustc does not load the rlib and the symbols stay undefined. Needs libcuda at run time
//! and an sm_120 GPU (the embedded PTX targets sm_120). Checks: check_symbols.sh (exported set),
//! abi_check.py (signatures vs the Rust declarations), examples/gate.rs (C vs twin, byte-exact).
#![allow(clippy::all, unsafe_op_in_unsafe_fn, unused_unsafe, unused_mut, unused_variables, unused_imports, dead_code)]
pub mod cu;
mod image;

pub mod candle_moe;
pub mod core_cuda;
pub mod paged_attn_a;
pub mod paged_attn_b;
pub mod quant_a;
pub mod quant_b;
pub mod quant_c;

/// cudart's profiler range hooks (mistralrs-cli `bench` declares them), forwarded to the driver API
/// so an `oxide` build needs no libcudart. 0 is success in both APIs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cudaProfilerStart() -> i32 {
    unsafe { cu::profiler_start() as i32 }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cudaProfilerStop() -> i32 {
    unsafe { cu::profiler_stop() as i32 }
}
