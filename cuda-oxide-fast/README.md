# cuda-oxide-fast: patched cuda-oxide codegen backend (release build)

`cargo oxide setup` builds the codegen backend in the **debug** profile (386 MB .so), and
`detect.rs::contains_instruction_family_modifier` searched the whole remaining LLVM text for five
escape strings per `mul.`/`add.`/`fma.` match (quadratic). Large kernel crates spent 30+ min in it.

Fix: `0001-*.patch` bounds that search to the instruction token (same result by construction).
Built with `--release` (43 MB). Verified 2026-09-27: candle-affine/unary/conv/quantized (436 kernels)
emit instruction-identical PTX (only label numbers, comments and function order differ).

Rebuild: copy cuda-oxide rev b9847e9 to `src/`, `git am 0001-*.patch`, then in
`src/crates/rustc-codegen-cuda`: `CARGO_TARGET_DIR=../../../target CARGO_ENCODED_RUSTFLAGS= cargo build --lib --release --target host-tuple`
and copy `target/x86_64-unknown-linux-gnu/release/librustc_codegen_cuda.so` here.

Use: `export CUDA_OXIDE_BACKEND=$HOME/titan-engine/cuda-oxide-fast/librustc_codegen_cuda.so`
(a new backend has a new identity hash: the first build of each crate recompiles its dependencies).
