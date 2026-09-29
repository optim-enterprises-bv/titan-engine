//! Links the REAL candle-kernels host launchers + kernels (libmoe.a, nvcc -O3 sm_120a) so the gate
//! can call `launch_mmvq_gguf_*` from C and from Rust on identical inputs.
fn main() {
    let home = std::env::var("HOME").unwrap();
    let refdir = format!("{home}/titan-engine/oxide-kernels/reference/candle-ffi");
    println!("cargo:rustc-link-search=native={refdir}");
    println!("cargo:rustc-link-lib=static=moe");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rerun-if-changed={refdir}/libmoe.a");
    println!("cargo:rerun-if-changed=build.rs");
}
