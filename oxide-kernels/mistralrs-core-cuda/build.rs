//! Links the REAL mistralrs-core host launchers + nvcc kernels (libmistralrscuda.a, -O3
//! --use_fast_math, sm_120a, --default-stream per-thread) so the gate can call each extern "C"
//! launcher from C and from Rust on identical inputs.
fn main() {
    let home = std::env::var("HOME").unwrap();
    let dir = format!("{home}/titan-engine/oxide-kernels/reference/mistralrs-core");
    println!("cargo:rustc-link-search=native={dir}");
    println!("cargo:rustc-link-lib=static=mistralrscuda");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rerun-if-changed={dir}/libmistralrscuda.a");
    println!("cargo:rerun-if-changed=build.rs");
}
