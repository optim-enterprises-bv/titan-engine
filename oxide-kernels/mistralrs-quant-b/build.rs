//! Links the REAL mistralrs-quant host launchers + kernels (libmistralrsquant.a, nvcc -O3
//! --use_fast_math sm_120a) so the gate can call each extern "C" launcher from C and from Rust.
fn main() {
    let home = std::env::var("HOME").unwrap();
    let refdir = format!("{home}/titan-engine/oxide-kernels/reference/mistralrs-quant");
    println!("cargo:rustc-link-search=native={refdir}");
    println!("cargo:rustc-link-lib=static=mistralrsquant");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rerun-if-changed={refdir}/libmistralrsquant.a");
    println!("cargo:rerun-if-changed=build.rs");
}
