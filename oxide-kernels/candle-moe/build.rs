//! Links candle's real MoE C launchers (libmoe.a, nvcc -O3 sm_120a) into the gate binary.
fn main() {
    let home = std::env::var("HOME").unwrap();
    let reference = format!("{home}/titan-engine/oxide-kernels/reference/candle-ffi");
    println!("cargo:rustc-link-search=native={reference}");
    println!("cargo:rustc-link-lib=static=moe");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rerun-if-changed={reference}/libmoe.a");
    println!("cargo:rerun-if-changed=build.rs");
}
