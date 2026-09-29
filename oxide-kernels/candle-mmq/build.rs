// The gate links the REAL candle C launchers (and their embedded fatbins) from the reference
// static library, so it can run them side by side with the Rust launchers in src/launch.rs.
fn main() {
    let home = std::env::var("HOME").unwrap();
    println!("cargo:rustc-link-search=native={home}/titan-engine/oxide-kernels/reference/candle-ffi");
    println!("cargo:rustc-link-lib=static=moe");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rerun-if-changed=build.rs");
}
