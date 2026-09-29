//! Links libcuda dynamically (the driver API; no cudart, no nvcc). For the gate example only, links
//! the reference static libraries renamed by make_ref.sh (if present) plus cudart and libstdc++.
fn main() {
    println!("cargo:rustc-link-lib=dylib=cuda");
    for d in ["/usr/local/cuda/lib64/stubs", "/usr/local/cuda/lib64"] {
        if std::path::Path::new(d).exists() {
            println!("cargo:rustc-link-search=native={d}");
        }
    }
    let dir = format!("{}/.ref", env!("CARGO_MANIFEST_DIR"));
    let libs = ["ref_moe", "ref_quant", "ref_core", "ref_pa"];
    if libs.iter().all(|l| std::path::Path::new(&format!("{dir}/lib{l}.a")).exists()) {
        println!("cargo:rustc-link-arg-examples=-L{dir}");
        println!("cargo:rustc-link-arg-examples=-Wl,--start-group");
        for l in libs {
            println!("cargo:rustc-link-arg-examples=-l{l}");
        }
        println!("cargo:rustc-link-arg-examples=-Wl,--end-group");
        println!("cargo:rustc-link-arg-examples=-lcudart");
        println!("cargo:rustc-link-arg-examples=-lstdc++");
        println!("cargo:rustc-cfg=has_ref");
    }
    println!("cargo:rustc-check-cfg=cfg(has_ref)");
    for l in libs {
        println!("cargo:rerun-if-changed={dir}/lib{l}.a");
    }
    // The PTX embedded by src/image.rs (`.incbin`, invisible to rustc's dependency tracking).
    let manifest = env!("CARGO_MANIFEST_DIR");
    let image = std::fs::read_to_string(format!("{manifest}/src/image.rs")).unwrap();
    for line in image.lines().filter(|l| l.starts_with("embed!(")) {
        let path = line.rsplit('"').nth(1).unwrap();
        let full = format!("{manifest}/../{path}");
        assert!(std::path::Path::new(&full).exists(), "embedded PTX missing: {full}");
        println!("cargo:rerun-if-changed={full}");
    }
    println!("cargo:rerun-if-changed=src/image.rs");
    println!("cargo:rerun-if-changed=build.rs");
}
