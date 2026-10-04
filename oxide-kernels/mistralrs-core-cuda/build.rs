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
    // v0.9.4 graph / input-packing launchers (cuda_graph_copy_2d_bytes, pad_decode_input_u32, pack_completion_input_u32):
    // the nvcc objects of the v0.9.4 build (reference/mistralrs-core-094), every exported C symbol renamed v094_<name>
    // (graph.o also defines the old cuda_graph_copy_bytes, which libmistralrscuda.a already has).
    let d094 = format!("{home}/titan-engine/oxide-kernels/reference/mistralrs-core-094");
    let out = std::env::var("OUT_DIR").unwrap();
    let mut objs = Vec::new();
    for stem in ["graph", "input_packing"] {
        let src = std::fs::read_dir(&d094)
            .unwrap_or_else(|e| panic!("{d094}: {e}"))
            .map(|e| e.unwrap().path())
            .find(|p| p.file_name().unwrap().to_str().unwrap().starts_with(&format!("{stem}-")) && p.extension().is_some_and(|x| x == "o"))
            .unwrap_or_else(|| panic!("{stem}-*.o not in {d094}"));
        let syms = std::process::Command::new("nm").args(["-g", "--defined-only"]).arg(&src).output().unwrap();
        let mut redef = Vec::new();
        for l in String::from_utf8(syms.stdout).unwrap().lines() {
            let f: Vec<&str> = l.split_whitespace().collect();
            if f.len() == 3 && f[1] == "T" && !f[2].starts_with("_Z") {
                redef.push(format!("--redefine-sym={}=v094_{}", f[2], f[2]));
            }
        }
        let dst = format!("{out}/{stem}_094.o");
        assert!(std::process::Command::new("objcopy").args(&redef).arg(&src).arg(&dst).status().unwrap().success());
        objs.push(dst);
        println!("cargo:rerun-if-changed={}", src.display());
    }
    let lib = format!("{out}/libmrcore094.a");
    let _ = std::fs::remove_file(&lib);
    assert!(std::process::Command::new("ar").arg("rcs").arg(&lib).args(&objs).status().unwrap().success());
    println!("cargo:rustc-link-search=native={out}");
    println!("cargo:rustc-link-lib=static=mrcore094");
    println!("cargo:rerun-if-changed=build.rs");
}
