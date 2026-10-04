//! Links the REAL mistralrs-paged-attn v0.9.4 host launchers + nvcc kernels (reference/mistralrs-paged-attn-094/
//! libmistralrspagedattention.a: the nvcc objects of titan model-swap / titan-094, see ../mistralrs-paged-attn-b/ref094/README;
//! -O3 --use_fast_math -DENABLE_FP8, sm_120a, `--default-stream per-thread`) for group 1 so the gate
//! can call each extern "C" launcher from C and from Rust on identical inputs. Only the objects of
//! group 1 are extracted into OUT_DIR and re-archived (flashinfer_* never enter the link).
use std::process::Command;

const OBJECTS: [&str; 13] = [
    "reshape_and_cache_kernel",
    "gather_kv_cache_kernel",
    "update_kvscales",
    "copy_blocks_kernel",
    "concat_and_cache_mla_kernel",
    "gather_mla_cache_kernel",
    "pagedattention_v1_f32",
    "pagedattention_v1_f16",
    "pagedattention_v1_bf16",
    "pagedattention_v2_f32",
    "pagedattention_v2_f16",
    "pagedattention_v2_bf16",
    "flash_attn_sinks",
];

fn main() {
    let home = std::env::var("HOME").unwrap();
    let archive = format!("{home}/titan-engine/oxide-kernels/reference/mistralrs-paged-attn-094/libmistralrspagedattention.a");
    let out = std::env::var("OUT_DIR").unwrap();
    let list = Command::new("ar").arg("t").arg(&archive).output().expect("ar t");
    let members: Vec<String> = String::from_utf8(list.stdout).unwrap().lines().map(|s| s.to_string()).collect();
    let mut picked = Vec::new();
    for o in OBJECTS {
        let m = members
            .iter()
            .find(|m| m.rsplit_once('-').map(|(stem, _)| stem == o).unwrap_or(false))
            .unwrap_or_else(|| panic!("{o}: not in {archive}"));
        picked.push(m.clone());
    }
    let st = Command::new("ar").arg("x").arg(&archive).args(&picked).current_dir(&out).status().expect("ar x");
    assert!(st.success());
    let lib = format!("{out}/libmrpa_ref.a");
    let _ = std::fs::remove_file(&lib);
    let st = Command::new("ar").arg("rcs").arg(&lib).args(&picked).current_dir(&out).status().expect("ar rcs");
    assert!(st.success());
    println!("cargo:rustc-link-search=native={out}");
    println!("cargo:rustc-link-lib=static=mrpa_ref");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rerun-if-changed={archive}");
    println!("cargo:rerun-if-changed=build.rs");
}
