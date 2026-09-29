//! Links the REAL mistralrs-quant host launchers + nvcc kernels (libmistralrsquant.a, -O3
//! --use_fast_math, sm_120a) for group A so the gate can call each extern "C" launcher from C and
//! from Rust on identical inputs. Only the objects of group A are extracted into OUT_DIR and
//! re-archived, so unrelated objects (marlin, cutlass, ...) never enter the link.
use std::process::Command;

const OBJECTS: [&str; 11] = [
    "gemv", "ops", "indexed_moe", "moe_grouped", "hqq", "rotary", "dequant", "moe_data", "moe_align", "hqq_bitpack",
    "gelu_tanh_and_mul",
];

fn main() {
    let home = std::env::var("HOME").unwrap();
    let archive = format!("{home}/titan-engine/oxide-kernels/reference/mistralrs-quant/libmistralrsquant.a");
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
    let lib = format!("{out}/libmrqa_ref.a");
    let _ = std::fs::remove_file(&lib);
    let st = Command::new("ar").arg("rcs").arg(&lib).args(&picked).current_dir(&out).status().expect("ar rcs");
    assert!(st.success());
    println!("cargo:rustc-link-search=native={out}");
    println!("cargo:rustc-link-lib=static=mrqa_ref");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rerun-if-changed={archive}");
    println!("cargo:rerun-if-changed=build.rs");
}
