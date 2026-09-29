//! Links the REAL mistralrs-quant host launchers + nvcc kernels (libmistralrsquant.a, -O3
//! --use_fast_math, sm_120a, `--default-stream per-thread`) for group C (GGUF mmvq / mmq) so the
//! gate can call each extern "C" launcher from C and from Rust on identical inputs. Only the group-C
//! objects are extracted into OUT_DIR and re-archived, so unrelated objects never enter the link.
use std::process::Command;

const OBJECTS: [&str; 12] = [
    "mmvq_gguf", "mmq_quantize", "mmq_instance_q4_0", "mmq_instance_q4_1", "mmq_instance_q5_0", "mmq_instance_q5_1",
    "mmq_instance_q8_0", "mmq_instance_q2_k", "mmq_instance_q3_k", "mmq_instance_q4_k", "mmq_instance_q5_k",
    "mmq_instance_q6_k",
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
    let lib = format!("{out}/libmrqc_ref.a");
    let _ = std::fs::remove_file(&lib);
    let st = Command::new("ar").arg("rcs").arg(&lib).args(&picked).current_dir(&out).status().expect("ar rcs");
    assert!(st.success());
    println!("cargo:rustc-link-search=native={out}");
    println!("cargo:rustc-link-lib=static=mrqc_ref");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rerun-if-changed={archive}");
    println!("cargo:rerun-if-changed=build.rs");
}
