//! Generates one `#[no_mangle] extern "C"` stub per line of nvcc_only.txt.
use std::fmt::Write;

fn main() {
    println!("cargo:rerun-if-changed=nvcc_only.txt");
    let list = std::fs::read_to_string("nvcc_only.txt").expect("nvcc_only.txt");
    let mut out = String::new();
    for line in list.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
        let name = line.split('\t').next().unwrap().trim();
        writeln!(
            out,
            "#[unsafe(no_mangle)]\npub unsafe extern \"C\" fn {name}() -> i64 {{ crate::hit(\"{name}\") }}"
        )
        .unwrap();
    }
    let dst = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("stubs.rs");
    std::fs::write(dst, out).unwrap();
}
