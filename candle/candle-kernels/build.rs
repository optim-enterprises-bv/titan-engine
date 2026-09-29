use cudaforge::{KernelBuilder, Result};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    const MODULES: [&str; 11] =
        ["affine", "binary", "cast", "conv", "fill", "indexing", "quantized", "reduce", "sort", "ternary", "unary"];

    // titan-engine: TITAN_OXIDE_DIR points at titan-engine/oxide-kernels, whose gated cuda-oxide ports
    // (candle-<m>/candle_<m>.ptx: same entry names, same ABI, bit-identical output) replace the nvcc PTX.
    // When every module has a port, nvcc is not invoked for the PTX at all.
    println!("cargo::rerun-if-env-changed=TITAN_OXIDE_DIR");
    let oxide: Option<Vec<PathBuf>> = env::var("TITAN_OXIDE_DIR").ok().map(|dir| {
        MODULES.iter().map(|m| PathBuf::from(&dir).join(format!("candle-{m}/candle_{m}.ptx"))).collect()
    });
    match oxide {
        Some(srcs) if srcs.iter().all(|p| p.exists()) => {
            let mut rs = String::new();
            for (m, src) in MODULES.iter().zip(&srcs) {
                println!("cargo::rerun-if-changed={}", src.display());
                std::fs::copy(src, out_dir.join(format!("{m}.ptx"))).expect("copy oxide ptx");
                rs.push_str(&format!(
                    "pub const {}: &str = include_str!(concat!(env!(\"OUT_DIR\"), \"/{m}.ptx\"));\n",
                    m.to_uppercase()
                ));
            }
            std::fs::write(&ptx_path, rs).expect("write ptx.rs");
            println!("cargo::warning=titan: all {} candle PTX modules served from cuda-oxide (nvcc not used)", MODULES.len());
        }
        Some(srcs) => {
            let missing: Vec<_> = srcs.iter().filter(|p| !p.exists()).collect();
            panic!("TITAN_OXIDE_DIR is set but these ports are missing: {missing:?}");
        }
        None => {
            // Build for PTX
            let bindings = KernelBuilder::new()
                .source_dir("src") // Scan src/ for .cu files
                .exclude(&["moe_*.cu", "mmvq_gguf.cu", "mmq_*.cu"]) // Exclude statically compiled kernels from ptx build
                .arg("--expt-relaxed-constexpr")
                .arg("-std=c++17")
                .arg("-O3")
                .build_ptx()?;
            bindings.write(&ptx_path)?;
        }
    }

    // titan-engine: with TITAN_OXIDE_DIR the moe/mmvq/mmq host launchers come from titan-oxide-ffi
    // (linked through mistral.rs's `oxide` feature), so libmoe.a is not built and nvcc is never run.
    if env::var("TITAN_OXIDE_DIR").is_ok() {
        println!("cargo::warning=titan: libmoe.a not built; moe/mmvq/mmq launchers from titan-oxide-ffi");
        return Ok(());
    }

    let mut moe_builder = KernelBuilder::default()
        .source_files(vec![
            "src/moe/moe_gguf.cu",
            "src/moe/moe_wmma.cu",
            "src/moe/moe_wmma_gguf.cu",
            "src/mmvq_gguf.cu",
            "src/mmq_gguf/mmq_quantize.cu",
            "src/mmq_gguf/mmq_instance_q4_0.cu",
            "src/mmq_gguf/mmq_instance_q4_1.cu",
            "src/mmq_gguf/mmq_instance_q5_0.cu",
            "src/mmq_gguf/mmq_instance_q5_1.cu",
            "src/mmq_gguf/mmq_instance_q8_0.cu",
            "src/mmq_gguf/mmq_instance_q2_k.cu",
            "src/mmq_gguf/mmq_instance_q3_k.cu",
            "src/mmq_gguf/mmq_instance_q4_k.cu",
            "src/mmq_gguf/mmq_instance_q5_k.cu",
            "src/mmq_gguf/mmq_instance_q6_k.cu",
        ])
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    // Disable bf16 WMMA kernels on GPUs older than sm_80 (Ampere).
    // bf16 WMMA fragments require compute capability >= 8.0.
    let compute_cap = cudaforge::detect_compute_cap()
        .map(|arch| arch.base())
        .unwrap_or(80);
    if compute_cap < 80 {
        moe_builder = moe_builder.arg("-DNO_BF16_KERNEL");
    }

    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            moe_builder = moe_builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    if !is_target_msvc {
        moe_builder = moe_builder.arg("-Xcompiler").arg("-fPIC");
    }

    moe_builder.build_lib(out_dir.join("libmoe.a"))?;
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=moe");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
    Ok(())
}
