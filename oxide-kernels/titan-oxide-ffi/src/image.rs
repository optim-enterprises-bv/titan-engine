//! The ports' committed cuda-oxide PTX, embedded at link time with the assembler's `.incbin` (not
//! `include_bytes!`, which would also copy all ~150 MB into the rlib metadata and rustc's memory),
//! each followed by a NUL so the embedded bytes are passed to `cuModuleLoadData` as they are. Each
//! module is JIT-compiled once per CUDA context by the launcher module that uses it; the driver
//! caches the JIT result (~/.nv/ComputeCache), so only a machine's first run pays the compile.
//! build.rs lists the files for rerun-if-changed.

macro_rules! embed {
    ($name:ident, $sym:literal, $path:literal) => {
        core::arch::global_asm!(concat!(
            ".pushsection .rodata.titan_oxide_ptx.", $sym, ",\"a\",@progbits\n",
            ".balign 16\n",
            ".globl ", $sym, "_start\n.hidden ", $sym, "_start\n",
            ".globl ", $sym, "_end\n.hidden ", $sym, "_end\n",
            $sym, "_start:\n",
            ".incbin \"", env!("CARGO_MANIFEST_DIR"), "/../", $path, "\"\n",
            $sym, "_end:\n",
            ".byte 0\n",
            ".popsection\n"
        ));
        pub fn $name() -> &'static [u8] {
            unsafe extern "C" {
                #[link_name = concat!($sym, "_start")]
                static START: u8;
                #[link_name = concat!($sym, "_end")]
                static END: u8;
            }
            unsafe {
                let s = &raw const START;
                let len = (&raw const END).offset_from(s) as usize + 1; // + the NUL
                core::slice::from_raw_parts(s, len)
            }
        }
    };
}

embed!(candle_moe, "titan_oxide_ffi_ptx_candle_moe", "candle-moe/candle_moe.ptx");
embed!(core_cuda, "titan_oxide_ffi_ptx_core_cuda", "mistralrs-core-cuda/mistralrs_core_cuda.ptx");
embed!(paged_attn_a, "titan_oxide_ffi_ptx_paged_attn_a", "mistralrs-paged-attn-a/mistralrs_paged_attn_a.ptx");
embed!(paged_attn_b, "titan_oxide_ffi_ptx_paged_attn_b", "mistralrs-paged-attn-b/mistralrs_paged_attn_b.ptx");
embed!(quant_a, "titan_oxide_ffi_ptx_quant_a", "mistralrs-quant-a/mistralrs_quant_a.ptx");
embed!(quant_b, "titan_oxide_ffi_ptx_quant_b", "mistralrs-quant-b/mistralrs_quant_b.ptx");
embed!(quant_b_marlin, "titan_oxide_ffi_ptx_quant_b_marlin", "mistralrs-quant-b/marlin-kernels/marlin_kernels.ptx");
embed!(quant_c, "titan_oxide_ffi_ptx_quant_c", "mistralrs-quant-c/mistralrs_quant_c.ptx");

/// Every embedded module (name, NUL-terminated PTX), for tools and the gate.
pub fn all() -> [(&'static str, &'static [u8]); 8] {
    [
        ("candle_moe", candle_moe()),
        ("core_cuda", core_cuda()),
        ("paged_attn_a", paged_attn_a()),
        ("paged_attn_b", paged_attn_b()),
        ("quant_a", quant_a()),
        ("quant_b", quant_b()),
        ("quant_b_marlin", quant_b_marlin()),
        ("quant_c", quant_c()),
    ]
}

/// The PTX files embedded above, relative to the oxide-kernels directory (for build.rs).
pub const FILES: [&str; 8] = [
    "candle-moe/candle_moe.ptx",
    "mistralrs-core-cuda/mistralrs_core_cuda.ptx",
    "mistralrs-paged-attn-a/mistralrs_paged_attn_a.ptx",
    "mistralrs-paged-attn-b/mistralrs_paged_attn_b.ptx",
    "mistralrs-quant-a/mistralrs_quant_a.ptx",
    "mistralrs-quant-b/mistralrs_quant_b.ptx",
    "mistralrs-quant-b/marlin-kernels/marlin_kernels.ptx",
    "mistralrs-quant-c/mistralrs_quant_c.ptx",
];
