import re

p = "~/titan-engine/candle/candle-core/src/quantized/cuda.rs"
s = open(p).read()

NEW = '''
// --- i-quant families (ggml types 16/18/22/23) -------------------------------------------------
// Ported from llama.cpp acecd56 into oxide-kernels/<name> and gated bit-identical against
// libggml-cuda.so (`1074 launches, 2909304 bytes compared, 0 failing`). Same MMVQ entry-point
// shape as IQ4_NL, so they reuse llama_mul_mat_vec / llama_dequantize unchanged.
//
// Each of these has three *distinct* grids (iq2xxs_grid, iq2s_grid, iq3xxs_grid) plus a per-8-group
// scale, so unlike IQ4_NL they cannot share one kernel body.

/// IQ4_XS (type 23): 4-bit, QK_K=256 values, 4-bit group scales.
const IQ4_XS_FMT: LlamaFmt = LlamaFmt {
    ptx: include_str!("iq4_xs_oxide.ptx"),
    module: "titan_iq4_xs",
    prefix: "iq4_xs",
    qk: 256,
    // vdr = 4, qi = 32 -> vdr * 32 / qi = 4
    blocks_per_iter_1warp: 2,
    deq_per_block: false,
};

/// IQ3_XXS (type 18): 3-bit, QK_K=256, 8-byte grid codebook + separate sign stream.
const IQ3_XXS_FMT: LlamaFmt = LlamaFmt {
    ptx: include_str!("iq3_xxs_oxide.ptx"),
    module: "titan_iq3_xxs",
    prefix: "iq3_xxs",
    qk: 256,
    // vdr = 2, qi = 16 -> vdr * 32 / qi = 4
    blocks_per_iter_1warp: 4,
    deq_per_block: false,
};

/// IQ2_S (type 22): 2-bit, QK_K=256, u64 grid codebook.
const IQ2_S_FMT: LlamaFmt = LlamaFmt {
    ptx: include_str!("iq2_s_oxide.ptx"),
    module: "titan_iq2_s",
    prefix: "iq2_s",
    qk: 256,
    blocks_per_iter_1warp: 4,
    deq_per_block: false,
};

/// IQ2_XXS (type 16): 2-bit, QK_K=256, u64 grid codebook derived from a 1-byte index.
const IQ2_XXS_FMT: LlamaFmt = LlamaFmt {
    ptx: include_str!("iq2_xxs_oxide.ptx"),
    module: "titan_iq2_xxs",
    prefix: "iq2_xxs",
    qk: 256,
    blocks_per_iter_1warp: 4,
    deq_per_block: false,
};

'''

anchor = "fn llama_fmt(dtype: GgmlDType) -> Option<&'static LlamaFmt> {"
assert anchor in s
s = s.replace(anchor, NEW.lstrip("\n") + anchor, 1)

a = """        GgmlDType::NVFP4 => Some(&NVFP4_FMT),
        _ => None,"""
b = """        GgmlDType::NVFP4 => Some(&NVFP4_FMT),
        GgmlDType::IQ4XS => Some(&IQ4_XS_FMT),
        GgmlDType::IQ3XXS => Some(&IQ3_XXS_FMT),
        GgmlDType::IQ2S => Some(&IQ2_S_FMT),
        GgmlDType::IQ2XXS => Some(&IQ2_XXS_FMT),
        _ => None,"""
assert a in s, "llama_fmt arms"
s = s.replace(a, b, 1)

open(p, "w").write(s)
print("cuda.rs: 4 LlamaFmt entries + dispatch (", len(s), "bytes )")
