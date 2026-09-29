#!/usr/bin/env python3
"""Writes the explicit #[kernel] wrappers between the GENERATED markers of src/main.rs."""
import re
FMTS = [  # (name, FMT const, service float program is .ftz)
    ("q4k", "FMT_Q4K", False), ("q5k", "FMT_Q5K", False), ("q6k", "FMT_Q6K", False),
    ("q1_0", "FMT_Q1_0", True), ("iq4_nl", "FMT_IQ4_NL", True), ("mxfp4", "FMT_MXFP4", True), ("nvfp4", "FMT_NVFP4", True),
]
ARGS = "w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32"
CALL = "(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1)"
out = []
def k(name, bounds, body, doc):
    out.append(f"    /// {doc}")
    out.append(f"    #[kernel]")
    out.append(f"    #[launch_bounds({bounds})]")
    out.append(f"    pub unsafe fn {name}({ARGS}) {{ {body}{CALL} }}")
for n, F, ftz in FMTS:
    f = "true" if ftz else "false"
    prog = "llama.cpp b1 program" if ftz else "candle's program"
    k(f"{n}_moe_mmvq", 256, f"moe_warp::<{F}, {f}, 4>", f"{n}: MoE grid, each column reduced as titan's b1 GEMV ({prog}). Service kernel, b = 2..8.")
    k(f"{n}_moe_mmvq_llama", 256, f"moe_warp::<{F}, true, 1>", f"{n}: llama.cpp mul_mat_vec_q_moe<type, 2, false> float program (gated against it).")
    k(f"{n}_moe_b1", 128, f"b1_block::<{F}, {f}, 1>", f"{n}: llama.cpp b1 MUL_MAT_ID grid, rpb 1 ({prog}). Service kernel, b = 1.")
    k(f"{n}_moe_b1_sk", 128, f"b1_block::<{F}, {f}, 4>", f"{n}: llama.cpp b1 MUL_MAT_ID grid, small_k (rpb 4) ({prog}). Service kernel, b = 1.")
    if not ftz:
        k(f"{n}_moe_b1_llama", 128, f"b1_block::<{F}, true, 1>", f"{n}: llama.cpp mul_mat_vec_q<type, 1, false, false> float program (gated against it).")
        k(f"{n}_moe_b1_sk_llama", 128, f"b1_block::<{F}, true, 4>", f"{n}: llama.cpp mul_mat_vec_q<type, 1, false, true> float program (gated against it).")
p = "src/main.rs"
s = open(p).read()
a, b = "    // GENERATED KERNELS BEGIN (gen_kernels.py)\n", "    // GENERATED KERNELS END\n"
i, j = s.index(a) + len(a), s.index(b)
s = s[:i] + "\n".join(out) + "\n" + s[j:]
open(p, "w").write(s)
print(len([l for l in out if "#[kernel]" in l]), "kernels")
