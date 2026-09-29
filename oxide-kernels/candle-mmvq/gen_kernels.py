#!/usr/bin/env python3
"""Emit the 240 explicit #[kernel] GEMV wrappers (mmvq_gguf_<q>_<dst>_plain_cuda<N>) between the
GENERATED markers in src/main.rs (the #[cuda_module] scan does not see macro-generated kernels)."""
import os

FMTS = ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k"]
DSTS = [("bf16", "u16", "B"), ("f16", "u16", "H"), ("f32", "f32", "f32")]

out = []
for fi, f in enumerate(FMTS):
    for d, cty, dty in DSTS:
        for n in range(1, 9):
            cast = "dst" if dty == "f32" else f"dst as *mut {dty}"
            out.append(
                f"    #[kernel] pub unsafe fn mmvq_gguf_{f}_{d}_plain_cuda{n}(vx: *const u8, vy: *const u8, dst: *mut {cty}, "
                f"ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) "
                f"{{ mmvq::<{fi}, {n}, {dty}>(vx, vy, {cast}, ncols_x, nrows_x, stride_col_y, stride_col_dst) }}\n")

p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src/main.rs")
src = open(p).read()
a = src.index("    // GENERATED KERNELS BEGIN\n") + len("    // GENERATED KERNELS BEGIN\n")
b = src.index("    // GENERATED KERNELS END\n")
open(p, "w").write(src[:a] + "".join(out) + src[b:])
print(len(out), "kernels")
