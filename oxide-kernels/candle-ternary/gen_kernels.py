#!/usr/bin/env python3
"""Emit the explicit #[kernel] wrappers for candle ternary.cu (the #[cuda_module] scan does not
see macro-generated kernels). Output is written between the GENERATED markers in src/main.rs."""
import re, os

ref = os.path.expanduser("~/titan-engine/oxide-kernels/reference/candle/ternary.ptx")
names = re.findall(r"^\.visible \.entry (\w+)\(", open(ref).read(), re.M)
VT = {"bf16": "u16", "f16": "u16", "fp8_e4m3": "u8", "f32": "f32", "f64": "f64", "u8": "u8", "u32": "u32", "i64": "i64"}
IT = {"i64": "i64", "u32": "u32", "u8": "u8", "i16": "i16", "i32": "i32"}
out = []
for n in names:
    it, vt = re.match(r"where_(i64|u32|u8|i16|i32)_(\w+)$", n).groups()
    I, T = IT[it], VT[vt]
    out.append(f"    #[kernel]\n    pub unsafe fn {n}(numel: usize, num_dims: usize, info: *const usize, ids: *const {I}, t: *const {T}, f: *const {T}, out: *mut {T}) {{\n"
               f"        unsafe {{ where_op(numel, num_dims, info, ids, t, f, out) }}\n    }}\n")
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src/main.rs")
src = open(src_path).read()
a = src.index("    // GENERATED KERNELS BEGIN\n") + len("    // GENERATED KERNELS BEGIN\n")
b = src.index("    // GENERATED KERNELS END\n")
open(src_path, "w").write(src[:a] + "\n".join(out) + src[b:])
print(len(names), "kernels")
