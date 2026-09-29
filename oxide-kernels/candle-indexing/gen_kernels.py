#!/usr/bin/env python3
"""Emit the explicit #[kernel] wrappers for candle indexing.cu (the #[cuda_module] scan does not
see macro-generated kernels). Output is pasted between the GENERATED markers in src/main.rs."""
import re, subprocess, os

ref = os.path.expanduser("~/titan-engine/oxide-kernels/reference/candle/indexing.ptx")
names = re.findall(r"^\.visible \.entry (\w+)\(", open(ref).read(), re.M)

VT = {"bf16": "u16", "f16": "u16", "f8_e4m3": "u8", "f32": "f32", "f64": "f64", "u8": "u8", "u32": "u32", "i64": "i64"}
IT = {"i64": "i64", "u32": "u32", "u8": "u8", "i16": "i16", "i32": "i32"}
ZERO = {"u16": "0u16", "u8": "0u8", "f32": "0.0f32", "f64": "0.0f64", "u32": "0u32", "i64": "0i64"}
ADD = {"bf16": "add_bf16", "f16": "add_f16", "f8_e4m3": "add_e4m3", "f32": "add_f32", "f64": "add_f64",
       "u8": "add_u8", "u32": "add_u32", "i64": "add_i64"}

out = []
for n in names:
    m = re.match(r"(is|gather|ia|sa|s)_(i64|u32|u8|i16|i32)_(\w+)$", n)
    op, it, vt = m.groups()
    I, T = IT[it], VT[vt]
    f8 = vt == "f8_e4m3"
    if op == "is":
        sig = f"numel: usize, num_dims: usize, info: *const usize, ids: *const {I}, inp: *const {T}, out: *mut {T}, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize"
        body = f"index_select(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, {ZERO[T]});"
    elif op == "gather":
        sig = f"numel: usize, ids: *const {I}, inp: *const {T}, out: *mut {T}, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize"
        body = f"gather(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, {ZERO[T]});"
    elif op == "ia":
        sig = f"ids: *const {I}, ids_dim_size: usize, inp: *const {T}, out: *mut {T}, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize"
        fn = "index_add_f8" if f8 else "index_add"
        extra = "" if f8 else f", {ADD[vt]}"
        body = f"{fn}(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size{extra});"
    else:
        sig = f"ids: *const {I}, inp: *const {T}, out: *mut {T}, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize"
        if op == "s":
            body = "scatter(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size);"
        elif f8:
            body = "scatter_add_f8(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size);"
        else:
            body = f"scatter_add(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size, {ADD[vt]});"
    out.append(f"    #[kernel]\n    pub unsafe fn {n}({sig}) {{\n        unsafe {{ {body} }}\n    }}\n")

src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src/main.rs")
src = open(src_path).read()
a = src.index("    // GENERATED KERNELS BEGIN\n") + len("    // GENERATED KERNELS BEGIN\n")
b = src.index("    // GENERATED KERNELS END\n")
open(src_path, "w").write(src[:a] + "\n".join(out) + src[b:])
print(len(names), "kernels")
