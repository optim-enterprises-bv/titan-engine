#!/usr/bin/env python3
"""Emit the explicit #[kernel] wrappers for candle conv.cu (the #[cuda_module] scan does not see
macro-generated kernels). Output is written between the GENERATED markers in src/main.rs."""
import re, os

ref = os.path.expanduser("~/titan-engine/oxide-kernels/reference/candle/conv.ptx")
names = re.findall(r"^\.visible \.entry (\w+)\(", open(ref).read(), re.M)

# dtype -> (T, A, load, mac, store, zeroA, avg_add, avg_fin, max, zeroT, to_f64, from_f64, add_T)
D = {
    "bf16": ("u16", "f32", "bf16_f32", "mac_f32", "f32_bf16", "0.0f32", "add_f32", "avg_bf16", "max_bf16", "0u16", "bf16_f64", "f64_bf16", "add_bf16"),
    "f16":  ("u16", "f32", "f16_f32", "mac_f32", "f32_f16", "0.0f32", "add_f32", "avg_f16", "max_f16", "0u16", "f16_f64", "f64_f16", "add_f16"),
    "f32":  ("f32", "f32", "id::<f32>", "mac_f32", "id::<f32>", "0.0f32", "add_f32", "avg_f32", "max_f32", "0.0f32", "f32_f64", "f64_f32", "add_f32"),
    "f64":  ("f64", "f64", "id::<f64>", "mac_f64", "id::<f64>", "0.0f64", "sass_dadd", "avg_f64", "sass_dmax", "0.0f64", "id::<f64>", "id::<f64>", "sass_dadd"),
    "u8":   ("u8", "u8", "id::<u8>", "mac_u8", "id::<u8>", "0u8", "add_u8", "avg_u8", "max_u8", "0u8", "u8_f64", "f64_u8", "add_u8"),
    "u32":  ("u32", "u32", "id::<u32>", "mac_u32", "id::<u32>", "0u32", "add_u32", "avg_u32", "max_u32", "0u32", "u32_f64", "f64_u32", "add_u32"),
}

out = []
for n in names:
    m = re.match(r"(conv1d|conv2d|conv_transpose1d|conv_transpose2d|avg_pool2d|max_pool2d|upsample_nearest2d|upsample_bilinear2d|im2col1d|col2im1d|im2col)_(\w+)$", n)
    op, dt = m.groups()
    T, A, load, mac, store, za, aadd, afin, mx, zt, tof, fromf, addt = D[dt]
    conv_tail = f"info, src, kernel, dst, {za}, {load}, {mac}, {store}"
    if op == "conv1d":
        sig = f"_src_numel: usize, l_out: usize, stride: usize, padding: usize, dilation: usize, info: *const usize, src: *const {T}, kernel: *const {T}, dst: *mut {T}"
        body = f"conv1d(l_out, stride, padding, dilation, {conv_tail})"
    elif op == "conv2d":
        sig = f"_src_numel: usize, w_out: usize, h_out: usize, stride: usize, padding: usize, dilation: usize, info: *const usize, src: *const {T}, kernel: *const {T}, dst: *mut {T}"
        body = f"conv2d(w_out, h_out, stride, padding, dilation, {conv_tail})"
    elif op == "conv_transpose1d":
        sig = f"_src_numel: usize, l_out: usize, stride: usize, padding: usize, out_padding: usize, dilation: usize, info: *const usize, src: *const {T}, kernel: *const {T}, dst: *mut {T}"
        body = f"conv_transpose1d(l_out, stride, padding, out_padding, dilation, {conv_tail})"
    elif op == "conv_transpose2d":
        sig = f"_src_numel: usize, w_out: usize, h_out: usize, stride: usize, padding: usize, out_padding: usize, dilation: usize, info: *const usize, src: *const {T}, kernel: *const {T}, dst: *mut {T}"
        body = f"conv_transpose2d(w_out, h_out, stride, padding, out_padding, dilation, {conv_tail})"
    elif op == "avg_pool2d":
        sig = f"_src_numel: usize, w_k: usize, h_k: usize, w_stride: usize, h_stride: usize, info: *const usize, src: *const {T}, dst: *mut {T}"
        body = f"avg_pool2d(w_k, h_k, w_stride, h_stride, info, src, dst, {za}, {load}, {aadd}, {afin})"
    elif op == "max_pool2d":
        sig = f"_src_numel: usize, w_k: usize, h_k: usize, w_stride: usize, h_stride: usize, info: *const usize, src: *const {T}, dst: *mut {T}"
        body = f"max_pool2d(w_k, h_k, w_stride, h_stride, info, src, dst, {zt}, {mx})"
    elif op == "upsample_nearest2d":
        sig = f"w_out: usize, h_out: usize, w_scale: f64, h_scale: f64, info: *const usize, src: *const {T}, dst: *mut {T}"
        body = "upsample_nearest2d(w_out, h_out, w_scale, h_scale, info, src, dst)"
    elif op == "upsample_bilinear2d":
        sig = f"w_out: usize, h_out: usize, align_corners: u8, has_scale_h: u8, scale_h_factor: f64, has_scale_w: u8, scale_w_factor: f64, info: *const usize, src: *const {T}, dst: *mut {T}"
        body = f"upsample_bilinear2d(w_out, h_out, align_corners, has_scale_h, scale_h_factor, has_scale_w, scale_w_factor, info, src, dst, {tof}, {fromf})"
    elif op == "im2col1d":
        sig = f"dst_numel: usize, l_out: usize, l_k: usize, stride: usize, padding: usize, dilation: usize, info: *const usize, src: *const {T}, dst: *mut {T}"
        body = f"im2col1d(dst_numel, l_out, l_k, stride, padding, dilation, info, src, dst, {zt})"
    elif op == "col2im1d":
        sig = f"dst_el: usize, l_out: usize, l_in: usize, c_out: usize, k_size: usize, stride: usize, src: *const {T}, dst: *mut {T}"
        body = f"col2im1d(dst_el, l_out, l_in, c_out, k_size, stride, src, dst, {zt}, {addt})"
    elif op == "im2col":
        sig = f"dst_numel: usize, h_out: usize, w_out: usize, h_k: usize, w_k: usize, stride: usize, padding: usize, dilation: usize, info: *const usize, src: *const {T}, dst: *mut {T}"
        body = f"im2col(dst_numel, h_out, w_out, h_k, w_k, stride, padding, dilation, info, src, dst, {zt})"
    out.append(f"    #[kernel]\n    pub unsafe fn {n}({sig}) {{\n        unsafe {{ {body}; }}\n    }}\n")

src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src/main.rs")
src = open(src_path).read()
a = src.index("    // GENERATED KERNELS BEGIN\n") + len("    // GENERATED KERNELS BEGIN\n")
b = src.index("    // GENERATED KERNELS END\n")
open(src_path, "w").write(src[:a] + "\n".join(out) + src[b:])
print(len(names), "kernels")
