#!/usr/bin/env python3
"""Generates the 18 attention_prep #[kernel] wrappers between the GENERATED AP markers of src/main.rs."""
import re
TYPES = [("f", "f32", "f32", "IfLb{n}EEvPKT_S2_"), ("13__nv_bfloat16", "B", "bf", ""), ("6__half", "H", "h", "")]
out = []
for tm, rt, _, _ in TYPES:
    for neox in (0, 1):
        nb = "true" if neox else "false"
        if tm == "f":
            qk = f"_Z23qk_rms_norm_rope_kernelIfLb{neox}EEvPKT_S2_S2_S2_S2_S2_PS0_S3_lllllllliiiiiiiff"
            qkp = f"_Z33qk_rms_norm_rope_positions_kernelIfLb{neox}EEvPKT_S2_S2_S2_S2_S2_PKjPS0_S5_lllllllliiiiiiff"
            qkv = f"_Z34qkv_rms_norm_rope_positions_kernelIfLb{neox}EEvPKT_S2_S2_S2_S2_S2_S2_S2_PKjPS0_S5_S5_lllllllllllliiiiiifff"
        else:
            qk = f"_Z23qk_rms_norm_rope_kernelI{tm}Lb{neox}EEvPKT_S3_S3_S3_S3_S3_PS1_S4_lllllllliiiiiiiff"
            qkp = f"_Z33qk_rms_norm_rope_positions_kernelI{tm}Lb{neox}EEvPKT_S3_S3_S3_S3_S3_PKjPS1_S6_lllllllliiiiiiff"
            qkv = f"_Z34qkv_rms_norm_rope_positions_kernelI{tm}Lb{neox}EEvPKT_S3_S3_S3_S3_S3_S3_S3_PKjPS1_S6_S6_lllllllllllliiiiiifff"
        T = rt
        out.append(f"""    #[kernel]
    pub unsafe fn {qk}(
        q: *const {T}, k: *const {T}, qw: *const {T}, kw: *const {T}, cos: *const {T}, sin: *const {T}, q_out: *mut {T}, k_out: *mut {T},
        qsb: i64, qsh: i64, qss: i64, qsd: i64, ksb: i64, ksh: i64, kss: i64, ksd: i64, batch: i32, q_heads: i32, k_heads: i32,
        seq_len: i32, head_dim: i32, rot_dim: i32, cbs: i32, q_eps: f32, k_eps: f32,
    ) {{
        qk_norm_rope::<{T}, {nb}, false>(q, k, qw, kw, cos, sin, core::ptr::null(), q_out, k_out, qsb, qsh, qss, qsd, ksb, ksh, kss, ksd, batch, q_heads, k_heads, seq_len, head_dim, rot_dim, cbs, q_eps, k_eps)
    }}
    #[kernel]
    pub unsafe fn {qkp}(
        q: *const {T}, k: *const {T}, qw: *const {T}, kw: *const {T}, cos: *const {T}, sin: *const {T}, positions: *const u32,
        q_out: *mut {T}, k_out: *mut {T}, qsb: i64, qsh: i64, qss: i64, qsd: i64, ksb: i64, ksh: i64, kss: i64, ksd: i64, batch: i32,
        q_heads: i32, k_heads: i32, seq_len: i32, head_dim: i32, rot_dim: i32, q_eps: f32, k_eps: f32,
    ) {{
        qk_norm_rope::<{T}, {nb}, true>(q, k, qw, kw, cos, sin, positions, q_out, k_out, qsb, qsh, qss, qsd, ksb, ksh, kss, ksd, batch, q_heads, k_heads, seq_len, head_dim, rot_dim, 0, q_eps, k_eps)
    }}
    #[kernel]
    pub unsafe fn {qkv}(
        q: *const {T}, k: *const {T}, v: *const {T}, qw: *const {T}, kw: *const {T}, vw: *const {T}, cos: *const {T}, sin: *const {T},
        positions: *const u32, q_out: *mut {T}, k_out: *mut {T}, v_out: *mut {T}, qsb: i64, qsh: i64, qss: i64, qsd: i64, ksb: i64,
        ksh: i64, kss: i64, ksd: i64, vsb: i64, vsh: i64, vss: i64, vsd: i64, batch: i32, q_heads: i32, k_heads: i32, seq_len: i32,
        head_dim: i32, rot_dim: i32, q_eps: f32, k_eps: f32, v_eps: f32,
    ) {{
        qkv_norm_rope::<{T}, {nb}>(q, k, v, qw, kw, vw, cos, sin, positions, q_out, k_out, v_out, qsb, qsh, qss, qsd, ksb, ksh, kss, ksd, vsb, vsh, vss, vsd, batch, q_heads, k_heads, seq_len, head_dim, rot_dim, q_eps, k_eps, v_eps)
    }}""")
p = "src/main.rs"
s = open(p).read()
a = s.index("    // GENERATED AP KERNELS BEGIN\n") + len("    // GENERATED AP KERNELS BEGIN\n")
b = s.index("    // GENERATED AP KERNELS END")
s = s[:a] + "\n".join(out) + "\n" + s[b:]
open(p, "w").write(s)
