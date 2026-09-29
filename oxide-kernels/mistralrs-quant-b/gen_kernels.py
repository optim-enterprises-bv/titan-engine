#!/usr/bin/env python3
"""Generates the explicit #[kernel] wrappers (between GENERATED markers in src/main.rs)."""
import re, pathlib
T = [("f32", "f32"), ("f16", "H"), ("bf16", "B")]
GS = [32, 64, 128]
out = []
def k(name, params, body):
    out.append(f"    #[kernel] pub unsafe fn {name}({params}) {{ {body} }}")
# ---- AFQ
for tn, ty in T:
    for b in [2, 3, 4, 6, 8]:
        for g in GS:
            wq = "*const u8" if b in (3, 6) else "*const u32"
            k(f"afq_dequantize_{b}bit_gs{g}_{tn}", f"w_q: {wq}, scales: *const {ty}, biases: *const {ty}, output: *mut {ty}, rows: i32, cols: i32",
              f"afq_dequant::<{ty}, {b}, {g}>(w_q as *const u8, scales, biases, output, rows, cols)")
            k(f"afq_qmv_{b}bit_gs{g}_{tn}", f"x: *const {ty}, w_q: {wq}, scales: *const {ty}, biases: *const {ty}, y: *mut {ty}, m: i32, n: i32, kk: i32",
              f"afq_qmv::<{ty}, {b}, {g}>(x, w_q as *const u8, scales, biases, y, m, n, kk)")
    for b in [2, 4, 8]:
        for g in GS:
            k(f"afq_quantize_{b}bit_gs{g}_{tn}", f"w: *const {ty}, w_q: *mut u32, scales: *mut {ty}, biases: *mut {ty}, rows: i32, cols: i32",
              f"afq_quantize::<{ty}, {b}, {g}>(w, w_q, scales, biases, rows, cols)")
            k(f"afq_qmm_{b}bit_gs{g}_{tn}", f"x: *const {ty}, w_q: *const u32, scales: *const {ty}, biases: *const {ty}, y: *mut {ty}, m: i32, n: i32, kk: i32",
              f"afq_qmm::<{ty}, {b}, {g}>(x, w_q as *const u8, scales, biases, y, m, n, kk)")

# ---- FP8
for tn, ty in T:
    k(f"fp8_to_{tn}", f"input: *const u8, output: *mut {ty}, n: usize", "fp8_to_dtype(input, output, n)")
    k(f"{tn}_to_fp8", f"input: *const {ty}, output: *mut u8, n: usize", "dtype_to_fp8(input, output, n)")
    k(f"dequant_fp8_vector_{tn}", f"weight: *const u8, scale: *const f32, output: *mut {ty}, n: usize", "dequant_fp8_vector(weight, scale, output, n)")
    k(f"quant_fp8_vector_{tn}", f"input: *const {ty}, weight: *mut u8, scale: *mut f32, n: usize", "quant_fp8_vector(input, weight, scale, n)")
    k(f"dequant_fp8_blockwise_{tn}", f"weight: *const u8, scale: *const f32, output: *mut {ty}, h: i32, w: i32, rs: i32, ss: i32, by: i32, bx: i32", "dequant_fp8_blockwise(weight, scale, output, h, w, rs, ss, by, bx)")
    k(f"quant_fp8_blockwise_{tn}", f"input: *const {ty}, weight: *mut u8, scale: *mut f32, h: i32, w: i32, rs: i32, ss: i32, by: i32, bx: i32", "quant_fp8_blockwise(input, weight, scale, h, w, rs, ss, by, bx)")
for tn, ty in T[1:]:
    k(f"fp8_matmul_{tn}", f"input: *const {ty}, weight: *const u8, ws: *const f32, output: *mut {ty}, m: i32, n: i32, kk: i32, srs: i32, by: i32, bx: i32", "fp8_matmul_tiled(input, weight, ws, output, m, n, kk, srs, by, bx)")
    k(f"fp8_moe_gemm_{tn}", f"input: *const {ty}, weights: *const u8, ws: *const f32, indices: *const u32, output: *mut {ty}, nt: i32, topk: i32, ne: i32, n: i32, kk: i32, srs: i32, by: i32, bx: i32, has_topk: u8", "fp8_moe_gemm(input, weights, ws, indices, output, nt, topk, ne, n, kk, srs, by, bx, has_topk)")

# ---- GPTQ
for b in [2, 3, 4, 8]:
    for mc in range(1, 9):
        k(f"gptq_gemm_{b}bit_m{mc}", "a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32",
          f"gptq_gemm::<{b}, {mc}>(a, bq, qz, sc, c, m, n, kk, groups, perm)")
    k(f"reconstruct_exllama_{b}bit", "bq: *const u32, perm: *const i32, qz: *const u32, sc: *const u16, size_k: i32, size_n: i32, groups: i32, out: *mut u16",
      f"reconstruct_exllama::<{b}>(bq, perm, qz, sc, size_k, size_n, groups, out)")
    k(f"reconstruct_gptq_{b}bit", "w: *const u32, sc: *const u16, qz: *const u32, g_idx: *const i32, height: i32, width: i32, group: i32, out: *mut u16",
      f"reconstruct_gptq::<{b}>(w, sc, qz, g_idx, height, width, group, out)")
    k(f"gptq_shuffle_{b}bit", "bq: *mut u32, size_k: i32, size_n: i32", f"gptq_shuffle::<{b}>(bq, size_k, size_n)")
    if b == 3:
        k("make_sequential_3bit", "w: *const u32, w_new: *mut u32, q_perm: *const i32, w_width: i32", "make_sequential_3(w, w_new, q_perm, w_width)")
    else:
        k(f"make_sequential_{b}bit", "w: *const u32, w_new: *mut u32, q_perm: *const i32, w_width: i32", f"make_sequential::<{b}>(w, w_new, q_perm, w_width)")
for b in [4, 8]:
    k(f"gemm_half_q_half_alt_{b}bit", "vec: *const u32, mat: *const u32, mul_out: *mut u16, sc: *const u16, zeros: *const u32, g_idx: *const i32, batch: i32, height: i32, width: i32",
      f"gptq_alt::<{b}>(vec, mat, mul_out, sc, zeros, g_idx, batch, height, width)")

# ---- MXFP4
for tn, ty in T[1:]:
    mm = f"input: *const {ty}, weight: *const u8, ws: *const u8, bias: *const {ty}, output: *mut {ty}, m: i32, n: i32, kk: i32, has_bias: u8"
    moe = f"input: *const {ty}, weights: *const u8, ws: *const u8, biases: *const {ty}, indices: *const u32, output: *mut {ty}, nt: i32, topk: i32, ne: i32, n: i32, kk: i32, has_bias: u8, has_topk: u8"
    k(f"mxfp4_vecmat_{tn}", mm, "mxfp4_vecmat(input, weight, ws, bias, output, m, n, kk, has_bias)")
    k(f"mxfp4_matmul_tiled_{tn}", mm, "mxfp4_matmul_tiled(input, weight, ws, bias, output, m, n, kk, has_bias)")
    k(f"mxfp4_moe_gemm_{tn}", moe, "mxfp4_moe_gemm(input, weights, ws, biases, indices, output, nt, topk, ne, n, kk, has_bias, has_topk)")
    k(f"mxfp4_moe_grouped_tiled_{tn}", moe, "mxfp4_moe_grouped_tiled(input, weights, ws, biases, indices, output, nt, topk, ne, n, kk, has_bias, has_topk)")
    k(f"mxfp4_matmul_wmma_{tn}", mm, "mxfp4_matmul_wmma(input, weight, ws, bias, output, m, n, kk, has_bias)")
    k(f"mxfp4_moe_grouped_wmma_{tn}", moe, "mxfp4_moe_grouped_wmma(input, weights, ws, biases, indices, output, nt, topk, ne, n, kk, has_bias, has_topk)")

# ---- bitsandbytes / grouped GEMM
for tn, ty in T:
    for dt, dn in [(0, "int8"), (1, "fp4"), (2, "nf4")]:
        k(f"bnb_dequant_{tn}_{dn}", f"code: *const f32, a: *const u8, absmax: *const f32, out: *mut {ty}, blocksize: i32, n: i32",
          f"bnb_dequant::<{ty}, {dt}>(code, a, absmax, out, blocksize, n)")
for cfg in ["large", "medium", "small"]:
    k(f"grouped_mm_2x_{cfg}", "a: *const *const u16, b: *const *const u16, d: *const *mut u16, ps: *const i32, pc: i32, lda: *const i64, ldb: *const i64, ldd: *const i64",
      "grouped_mm(a, b, d, ps, pc, lda, ldb, ldd)")

ALL_NAMES = []
def emit(path, lines):
    ALL_NAMES.extend(re.findall(r"pub unsafe fn (\w+)\(", "\n".join(lines)))
    p = pathlib.Path(__file__).parent / path
    s = p.read_text()
    s = re.sub(r"(    // GENERATED KERNELS BEGIN\n).*?(    // GENERATED KERNELS END)", lambda m: m.group(1) + "\n".join(lines) + "\n" + m.group(2), s, flags=re.S)
    p.write_text(s)
    print(len(lines), "kernels ->", path)
emit("src/main.rs", out)
out = []
# ---- Marlin
MCFG = [(256, 8, 8), (256, 16, 4), (128, 8, 4), (128, 4, 8)]
for q, zp in [("gptq", "false"), ("awq", "true")]:
    for tn, ty in T[1:]:
        for (th, nb, kb) in MCFG:
            for mb in range(1, 5):
                for gb in [-1, 4, 8]:
                    gn = "m1" if gb == -1 else str(gb)
                    k(f"marlin_{q}_{tn}_t{th}_m{mb}_n{nb}_k{kb}_g{gn}",
                      "a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32",
                      f"let _ = g_idx; marlin::<{ty}, {th}, {mb}, {nb}, {kb}, {gb}, {zp}>(a, b, c, s, zp, m, n, kk, num_groups, locks)")
for bits in [4, 8]:
    for perm in ["true", "false"]:
        k(f"gptq_marlin_repack_{bits}_{'perm' if perm == 'true' else 'noperm'}", "w: *const u32, perm: *const u32, out: *mut u32, size_k: i32, size_n: i32",
          f"marlin_repack::<{bits}, {perm}, false>(w, perm, out, size_k, size_n)")
    k(f"awq_marlin_repack_{bits}", "w: *const u32, out: *mut u32, size_k: i32, size_n: i32", f"marlin_repack::<{bits}, false, true>(w, core::ptr::null(), out, size_k, size_n)")

emit("marlin-kernels/src/main.rs", out)

# ---------------------------------------------------------------- launchers (src/launch.rs)
L = []
RT = {"f32": "f32", "f16": "u16", "bf16": "u16"}
def l(sig, body):
    L.append(f"pub unsafe extern \"C\" fn {sig} {{\n    unsafe {{ {body} }}\n}}")
for tn, _ in T:
    r = RT[tn]
    for b in [2, 3, 4, 6, 8]:
        for g in GS:
            wq = "*const u8" if b in (3, 6) else "*const u32"
            n = f"afq_dequantize_{b}bit_gs{g}_{tn}"
            l(f"{n}(w_q: {wq}, scales: *const {r}, biases: *const {r}, output: *mut {r}, rows: i32, cols: i32)",
              f'afq_dequant("{n}", w_q as _, scales as _, biases as _, output as _, rows, cols)')
            n = f"afq_qmv_{b}bit_gs{g}_{tn}"
            l(f"{n}(x: *const {r}, w_q: {wq}, scales: *const {r}, biases: *const {r}, y: *mut {r}, m: i32, n: i32, k: i32)",
              f'afq_qmv("{n}", x as _, w_q as _, scales as _, biases as _, y as _, m, n, k)')
    for b in [2, 4, 8]:
        for g in GS:
            n = f"afq_quantize_{b}bit_gs{g}_{tn}"
            l(f"{n}(w: *const {r}, w_q: *mut u32, scales: *mut {r}, biases: *mut {r}, rows: i32, cols: i32)",
              f'afq_quant("{n}", {b}, {g}, w as _, w_q as _, scales as _, biases as _, rows, cols)')
            n = f"afq_qmm_{b}bit_gs{g}_{tn}"
            l(f"{n}(x: *const {r}, w_q: *const u32, scales: *const {r}, biases: *const {r}, y: *mut {r}, m: i32, n: i32, k: i32)",
              f'afq_qmm("{n}", x as _, w_q as _, scales as _, biases as _, y as _, m, n, k)')
p = pathlib.Path(__file__).parent / "src/launch.rs"
s = p.read_text()
s = re.sub(r"(// GENERATED LAUNCHERS BEGIN\n).*?(// GENERATED LAUNCHERS END)", lambda m: m.group(1) + "\n".join(L) + "\n" + m.group(2), s, flags=re.S)
p.write_text(s)
print(len(L), "launchers")

# ---------------------------------------------------------------- gate tables (src/gen_cref.rs, src/gen_pairs.rs)
# C declarations: same signatures as the Rust launchers.
decls = []
for x in L:
    sig = x.split("\n")[0][len('pub unsafe extern "C" fn '):-2]
    decls.append(f"    pub fn {sig};")
(pathlib.Path(__file__).parent / "src/gen_cref.rs").write_text("// GENERATED by gen_kernels.py\nunsafe extern \"C\" {\n" + "\n".join(decls) + "\n}\n")
P = []
def table(fn, ty, names):
    # names: list over t of list of names; index by (t, key)
    arms = []
    for key, n in names:
        arms.append(f"        {key} => unsafe {{ (std::mem::transmute::<*const (), {ty}>(cref::{n} as *const ()), std::mem::transmute::<*const (), {ty}>(ox::{n} as *const ())) }},")
    P.append(f"#[allow(clippy::type_complexity)]\nfn {fn}(t: usize, bits: i32, gi: usize) -> ({ty}, {ty}) {{\n    match (t, bits, gi) {{\n" + "\n".join(arms) + "\n        _ => unreachable!(),\n    }\n}")
def afq_names(pfx, bitsl):
    out = []
    for ti, (tn, _) in enumerate(T):
        for b in bitsl:
            for gi, g in enumerate(GS):
                out.append((f"({ti}, {b}, {gi})", f"afq_{pfx}_{b}bit_gs{g}_{tn}"))
    return out
P.append("type AfqDq = unsafe extern \"C\" fn(*const u32, *const u16, *const u16, *mut u16, i32, i32);")
P.append("type AfqQ = unsafe extern \"C\" fn(*const u16, *mut u32, *mut u16, *mut u16, i32, i32);")
P.append("type AfqMv = unsafe extern \"C\" fn(*const u16, *const u32, *const u16, *const u16, *mut u16, i32, i32, i32);")
table("afq_dequant_pair", "AfqDq", afq_names("dequantize", [2, 3, 4, 6, 8]))
table("afq_quant_pair", "AfqQ", afq_names("quantize", [2, 4, 8]))
table("afq_qmv_pair", "AfqMv", afq_names("qmv", [2, 3, 4, 6, 8]))
table("afq_qmm_pair", "AfqMv", afq_names("qmm", [2, 4, 8]))
(pathlib.Path(__file__).parent / "src/gen_pairs.rs").write_text("// GENERATED by gen_kernels.py\n" + "\n\n".join(P) + "\n")

(pathlib.Path(__file__).parent / "src/gen_names.rs").write_text("// GENERATED by gen_kernels.py: every oxide kernel entry (both modules)\npub const ALL_KERNELS: &[&str] = &[\n" + "".join(f"    \"{n}\",\n" for n in ALL_NAMES) + "];\n")

# ---------------------------------------------------------------- Marlin reference-instance table
import subprocess
R = pathlib.Path.home() / "titan-engine/oxide-kernels"
rows = []
for cub, q, tn in [("marlin_matmul_f16", "gptq", "f16"), ("marlin_matmul_bf16", "gptq", "bf16"), ("marlin_matmul_awq_f16", "awq", "f16"), ("marlin_matmul_awq_bf16", "awq", "bf16")]:
    sass = subprocess.run([str(R / "tools/cuobjdump"), "-sass", str(R / f"reference/mistralrs-quant/{cub}.cubin")], capture_output=True, text=True).stdout
    for mangled in re.findall(r"Function : (\S+)", sass):
        dem = subprocess.run(["c++filt", mangled], capture_output=True, text=True).stdout.strip()
        m = re.search(r"Marlin<[^,]+, (\d+), (\d+), (\d+), (\d+), 4, (-?\d+),", dem)
        th, mb, nb, kb, gb = m.groups()
        g = "m1" if gb == "-1" else gb
        rows.append((cub, mangled, f"marlin_{q}_{tn}_t{th}_m{mb}_n{nb}_k{kb}_g{g}", int(th), int(mb), int(nb), int(kb), int(gb), q == "awq", tn == "bf16"))
(pathlib.Path(__file__).parent / "src/gen_marlin_names.rs").write_text(
    "// GENERATED by gen_kernels.py: (cubin, mangled reference entry, oxide entry, threads, m_blocks, n_blocks, k_blocks, group_blocks, awq, bf16)\n"
    "pub const MARLIN_INSTANCES: &[(&str, &str, &str, u32, i32, i32, i32, i32, bool, bool)] = &[\n"
    + "".join(f'    ("{r[0]}", "{r[1]}", "{r[2]}", {r[3]}, {r[4]}, {r[5]}, {r[6]}, {r[7]}, {str(r[8]).lower()}, {str(r[9]).lower()}),\n' for r in rows) + "];\n")
print(len(rows), "marlin reference instances")
