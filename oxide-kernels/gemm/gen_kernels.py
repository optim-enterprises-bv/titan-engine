#!/usr/bin/env python3
"""Writes the explicit #[kernel] wrappers between the GENERATED markers of src/main.rs (the #[cuda_module] scan does not
see macro-generated kernels). Every GEMM / GEMV kernel has the same parameter list (see the crate doc)."""
import re
P = "src/main.rs"
PARAMS = ("a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, "
          "b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, "
          "sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32")
ARGS = "a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta"
ARGSW = ARGS + ", ws, nsplit"
DT = {"f32": "DT_F32", "f16": "DT_F16", "bf16": "DT_BF16"}
SHM = "cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8)"
out = []
def k(name, bounds, smem, body, doc):
    out.append(f"    /// {doc}\n    #[kernel]\n    #[launch_bounds({bounds})]\n    pub unsafe fn {name}({PARAMS}) {{")
    if smem:
        out.append(f"        static mut SMEM: SharedArray<{smem[0]}, {smem[1]}, 16> = SharedArray::UNINIT;")
        out.append(f"        let sm = {SHM};")
    out.append(f"        {body}\n    }}\n")
LAY = {(False, False): "kk", (False, True): "kn", (True, False): "mk", (True, True): "mn"}
for dt in ("f16", "bf16"):
    for (at, bt), l in LAY.items():
        for bm, words, lb in ((128, 24576, "256, 1"), (64, 18432, "256, 2"), (32, 15360, "256, 2")):
            k(f"tc_{dt}_{l}_{bm}", lb, ("u16", words),
              f"tc_body::<{DT[dt]}, {str(at).lower()}, {str(bt).lower()}, {bm}>(sm, {ARGSW});",
              f"tensor cores, {dt}, A {'m' if at else 'k'}-contiguous, B {'n' if bt else 'k'}-contiguous, BM {bm}")
for dt in ("f32", "f16", "bf16"):
    for (at, bt), l in LAY.items():
        for t, words, lb in ((128, 4224, "256, 2"), (64, 2176, "256, 2")):
            k(f"simt_{dt}_{l}_{t}", lb, ("f32", words),
              f"simt_body::<{DT[dt]}, {str(at).lower()}, {str(bt).lower()}, {t}>(sm, {ARGSW});",
              f"SIMT {t} x {t}, {dt}, thread mapping for A {'m' if at else 'k'}-contiguous, B {'n' if bt else 'k'}-contiguous")
for dt in ("f32", "f16", "bf16"):
    for r in (1, 2, 4, 8):
        for vec in (True, False):
            v = "_v" if vec else ""
            k(f"gemv_k{r}_{dt}{v}", "256, 1", None,
              f"gemv_k_body::<{DT[dt]}, {r}, {str(vec).lower()}, 8>(0, {ARGS});",
              f"GEMV, {dt}, m <= {r}, B k-contiguous, a warp per column{', 16-byte loads' if vec else ''}")
            k(f"gemv_ks{r}_{dt}{v}", "256, 1", ("f32", 64),
              f"gemv_k_body::<{DT[dt]}, {r}, {str(vec).lower()}, 1>(sm, {ARGS});",
              f"GEMV, {dt}, m <= {r}, B k-contiguous, a block per column (8 warps split k){', 16-byte loads' if vec else ''}")
        k(f"gemv_n{r}_{dt}", "256, 1", ("f32", 2048),
          f"gemv_n_body::<{DT[dt]}, {r}>(sm, {ARGS});", f"GEMV, {dt}, m <= {r}, B n-contiguous")
for dt in ("f32", "f16", "bf16"):
    k(f"gemm_ref_{dt}", "256, 1", None, f"ref_body::<{DT[dt]}>({ARGS});", f"f64-accumulating reference, {dt} (diagnostics)")
for dt in ("f32", "f16", "bf16"):
    out.append(f"""    /// split-K merge + epilogue, {dt}
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn splitk_reduce_{dt}(ws: *const f32, nsplit: i32, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sd: i64, sc: i64, alpha: f32, beta: f32) {{
        reduce_body::<{DT[dt]}>(ws, nsplit, d, c, bias, m, n, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sd, sc, alpha, beta);
    }}
""")
gen = "\n".join(out)
s = open(P).read()
s = re.sub(r"(    // BEGIN GENERATED\n).*?(    // END GENERATED\n)", lambda mm: mm.group(1) + gen + mm.group(2), s, flags=re.S)
open(P, "w").write(s)
print(f"{gen.count('#[kernel]')} kernels")
