#!/usr/bin/env python3
"""Generates the explicit #[kernel] wrappers of sort.cu (bitonic, fused RMSNorm, MoE router top-k)
between the GENERATED markers of src/main.rs. Entry names are read from the reference cubin."""
import subprocess, re, os
HERE = os.path.dirname(os.path.abspath(__file__))
names = subprocess.run([f"{HERE}/../tools/cuobjdump", "-sass", f"{HERE}/../reference/mistralrs-core/sort.cubin"],
                       capture_output=True, text=True).stdout
names = [l.split()[2] for l in names.splitlines() if "Function :" in l]
TY = {"f": "f32", "d": "f64", "h": "u8", "j": "u32", "l": "i64", "6__half": "H", "13__nv_bfloat16": "B"}
out = []
for n in names:
    m = re.match(r"_Z19bitonic_sort_kernelI(f|d|h|j|l|6__half|13__nv_bfloat16)Lb([01])E", n)
    if m:
        t = TY[m.group(1)]; asc = "true" if m.group(2) == "1" else "false"
        out.append(f"    #[kernel]\n    pub unsafe fn {n}(arr: *mut {t}, dst: *mut u32, j: i32, k: i32) {{\n        bitonic::<{t}, {asc}>(arr, dst, j, k)\n    }}")
        continue
    m = re.match(r"_Z(24rms_norm_residual|29rms_norm_residual_vec8)_kernelI(f|6__half|13__nv_bfloat16)E", n)
    if m:
        t = TY[m.group(2)]
        f = "rms_residual_vec8::<%s>" % t if "vec8" in m.group(1) else "rms_residual::<%s, false>" % t
        out.append(f"    #[kernel]\n    pub unsafe fn {n}(x: *const {t}, r: *const {t}, w: *const {t}, s: *const {t}, d: *mut {t}, ncols: i32, eps: f32) {{\n        {f}(x, r, w, s, d, ncols, eps)\n    }}")
        continue
    m = re.match(r"_Z(38rms_norm_residual_then_rms_norm|43rms_norm_residual_then_rms_norm_vec8)_kernelI(f|6__half|13__nv_bfloat16)E", n)
    if m:
        t = TY[m.group(2)]; v = "true" if "vec8" in m.group(1) else "false"
        out.append(f"    #[kernel]\n    pub unsafe fn {n}(x: *const {t}, r: *const {t}, rw: *const {t}, s: *const {t}, nw: *const {t}, rd: *mut {t}, nd: *mut {t}, ncols: i32, re: f32, ne: f32) {{\n        rms_residual_then::<{t}, {v}>(x, r, rw, s, nw, rd, nd, ncols, re, ne)\n    }}")
        continue
    m = re.match(r"_Z26rms_norm_strided_4d_kernelI(f|6__half|13__nv_bfloat16)E", n)
    if m:
        t = TY[m.group(1)]
        out.append(f"    #[kernel]\n    pub unsafe fn {n}(x: *const {t}, w: *const {t}, d: *mut {t}, sb: i64, sh: i64, ss: i64, sd: i64, b: i32, h: i32, s: i32, hd: i32, eps: f32) {{\n        rms_strided_4d::<{t}>(x, w, d, sb, sh, ss, sd, b, h, s, hd, eps)\n    }}")
        continue
    m = re.match(r"_Z22moe_router_topk_kernelI(f|6__half|13__nv_bfloat16)Li(\d+)ELb([01])ELb([01])E", n)
    if m:
        t = TY[m.group(1)]; ne = m.group(2); hb = "true" if m.group(3) == "1" else "false"; hs = "true" if m.group(4) == "1" else "false"
        out.append(f"    #[kernel]\n    pub unsafe fn {n}(l: *const {t}, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {{\n        moe_router::<{t}, {ne}, {hb}, {hs}>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)\n    }}")
        continue
sort_k = [o for o in out if "router" not in o]
router_k = [o for o in out if "router" in o]
p = f"{HERE}/src/main.rs"
s = open(p).read()
def put(s, tag, body):
    a = s.index(f"    // GENERATED {tag} BEGIN\n") + len(f"    // GENERATED {tag} BEGIN\n")
    b = s.index(f"    // GENERATED {tag} END")
    return s[:a] + "\n".join(body) + "\n" + s[b:]
s = put(s, "SORT KERNELS", sort_k)
s = put(s, "ROUTER KERNELS", router_k)
open(p, "w").write(s)
print(len(sort_k), len(router_k))
