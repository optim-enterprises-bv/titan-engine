#!/usr/bin/env python3
"""Generate the explicit #[kernel] wrappers for the paged-attention template instances (the
#[cuda_module] scan does not see macro-generated kernels) between the GENERATED markers of
src/main.rs."""
import os, re

HEADS = [int(x) for x in os.environ.get("GEN_HEADS", "64,80,96,112,128,192,256,512").split(",")]
BLOCKS = [int(x) for x in os.environ.get("GEN_BLOCKS", "8,16,32").split(",")]
DTS = [("f32", "f32", 4, 4), ("f16", "H", 2, 8), ("bf16", "B", 2, 8)]

def nrt(head, block, vv):
    nvpr = block // vv
    nrpi = 32 // nvpr
    return (head + nrpi - 1) // nrpi

lines = []
for dt, ty, size, vv in DTS:
    for fp8 in (False, True):
        c = "e" if fp8 else "a"
        for block in BLOCKS:
            for head in HEADS:
                n = nrt(head, block, vv)
                args = "q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32"
                call = "q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk"
                g = f"{ty}, {'true' if fp8 else 'false'}, {head}, {block}"
                lines.append(f"    #[kernel] pub unsafe fn pa1_{dt}_{c}_{head}_{block}(o: *mut u8, {args}) {{ paged_attention::<{g}, 0, {n}>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, {call}) }}")
                lines.append(f"    #[kernel] pub unsafe fn pa2_{dt}_{c}_{head}_{block}(es: *mut f32, ml: *mut f32, o: *mut u8, {args}) {{ paged_attention::<{g}, 512, {n}>(es, ml, o as _, {call}) }}")
    for head in HEADS:
        lines.append(f"    #[kernel] pub unsafe fn pa2r_{dt}_{head}(o: *mut u8, es: *const f32, ml: *const f32, t: *const u8, cl: *const u32, mp: i32, sk: *const f32) {{ paged_attention_v2_reduce::<{ty}, {head}>(o as _, es, ml, t as _, cl, mp, sk) }}")

src = open("src/main.rs").read()
begin = src.index("    // GENERATED paged attention kernels (gen_kernels.py) BEGIN\n")
end = src.index("    // GENERATED END\n")
head_line = "    // GENERATED paged attention kernels (gen_kernels.py) BEGIN\n"
src = src[:begin] + head_line + "\n".join(lines) + "\n" + src[end:]
open("src/main.rs", "w").write(src)
print(len(lines), "kernels")

# flash_attn_sinks: (HEAD_DIM, BC)
FA = [(h, b) for h, b in [(64, 64), (80, 32), (96, 32), (112, 32), (128, 32), (192, 16), (256, 16)] if h in HEADS]
fl = []
for dt, ty, size, vv in DTS:
    for head, bc in FA:
        ept = (head + 31) // 32
        fl.append(f"    #[kernel] pub unsafe fn fas_{dt}_{head}(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, sc: f32, ql: i32, kl: i32, nh: i32, nkv: i32, w: i32) {{ flash_attn_sinks::<{ty}, {head}, {bc}, {ept}, false>(q as _, k as _, v as _, o as _, sk, core::ptr::null(), core::ptr::null(), sc, ql, kl, nh, nkv, w) }}")
        fl.append(f"    #[kernel] pub unsafe fn fasv_{dt}_{head}(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, cq: *const u32, ck: *const u32, sc: f32, mq: i32, nh: i32, nkv: i32, w: i32) {{ flash_attn_sinks::<{ty}, {head}, {bc}, {ept}, true>(q as _, k as _, v as _, o as _, sk, cq, ck, sc, mq, 0, nh, nkv, w) }}")
src = open("src/main.rs").read()
b = src.index("    // GENERATED flash_attn_sinks kernels BEGIN\n")
e = src.index("    // GENERATED FA END\n")
src = src[:b] + "    // GENERATED flash_attn_sinks kernels BEGIN\n" + "\n".join(fl) + "\n" + src[e:]
open("src/main.rs", "w").write(src)
print(len(fl), "flash kernels")
