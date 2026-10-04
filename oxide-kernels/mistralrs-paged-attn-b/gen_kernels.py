#!/usr/bin/env python3
"""Generates the explicit #[kernel] wrappers of flashinfer_decode.cu / flashinfer_mla_decode.cu between the
GENERATED markers of src/main.rs, and src/instances.rs (the table the launchers and the gate use).
Entry names are read from the reference cubins; each oxide kernel carries the reference's mangled name.

Instances whose NUM_STAGES_SMEM is 1 are only dispatched on devices with compute capability < 8 (not
built: the reference is sm_120a only); they are generated too when STAGES1=1 is set in the environment."""
import subprocess, re, os
HERE = os.path.dirname(os.path.abspath(__file__))
REF = f"{HERE}/../reference/mistralrs-paged-attn-094"  # v0.9.4 (ref094/README)
WITH_S1 = os.environ.get("STAGES1") == "1"
ONLY = os.environ.get("ONLY")  # debugging: only emit kernels whose name matches this regex

def names(cubin):
    out = subprocess.run([f"{HERE}/../tools/cuobjdump", "-sass", f"{REF}/{cubin}"], capture_output=True, text=True).stdout
    return [l.split()[2] for l in out.splitlines() if "Function :" in l]

TY = {"f": "f32", "6__half": "H", "13__nv_bfloat16": "B"}
DT = {"f": 2, "6__half": 0, "13__nv_bfloat16": 1}
kern, table = [], []
# the oxide entry name of each decode instance: `bd<n>` (the opt step of cargo oxide passes every kernel name in one
# argument; 864 mangled decode names exceed Linux's 128 KiB per-argument limit and opt is skipped)
bd = [0]
def short():
    bd[0] += 1
    return f"bd{bd[0] - 1}"
FP8 = "13__nv_fp8_e4m3"
for cub in ["flashinfer_decode.cubin", "flashinfer_mla_decode.cubin"]:
    for n in names(cub):
        # v0.9.4 FP8 E4M3 KV-cache decode instances (BatchDecodeParams<DTypeQ, __nv_fp8_e4m3, DTypeO, int>): batch_decode
        # with KV = E (fp8 bytes); only the NUM_STAGES_SMEM=2 ones unless STAGES1=1, like the others.
        m = re.match(r"_ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj(\d+)ELj(\d+)ELj(\d+)ELj(\d+)ELj(\d+)ELj(\d+)ENS_16DefaultAttentionILb0ELb([01])ELb([01])ELb0EEENS_17BatchDecodeParamsI(f|6__half|13__nv_bfloat16)" + FP8, n)
        if m:
            s, tile, vec, bdx, bdy, bdz, sw, sc, t = m.groups()
            ox = short()
            table.append(f'    Inst {{ kind: Kind::DecodeFp8, name: "{n}", ox: "{ox}", dtype: {DT[t]}, cache: 3, stages: {s}, tile: {tile}, vec: {vec}, bdx: {bdx}, bdy: {bdy}, bdz: {bdz}, sw: {str(sw == "1").lower()}, sc: {str(sc == "1").lower()} }},')
            if s == "1" and not WITH_S1:
                continue
            T = TY[t]
            kern.append(
                f"    /// {n}\n    #[kernel]\n    pub unsafe fn {ox}(q: *const {T}, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut {T}, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {{\n"
                f"        batch_decode::<{T}, E, {s}, {tile}, {vec}, {bdx}, {bdy}, {bdz}, {'true' if sw == '1' else 'false'}, {'true' if sc == '1' else 'false'}>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)\n    }}")
            continue
        m = re.match(r"_ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj(\d+)ELj(\d+)ELj(\d+)ELj(\d+)ELj(\d+)ELj(\d+)ENS_16DefaultAttentionILb0ELb([01])ELb([01])ELb0EEENS_17BatchDecodeParamsI(f|6__half|13__nv_bfloat16)", n)
        if m:
            s, tile, vec, bdx, bdy, bdz, sw, sc, t = m.groups()
            ox = short()
            table.append(f'    Inst {{ kind: Kind::Decode, name: "{n}", ox: "{ox}", dtype: {DT[t]}, cache: {DT[t]}, stages: {s}, tile: {tile}, vec: {vec}, bdx: {bdx}, bdy: {bdy}, bdz: {bdz}, sw: {str(sw == "1").lower()}, sc: {str(sc == "1").lower()} }},')
            if s == "1" and not WITH_S1:
                continue
            T = TY[t]
            kern.append(
                f"    /// {n}\n    #[kernel]\n    pub unsafe fn {ox}(q: *const {T}, k: *const {T}, v: *const {T}, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut {T}, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {{\n"
                f"        batch_decode::<{T}, {T}, {s}, {tile}, {vec}, {bdx}, {bdy}, {bdz}, {'true' if sw == '1' else 'false'}, {'true' if sc == '1' else 'false'}>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)\n    }}")
            continue
        m = re.match(r"_ZN10flashinfer36BatchDecodeWithPagedKVCacheKernelMLAILj(\d+)ELj16ELj2ELj32ELj8ELj1ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_20BatchDecodeParamsMLAI(f|6__half|13__nv_bfloat16)", n)
        if m:
            s, t = m.groups()
            table.append(f'    Inst {{ kind: Kind::Mla, name: "{n}", ox: "{n}", dtype: {DT[t]}, cache: {DT[t]}, stages: {s}, tile: 0, vec: 16, bdx: 32, bdy: 8, bdz: 1, sw: false, sc: false }},')
            if s == "1" and not WITH_S1:
                continue
            T = TY[t]
            kern.append(
                f"    #[kernel]\n    pub unsafe fn {n}(qn: *const {T}, qp: *const {T}, ckv: *const {T}, kpe: *const {T}, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut {T}, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, spc: u32, spk: u32, snc: u32, snk: u32) {{\n"
                f"        mla_decode::<{T}, {s}>(qn, qp, ckv, kpe, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, sm, part, fd, fm, fs, fa, bs, spc, spk, snc, snk)\n    }}")
            continue
        m = re.match(r"_ZN10flashinfer41PersistentVariableLengthMergeStatesKernelILj(\d+)ELj(\d+)ELj(\d+)ELj4E(f|6__half|13__nv_bfloat16)", n)
        if m:
            vec, bdx, bdy, t = m.groups()
            if cub == "flashinfer_mla_decode.cubin":
                # the same template instances as flashinfer_decode.cubin's (one oxide kernel serves both)
                table.append(f'    Inst {{ kind: Kind::MergeMla, name: "{n}", ox: "{n}", dtype: {DT[t]}, cache: {DT[t]}, stages: 4, tile: 0, vec: {vec}, bdx: {bdx}, bdy: {bdy}, bdz: 1, sw: false, sc: false }},')
                continue
            table.append(f'    Inst {{ kind: Kind::Merge, name: "{n}", ox: "{n}", dtype: {DT[t]}, cache: {DT[t]}, stages: 4, tile: 0, vec: {vec}, bdx: {bdx}, bdy: {bdy}, bdz: 1, sw: false, sc: false }},')
            T = TY[t]
            kern.append(
                f"    #[kernel]\n    pub unsafe fn {n}(v: *const {T}, s: *const f32, indptr: *const i32, vm: *mut {T}, smg: *mut f32, max_seq_len: u32, seq_len: *const u32, num_heads: u32) {{\n"
                f"        merge_states::<{T}, {vec}, {bdx}, {bdy}>(v, s, indptr, vm, smg, max_seq_len, seq_len, num_heads)\n    }}")
            continue
        # v0.9.4: reshape_and_cache_flashinfer_kernel<DType, CacheType> (CacheType = DType or __nv_fp8_e4m3)
        m = re.match(r"_ZN20mistralrs_flashinfer35reshape_and_cache_flashinfer_kernelI(f|6__half|13__nv_bfloat16)(S1_|f|13__nv_fp8_e4m3)E", n)
        if m:
            t, c = m.groups()
            fp8 = c == FP8
            table.append(f'    Inst {{ kind: Kind::Reshape, name: "{n}", ox: "{n}", dtype: {DT[t]}, cache: {3 if fp8 else DT[t]}, stages: 0, tile: 0, vec: 0, bdx: 0, bdy: 0, bdz: 0, sw: false, sc: false }},')
            T = {"f": "f32", "6__half": "H", "13__nv_bfloat16": "B"}[t]
            C = "u8" if fp8 else T
            kern.append(
                f"    #[kernel]\n    pub unsafe fn {n}(k: *const {T}, v: *const {T}, kc: *mut {C}, vc: *mut {C}, slot: *const i64, nh: i32, hs: i32, bs: i32, ks: i32, vs: i32, ksc: f32, vsc: f32) {{\n"
                f"        reshape_fi::<{T}, {C}, {'true' if fp8 else 'false'}>(k, v, kc, vc, slot, nh, hs, bs, ks, vs, ksc, vsc)\n    }}")
            continue
        # v0.9.4: gather_kv_cache_flashinfer_kernel<CacheType, OutType>
        m = re.match(r"_ZN20mistralrs_flashinfer33gather_kv_cache_flashinfer_kernelI(f|6__half|13__nv_bfloat16|13__nv_fp8_e4m3)(S1_|f|6__half|13__nv_bfloat16)E", n)
        if m:
            c, o = m.groups()
            fp8 = c == FP8
            if not fp8:
                o = c
            table.append(f'    Inst {{ kind: Kind::Gather, name: "{n}", ox: "{n}", dtype: {DT[o]}, cache: {3 if fp8 else DT[o]}, stages: 0, tile: 0, vec: 0, bdx: 0, bdy: 0, bdz: 0, sw: false, sc: false }},')
            O = {"f": "f32", "6__half": "H", "13__nv_bfloat16": "B"}[o]
            C = "u8" if fp8 else O
            kern.append(
                f"    #[kernel]\n    pub unsafe fn {n}(kc: *const {C}, vc: *const {C}, ko: *mut {O}, vo: *mut {O}, bt: *const i32, cu: *const i32, nt: i32, ns: i32, bs: i32, bts: i32, nkv: i32, hs: i32, ksc: f32, vsc: f32) {{\n"
                f"        gather_fi::<{C}, {O}, {'true' if fp8 else 'false'}>(kc, vc, ko, vo, bt, cu, nt, ns, bs, bts, nkv, hs, ksc, vsc)\n    }}")
            continue
        raise SystemExit(f"unknown reference entry {n}")

if ONLY:
    kern = [k for k in kern if re.search(ONLY, k.split("pub unsafe fn ")[1].split("(")[0]) or re.search(ONLY, k)]
p = f"{HERE}/src/main.rs"
s = open(p).read()
a = s.index("    // GENERATED KERNELS BEGIN\n") + len("    // GENERATED KERNELS BEGIN\n")
b = s.index("    // GENERATED KERNELS END")
s = s[:a] + "\n".join(kern) + "\n" + s[b:]
open(p, "w").write(s)
open(f"{HERE}/src/instances.rs", "w").write(
    "//! GENERATED by gen_kernels.py from the reference cubins: every reference kernel instance.\n"
    "#[derive(Clone, Copy, PartialEq, Eq, Debug)]\n"
    "pub enum Kind { Decode, DecodeFp8, Mla, Merge, MergeMla, Reshape, Gather }\n"
    "#[derive(Clone, Copy, Debug)]\n"
    "pub struct Inst { pub kind: Kind, pub name: &'static str, pub ox: &'static str, pub dtype: u32, pub cache: u32, pub stages: u32, pub tile: u32, pub vec: u32, pub bdx: u32, pub bdy: u32, pub bdz: u32, pub sw: bool, pub sc: bool }\n"
    "pub static INSTANCES: &[Inst] = &[\n" + "\n".join(table) + "\n];\n")
print(len(kern), "kernels,", len(table), "reference instances")
