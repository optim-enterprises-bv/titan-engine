#!/usr/bin/env python3
"""Emit the explicit #[kernel] wrappers between the GENERATED markers in src/main.rs (the
#[cuda_module] scan does not see macro-generated kernels).

  python3 gen_kernels.py            # everything
  MMQ_TYPES=q4_k python3 gen_kernels.py   # all mmvq + quantizers, mmq only for the listed types
  MMQ_TYPES= python3 gen_kernels.py       # no mmq instances (mmvq iteration)
"""
import os

FMTS = ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k"]
DSTS = [("bf16", "u16", "B"), ("f16", "u16", "H"), ("f32", "f32", "f32")]

out = []
for fi, f in enumerate(FMTS):
    for d, cty, dty in DSTS:
        cast = lambda v: v if dty == "f32" else f"{v} as *mut {dty}"
        for n in range(1, 9):
            out.append(
                f"    #[kernel] pub unsafe fn mmvq_gguf_{f}_{d}_plain_cuda{n}(vx: *const u8, vy: *const u8, dst: *mut {cty}, "
                f"ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) "
                f"{{ mmvq::<{fi}, {n}, {dty}>(vx, vy, {cast('dst')}, ncols_x, nrows_x, stride_col_y, stride_col_dst) }}\n")
        for n in range(1, 9):
            out.append(
                f"    #[kernel] pub unsafe fn mmvq_gguf_{f}_{d}_fused_glu_cuda{n}(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut {cty}, "
                f"ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) "
                f"{{ mmvq_glu::<{fi}, {n}, {dty}>(vx_gate, vx_up, vy, {cast('dst')}, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }}\n")
        for n in range(1, 9):
            out.append(
                f"    #[kernel] pub unsafe fn mmvq_gguf_{f}_{d}_fused_qkv_cuda{n}(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, "
                f"q_dst: *mut {cty}, k_dst: *mut {cty}, v_dst: *mut {cty}, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) "
                f"{{ mmvq_qkv::<{fi}, {n}, {dty}>(vx_q, vx_k, vx_v, vy, {cast('q_dst')}, {cast('k_dst')}, {cast('v_dst')}, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }}\n")


# ---- MMQ: quantizers, mul_mat_q per (type, mmq_x, need_check), fixup per (qk, mmq_x, need_check)
MMQ_TYPES = os.environ.get("MMQ_TYPES")
MMQ_TYPES = FMTS if MMQ_TYPES is None else [t for t in MMQ_TYPES.split(",") if t]
# mmq_x values reachable on sm_120 (turing mma: granularity 16 from 48 on).
MMQ_X = [8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128]
LAYOUTS = ["d4", "ds4", "d2s6"]
for li, lay in enumerate(LAYOUTS):
    for it, tn in enumerate(["f32", "f16", "bf16"]):
        out.append(
            f"    #[kernel] pub unsafe fn quantize_mmq_q8_1_{tn}_{lay}(x: *const u8, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i32, ne2: i32) "
            f"{{ quantize_mmq_q8_1::<{it}, {li}>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2) }}\n")
    for it, tn in enumerate(["f32", "f16", "bf16"]):
        cty = ["f32", "u16", "u16"][it]
        out.append(
            f"    #[kernel] pub unsafe fn quantize_mmq_q8_1_glu_{tn}_{lay}(gate: *const {cty}, up: *const {cty}, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, ne0: i64, ne1: i32, activation: i32) "
            f"{{ quantize_mmq_q8_1_glu::<{it}, {li}>(gate as *const u8, up as *const u8, ids, vy, ne00, s01, ne0, ne1, activation) }}\n")
FD = lambda n: f"{n}_mp: u32, {n}_l: u32, {n}_d: u32"
FV = lambda n: f"Fd {{ mp: {n}_mp, l: {n}_l, d: {n}_d }}"
for t in MMQ_TYPES:
    for x in MMQ_X:
        for nc in (0, 1):
            out.append(
                f"    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_{t}_x{x}_nc{nc}(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, "
                f"{FD('bpn')}, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, {FD('cr')}, {FD('ncy')}, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, "
                f"{FD('sr')}, {FD('nsy')}, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, {FD('ntx')}) "
                f"{{ mul_mat_q::<{t.upper()}, {x}, {'true' if nc else 'false'}>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, {FV('bpn')}, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, "
                f"{FV('cr')}, {FV('ncy')}, stride_channel_x, stride_channel_y, stride_channel_dst, {FV('sr')}, {FV('nsy')}, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, {FV('ntx')}) }}\n")
for qk in sorted({256 if t.endswith("_k") else 32 for t in MMQ_TYPES}):
    for x in MMQ_X:
        for nc in (0, 1):
            out.append(
                f"    #[kernel] pub unsafe fn mmq_fixup_qk{qk}_x{x}_nc{nc}(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, "
                f"{FD('bpn')}, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, {FD('ncy')}, stride_channel_dst: i32, {FD('nsy')}, stride_sample_dst: i32, {FD('ntx')}) "
                f"{{ stream_k_fixup::<{qk}, {x}, {'true' if nc else 'false'}>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, {FV('bpn')}, nrows_x, ncols_dst, stride_col_dst, "
                f"{FV('ncy')}, stride_channel_dst, {FV('nsy')}, stride_sample_dst, {FV('ntx')}) }}\n")

p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src/main.rs")
src = open(p).read()
a = src.index("    // GENERATED KERNELS BEGIN\n") + len("    // GENERATED KERNELS BEGIN\n")
b = src.index("    // GENERATED KERNELS END\n")
open(p, "w").write(src[:a] + "".join(out) + src[b:])
print(len(out), "kernels; mmq types", MMQ_TYPES)
