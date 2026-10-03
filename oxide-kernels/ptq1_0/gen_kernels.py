#!/usr/bin/env python3
"""Regenerate the explicit #[kernel] wrappers between the GENERATED markers of src/main.rs
(PORTING.md rule 14: the #[cuda_module] scan does not see macro-generated kernels)."""
import os, re
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src/main.rs")
MMVQ_P = ("vx: *const u8, vy: *const u8, x_bias: *const f32, gate: *const u8, gate_bias: *const f32, _x_scale: u64, _gate_scale: u64, "
          "glu_op: u32, _pad: u32, dst: *mut {dt}, ncols_x: i32, nrows_x: i32, stride_row_x: i32, stride_col_y: i32, stride_col_dst: i32, "
          "rows_per_cta: i32, bpr_mp: u32, bpr_l: u32, _bpr_d: u32")
MMVQ_A = ("vx, vy, x_bias, gate, gate_bias, glu_op, {dst}, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, rows_per_cta, "
          "bpr_mp, bpr_l")
MMQ_P = ("x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, "
         "_y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, "
         "stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, "
         "stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, "
         "stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32")
MMQ_A = ("x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, "
         "stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, "
         "stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, "
         "stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }")
FIX_P = ("ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, "
         "nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, "
         "nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32")
FIX_A = ("ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, "
         "Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, "
         "Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }")
# (J, fallback) instances of mmq-config-ampere.cuh for GGML_TYPE_PTQ1_0 (same table as Q1_0)
MMQ = [(j, False) for j in (8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128)] + [(j, True) for j in (8, 16, 32, 64, 128)]

out = []
for n in range(1, 5):
    lb = "#[launch_bounds(128, 4)]" if n <= 2 else "#[launch_bounds(128, 3)]"
    for (suf, dt, ty, glu) in (("", "f32", "f32", "false"), ("_bf16", "u16", "B", "false")):
        dst = "dst" if ty == "f32" else f"dst as *mut {ty}"
        out.append(f"    #[kernel] {lb} pub unsafe fn ptq1_0_mmvq_pt_c{n}{suf}({MMVQ_P.format(dt=dt)}) {{ mmvq_pt::<{n}, {glu}, {ty}>({MMVQ_A.format(dst=dst)}) }}")
    out.append(f"    #[kernel] {lb} pub unsafe fn ptq1_0_mmvq_pt_glu_c{n}({MMVQ_P.format(dt='f32')}) {{ mmvq_pt::<{n}, true, f32>({MMVQ_A.format(dst='dst')}) }}")
for (j, fb) in MMQ:
    out.append(f"    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn ptq1_0_mmq_j{j}_f{int(fb)}({MMQ_P}) {{ mul_mat_q::<{j}, {'true' if fb else 'false'}>({MMQ_A}) }}")
for (j, fb) in MMQ:
    out.append(f"    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn ptq1_0_mmq_fixup_j{j}_f{int(fb)}({FIX_P}) {{ stream_k_fixup::<{j}, {'true' if fb else 'false'}>({FIX_A}) }}")

s = open(P).read()
s = re.sub(r"(    // GENERATED KERNELS BEGIN\n).*?(    // GENERATED KERNELS END\n)", lambda m: m.group(1) + "\n".join(out) + "\n" + m.group(2), s, flags=re.S)
open(P, "w").write(s)
print(len(out), "generated kernels")
