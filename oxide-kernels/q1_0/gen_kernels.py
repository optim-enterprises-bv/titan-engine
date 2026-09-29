#!/usr/bin/env python3
"""Regenerate the explicit #[kernel] wrappers between the GENERATED markers of src/main.rs
(PORTING.md rule 14: the #[cuda_module] scan does not see macro-generated kernels)."""
import re
P = "src/main.rs"
MMVQ_P = ("vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, "
          "_glu_limit: f32, dst: *mut {dt}, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, "
          "stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, "
          "sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32")
MMVQ_A = ("vx, vy, ids, {dst}, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, "
          "stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst")
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

# (J, fallback) instances of mmq-config-ampere.cuh for GGML_TYPE_Q1_0
MMQ = [(j, False) for j in (8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128)] + [(j, True) for j in (8, 16, 32, 64, 128)]

out = []
for (dt, suf, ty) in (("f32", "", "f32"), ("u16", "_f16", "H"), ("u16", "_bf16", "B")):
    for (n, sk) in [(1, True)] + [(n, False) for n in range(1, 9)]:
        name = f"q1_0_mmvq_{n}{'_small_k' if sk else ''}{suf}"
        dst = "dst" if ty == "f32" else f"dst as *mut {ty}"
        out.append(f"    #[kernel] pub unsafe fn {name}({MMVQ_P.format(dt=dt)}) {{ mmvq::<{n}, {'true' if sk else 'false'}, {ty}>({MMVQ_A.format(dst=dst)}) }}")
for (j, fb) in MMQ:
    out.append(f"    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn q1_0_mmq_j{j}_f{int(fb)}({MMQ_P}) {{ mul_mat_q::<{j}, {'true' if fb else 'false'}>({MMQ_A}) }}")
for (j, fb) in MMQ:
    out.append(f"    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn q1_0_mmq_fixup_j{j}_f{int(fb)}({FIX_P}) {{ stream_k_fixup::<{j}, {'true' if fb else 'false'}>({FIX_A}) }}")

s = open(P).read()
s = re.sub(r"(    // GENERATED KERNELS BEGIN\n).*?(    // GENERATED KERNELS END\n)", lambda m: m.group(1) + "\n".join(out) + "\n" + m.group(2), s, flags=re.S)
open(P, "w").write(s)
print(len(out), "kernels")
