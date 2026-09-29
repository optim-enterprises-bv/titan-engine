#!/usr/bin/env python3
"""Regenerate the explicit #[kernel] wrappers in src/main.rs (between the GENERATED markers).

cuda-oxide only exports kernels written out with #[kernel] (macro_rules!-generated ones are not
seen by the module scan), so every (type, mmq_x, need_check) instance and every fixup instance
gets a one-line wrapper around the const-generic helpers.
"""
import re, sys

TYPES = sys.argv[1].split(",") if len(sys.argv) > 1 else [
    "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k"]
# mmq_x values reachable on sm_120 (turing mma: granularity 16 from 48 on).
MMQ_X = [8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128]
QKS = sorted({256 if t.endswith("_k") else 32 for t in TYPES})

lines = []
for t in TYPES:
    const = t.upper()
    for x in MMQ_X:
        for nc in (0, 1):
            lines.append(
                f"    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_{t}_x{x}_nc{nc}(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, "
                f"ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) "
                f"{{ unsafe {{ mul_mat_q::<{const}, {x}, {'true' if nc else 'false'}>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) }} }}")
for qk in QKS:
    for x in MMQ_X:
        for nc in (0, 1):
            lines.append(
                f"    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk{qk}_x{x}_nc{nc}(dst: *mut f32, tmp_last_tile: *const f32, "
                f"ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) "
                f"{{ unsafe {{ stream_k_fixup::<{qk}, {x}, {'true' if nc else 'false'}>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) }} }}")

path = "src/main.rs"
src = open(path).read()
begin = "    // GENERATED KERNELS BEGIN (gen_kernels.py)\n"
end = "    // GENERATED KERNELS END\n"
a = src.index(begin) + len(begin)
b = src.index(end)
open(path, "w").write(src[:a] + "\n".join(lines) + "\n" + src[b:])
print(f"{len(lines)} kernels for {TYPES}")
