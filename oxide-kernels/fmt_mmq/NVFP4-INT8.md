# NVFP4 int8 MMQ: open defect

Status: **the trap is fixed; the values are still wrong** (opt-in slot `--fmt=nvfp4q8`).

## Fixed: CUDA 719 trace/breakpoint trap

The dot stepped `j0 += 8` where the reference steps `j0 += Jc::<J>::JSTEP` (16 whenever `NTX == 2`, i.e.
`J >= 48`). That made `idx = (j0/8 + n)*4 + l` reach **67 on a `sum[64]`**, so the Rust bounds check fired
and the GPU codegen lowered the panic to `trap;` -- CUDA 719. With the step corrected the kernel runs
end-to-end: `469 launches, 169,161,784 bytes compared`.

This is why the earlier evidence looked contradictory: the generated SASS was instruction-identical to the
reference (the trap was never in the MMA), the "proven" mma still appeared to trap (the bad step was still
there), and shared-memory changes did nothing (never a memory-size problem). The earlier "the k16
instruction traps" and "the scale index is causal" conclusions were both wrong for the same reason.

## Fixed: rounding order

The reference nests `k01` (one 16-value step) outermost, then `j0`, then `n`, with a single
`sum += C*dA*dB` per 16-value group. Accumulating both halves of a 32-value chunk together is
arithmetically equal but rounds differently.

## Fixed: y-scale stride

`MMQ_TILE_Y_K = MMQ_TILE_NE_K + MMQ_TILE_NE_K/QI8_1 = 32 + 4 = 36`, so `QI8_1 = 8`: the q8_1 activation
scale steps every **8 ints** (32 values), not every 32 ints. The dot used `k01 / 32`, which collapsed all
four scales of a chunk onto the first one.

## What still differs (real, not noise)

Two regimes, both measured with `FMT_MMQ_RATE` (which forces a clean, unpoisoned case set):

- **Clean data**: the two sides agree closely but not exactly. Relative error distribution over 10,378
  differing words of one dumped case: **p50 1.4e-7 (about 1 ULP), p90 6.2e-7, p99 6.4e-6, max 4.8e-4**.
- **Poisoned data** (`rate=256`): the sums overflow at different points, one side reaches inf while the
  other stays finite, so they cannot be compared at all. This produced the "34,673,786 out-of-tolerance"
  figure, and an earlier note here wrongly read the 217 *label* count as a byte count.

Across all cases, 8,867 words exceed a 5e-4 relative tolerance: this is a **real numerical defect with a
long tail**, not a rounding artefact. A tolerance gate was tried and reverted -- shipping one would have
hidden it.

## Ruled out, with measurements

| tried | result |
|---|---|
| exact `div.rn.ftz` in the three scale paths (`amax/2688`, `amax_sub/6`, `1/row_scale`) | **worse** -> the reference is built fast-math, so `div_approx` is the correct match |
| exact `x * 0.5` instead of `div_approx(x, 2.0)` in `ue4m3` | neutral |
| shared y quantization kernel (`quantize_mmq_d4`) | innocent -- slot `mxfp4q8` passes bit-identically through it |
| FTZ vs non-FTZ multiply | both reference kernels are `.FTZ`, same as ours |
| the m16n8k16 MMA | legal for `sm_120`; ptxas emits `IMMA.16816.S8.S8` and the reference cubin runs it |
| reference error metric / candidate search (`nv_err` vs `nvfp4_native_scale_error`) | line-for-line identical: `2.0f*scale`, `|v| - |dq|*2s`, offsets `{0,-1,+1,-2,+2}`, `0.5f/s`, strict `<` |
| `ue4m3` decode | identical to `ggml_cuda_ue4m3_to_fp32` (NaN codes 0x7F/0xFF -> 0, e4m3->f16->f32, halve) |
| driver JIT vs a real `sm_120a` cubin (`FMT_MMQ_MODULE=<cubin>`) | same failure, so the loading path is not the variable |
| k16 fragment lane mapping | A loaded as an x4 pair `(lane%16)*stride + (lane/16)*4`, B as `(lane/4)*stride + lane%4`; both match `tile<16,4>` / `tile<8,4,int>` for `mma.m16n8k16` |

## Next

Dump the tile contents themselves -- A fragments and the `ue4m3` `dA` per 16-value sub-block -- for one
clean case, and diff them against the reference. The residual is most likely a sub-block scale decision:
the median error is 1 ULP but the tail is three orders of magnitude larger.

## Nothing else depends on this

The default run (all seven slots, including block-scaled `mxfp4`/`nvfp4`) is bit-identical:
`1753 launches, 910,060,092 bytes compared, 0 failing`. The `nvfp4q8` slot below is opt-in and cannot
affect it.

## Gate knobs added for this work

- `FMT_MMQ_MODULE=<path>` -- load a cubin (or any module) instead of `fmt_mmq.ptx`, so the driver-JIT path
  can be compared against a real `sm_120a` binary.
- `FMT_MMQ_RATE=<n>` -- override the specials density for every case; a huge value means no inf/nan/denormal
  poisoning, which is the only way to see the arithmetic on its own.
