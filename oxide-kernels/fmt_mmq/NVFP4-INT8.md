# NVFP4 int8 MMQ: open defect

Status: **the trap is fixed; 217 bytes out of 169,161,784 still differ** (opt-in slot `--fmt=nvfp4q8`).

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

## What still differs (217 bytes)

All remaining differences are in cases with `rate=256`, i.e. where the harness injects inf / nan /
denormal / 3e38 activations, and every one is a low-byte ULP difference -- the arithmetic is right, the
last bit is not. Ruled out with measurements:

| tried | result |
|---|---|
| exact `div.rn.ftz` in the three scale paths (`amax/2688`, `amax_sub/6`, `1/row_scale`) | **worse**: 261 failing -> the reference is built with fast-math, so `div_approx` is the correct match |
| exact `x * 0.5` instead of `div_approx(x, 2.0)` in `ue4m3` | neutral (217) |
| shared y quantization kernel (`quantize_mmq_d4`) | innocent -- slot `mxfp4q8` passes bit-identically through it |
| FTZ vs non-FTZ multiply | both reference kernels are `.FTZ`, same as ours |
| the m16n8k16 MMA | legal for `sm_120`; ptxas emits `IMMA.16816.S8.S8` and the reference cubin runs it |

Remaining candidate: the `ue4m3` candidate search's error metric (`nv_err`) and tie-breaking, which can
flip a near-tie scale by one step on 16-value sub-blocks.

## Nothing else depends on this

The default run (`iq4_nl`, `mxfp4`, `nvfp4`, `iq4_xs`, `mxfp4q8`) is bit-identical:
`1015 launches, 562,321,320 bytes, 0 failing`. Slot 5 (`nvfp4q8`) is opt-in and cannot affect it.
