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

## What still differs (real, not noise)

The outputs differ **materially**, not at ULP level: a `FMT_MMQ_DUMP`-instrumented case gives `ref
-2.317e3` against `oxide -1.358e3`, and the ULP-classified comparison counts **34,673,786 words
out-of-tolerance, 0 within 1 ULP** (1,861,705 both-non-finite). An earlier note in this file claimed
"217 bytes, <=1 ULP"; that was wrong -- the 217 was a *label* count, and the surviving differences are
real arithmetic errors in the NVFP4 dot/loader for some shapes.

Ruled out with measurements:

| tried | result |
|---|---|
| exact `div.rn.ftz` in the three scale paths (`amax/2688`, `amax_sub/6`, `1/row_scale`) | **worse**: 261 labels -> the reference is built fast-math, so `div_approx` is the correct match |
| exact `x * 0.5` instead of `div_approx(x, 2.0)` in `ue4m3` | neutral |
| shared y quantization kernel (`quantize_mmq_d4`) | innocent -- slot `mxfp4q8` passes bit-identically through it |
| FTZ vs non-FTZ multiply | both reference kernels are `.FTZ`, same as ours |
| the m16n8k16 MMA | legal for `sm_120`; ptxas emits `IMMA.16816.S8.S8` and the reference cubin runs it |
| reference error metric / candidate search (`nv_err` vs `nvfp4_native_scale_error`) | line-for-line identical: `2.0f*scale`, `|v| - |dq|*2s`, offsets `{0,-1,+1,-2,+2}`, `0.5f/s`, strict `<` |
| `ue4m3` decode | identical to `ggml_cuda_ue4m3_to_fp32` (NaN codes 0x7F/0xFF -> 0, e4m3->f16->f32, halve) |
| driver JIT vs a real `sm_120a` cubin (`FMT_MMQ_MODULE=<cubin>`) | identical failure, so the loader path is not the variable |

Next: the k16 fragment lane mapping -- our `load_b16`/`ldmatrix_a` pair against the reference's
`tile<16,4>` loads for `mma.m16n8k16`, one case at a time.

## Nothing else depends on this

The default run (`iq4_nl`, `mxfp4`, `nvfp4`, `iq4_xs`, `mxfp4q8`) is bit-identical:
`1015 launches, 562,321,320 bytes, 0 failing`. Slot 5 (`nvfp4q8`) is opt-in and cannot affect it.
