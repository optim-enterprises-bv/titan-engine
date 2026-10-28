# NVFP4 int8 MMQ: open defect

Status: **the tile loader is correct, the dot traps.** This note records what has been ruled out so the
next attempt does not re-walk it.

## Symptom

`./target/release/fmt_mmq --fmt=nvfp4q8` (slot 5, opt-in) fails on the first launch with **CUDA 719**
(`CUDA_ERROR_LAUNCH_FAILED`); `compute-sanitizer` reports `Trace/breakpoint trap` inside
`nvfp4q8_mmq_j*_f*`. The reference side of the same case runs fine (`FMT_MMQ_SKIP_OX=1`: 444 launches,
no trap).

## Ruled out, with the evidence

| hypothesis | how it was excluded |
|---|---|
| wrong reference revision / ABI | reference rebuilt from the pinned `acecd560` with `ref/nvcc_mmq_*.sh` flags plus `-DGGML_MMQ_FORCE_Q8`; its mangling matches every other slot, and the MXFP4 slot built the same way is **bit-identical** |
| tile loader wrong | the same loader driven by the plain `vec_dot_q8_0::<J, 84>` **runs** (no trap, value mismatches only), so it produces sane tiles and does not fault |
| `mma.m16n8k16.s8` not supported on cc 12.0 | `ptxas -arch=sm_120` compiles it (cubin produced), and the reference cubin uses `IMMA.16816.S8.S8` and executes on this GPU |
| generated code differs from the reference | our PTX compiled with `ptxas` gives the identical instruction mix: `32 IMMA.16816.S8.S8`, `16 LDSM.16.M88.4` |
| shared memory undersized | sized from llama.cpp's own formula `J*4 + I*sram_stride*4 + PAD(J*144, nthreads*4)` = 45,088 B for J=8; the harness requests 49,184 B, and +8 KB / +16 KB also trap |
| the k16 instruction itself | a second dot was written using **only** proven pieces (`mma_s8` m16n8k32 twice with half of A zeroed, so the 16-value `ue4m3` scales apply exactly) — it still traps |

## What that leaves

The fault is inside this slot's kernel at run time, and it is not explained by the loader, the config,
the reference, the instruction set, the generated code or the shared-memory size. The next step is
instrumentation rather than reasoning: dump the k16 fragments (A registers, B register, dB, dA) from
inside the kernel for one case and compare against the reference kernel's values for the same case, or
build the dp4a variant (`ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_dp4a`, which uses the dp4a tile layout and
avoids the MMA fragment path entirely).

## Nothing else depends on this

The default run (`iq4_nl`, `mxfp4`, `nvfp4`, `iq4_xs`, `mxfp4q8`) is bit-identical:
`1015 launches, 562,321,320 bytes, 0 failing`. Slot 5 is opt-in and cannot affect it.
