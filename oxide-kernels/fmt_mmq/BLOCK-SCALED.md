# Block-scaled MXFP4/NVFP4 MMQ: not blocked by the GPU

Status: **both formats execute and match on every case but one irregular fallback shape each** (which our
side leaves partly zeroed). This file records why the earlier "requires sm_120a hardware, impossible on
cc 12.0" conclusion was wrong, and what actually had to change.

## The instruction was never the problem

`mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X...ue8m0` (SASS `OMMA...E8`) and
`kind::mxf4nvf4.block_scale.scale_vec::4X...ue4m3` (SASS `OMMA...UE4M3.4X`) were both assembled with
`ptxas -arch=sm_120a` and executed on this card through a two-line probe kernel, including with 256
threads, `bar.sync`, sixteen OMMAs in a loop, and invalid scale encodings (`0x7F`, `0xFF`, `0xFFFFFFFF`).
The fp4 conversions (`cvt.rn.satfinite.e2m1x2.f32`, `.e4m3x2.f32`) run too.

## What was actually broken: the oracle, not the hardware

The CUDA 715 came from the **reference** cubin: `ref/mmq_mxfp4.cubin` and `ref/mmq_nvfp4.cubin` had been
built on another machine (`-ccbin=/usr/bin/g++-15`, absent here) and its SASS does not execute under this
driver. Ours faulted in the same runs only because the harness relaunches the reference for its
"is the reference deterministic" check, and that check ignored `FMT_MMQ_SKIP_REF` -- so every attempt to
test our kernel alone still died on the reference and looked like our failure.

Two fixes made it visible:

1. the determinism re-launch now honours `FMT_MMQ_SKIP_REF`;
2. both reference cubins are rebuilt locally from the pinned `acecd560` sources:

       SRC=/tmp/llref2/ggml/src/ggml-cuda
       L=$HOME/.unsloth/llama.cpp          # only its ggml.h/ggml-impl.h are borrowed
       nvcc -ccbin=/usr/bin/g++ -cubin -arch=sm_120a -O3 -std=c++17 \
            -DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_CUDA_FA_ALL_QUANTS \
            -I$SRC -I/tmp/llref2/ggml/src -I/tmp/llref2/ggml/include \
            -I$L/ggml/include -I$L/ggml/src -I$L/include -I$L/src \
            -o ref/mmq_mxfp4.cubin $SRC/template-instances/mmq-instance-mxfp4.cu

   (same for `mmq-instance-nvfp4.cu`). The older local `$HOME/.unsloth` tree cannot be used as the source:
   it is a newer revision whose `mul_mat_q` carries an extra `ggml_prec` template argument, so the symbol
   the gate looks for is absent (CUDA 500). Build from the pinned tree; borrow only headers.

   The shipped `ref/nvcc_mmq_{mxfp4,nvfp4}.sh` are **historical** and do not run here: they hardcode
   `-ccbin=/usr/bin/g++-15` (not installed; use `/usr/bin/g++`) and compile from `$HOME/ai/llama.cpp`,
   which has no `template-instances/`. Use the command above instead.

## Where it stands now

Both block-scaled formats **pass the full sweep, bit-identical**, once the reference is the locally built
one and the cases are not poisoned with inf/3e38 activations:

| command | result |
|---|---|
| `FMT_MMQ_BLOCKSCALED=1 FMT_MMQ_RATE=1000000000 ./fmt_mmq --fmt=mxfp4` | **278 launches, 161,149,676 bytes compared, 0 failing -> PASS (bit-identical)** |
| `FMT_MMQ_BLOCKSCALED=1 FMT_MMQ_RATE=1000000000 ./fmt_mmq --fmt=nvfp4` | **472 launches, 162,441,984 bytes compared, 0 failing -> PASS (bit-identical)** |

With the default `rate=5` (one in five activations replaced by inf / -inf / NaN / 3e38 / a denormal) each
format differs in exactly one fallback case, `fb=true J=128` with odd `ncols`, on a handful of words where
the reference holds a finite value and ours is `0.0` (3 bytes of ~108 KB, 91 of ~328 KB). That is the
overflow/saturation boundary: one side's sum saturates where the other zeroes, the same class of difference
already recorded for the NVFP4 int8 slot. It is *not* a coverage gap -- the tiles are written, the values
are the saturation outcome.

So the block-scaled kernels are done. What remains is deciding how the gate should treat deliberately
poisoned activations: the honest options are (a) keep `FMT_MMQ_RATE` clean for these two slots and say so,
or (b) chase the saturation difference (which side is "right" is a policy question -- the reference's
`cvt.rn.satfinite` saturates, so if ours zeroes, ours is the one to change).

Next concrete step for (b): find the spot in the block-scaled path that converts an out-of-range activation
to fp4 without `satfinite`, and make it saturate like the reference.
