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

## Where it stands

| format | result |
|---|---|
| `--fmt=mxfp4` with `FMT_MMQ_BLOCKSCALED=1` | 253 launches, 49,459,948 bytes compared, **1 failing** |
| `--fmt=nvfp4` with `FMT_MMQ_BLOCKSCALED=1` | 447 launches, 50,752,256 bytes compared, **1 failing** |

Both remaining failures are the same shape: `fb=true`, `J=128`, irregular `nrows`/`ncols`
(`nrows=186 k=192 ncols=145`, `nrows=298 k=64 ncols=274`), and the differing bytes are **zeros on our
side** (3 and 91 bytes respectively out of ~108 KB and ~328 KB). So the MMA and the loader are right; some
output elements are never written for those tilings -- a stream-k / fixup coverage gap in the fallback
shape, the same class as the `titan-doorbell` missed rows already documented in `GATES.md`.

Next: instrument those two cases for which `(i, j)` the kernel never writes (`FMT_MMQ_TMP` already
compares the stream-k partials before the fixup), then check the `bpn`/`ncy`/`sr` packed counters for
`ncols` that are not multiples of the tile width.
