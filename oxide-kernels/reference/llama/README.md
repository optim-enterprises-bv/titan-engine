# llama.cpp mmvq reference

`mmvq-sm120.cubin` is the reference the `q1_0` / `iq4_nl` / `mxfp4` / `nvfp4` gates compare
against: llama.cpp's `mul_mat_vec_q<type, ncols_dst=1, ...>`, the MUL_MAT_ID decode kernel.

The file itself is gitignored (`*.cubin` is a build artifact class in this tree), so it has to be
mined once per checkout:

```sh
cd titan-kernels-oxide
uv run --with zstandard python scripts/mine-mmvq-ref.py \
    /home/dingo/models/bonsai2-27b/bonsai2-small-gpu-linux-x64-cuda12.8-sm120-ff41412/lib/libggml-cuda.so.0
# -> ../oxide-kernels/reference/llama/mmvq-sm120.cubin
```

Any CUDA build carrying this kernel for those four formats works — cubins are OS-independent, so a
Windows release zip is fine. Without the file the four sections skip and say so; `GATE_DEBUG=1`
prints which path was chosen.

## Why not the NOTICE.md pin

The pin is `ggml-org/llama.cpp acecd56032ddc` (tag b10934, 2026-09-12), and it is the right
provenance for this kernel. Its own sm_120 image, however, does not run from an external harness:
it loads, resolves, launches, and writes nothing (all-zero output), and requesting the PDL launch
attribute it needs is rejected with `CUDA_ERROR_INVALID_VALUE`. That is upstream's Blackwell bug.

The vendored cubin is the PrismML fork's build of the same kernel, which does run and is
bit-identical to the oxide implementation across all four formats. Their tree carries a newer
template — `has_gate` between `has_fusion` and `small_k`, so symbols have six template parameters
instead of five. The gate accepts both mangled names; the runtime parameter list (160 bytes:
3 pointers, the 48-byte fusion struct, `dst`, then twenty `u32`s) is identical either way.
