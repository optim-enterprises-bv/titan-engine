# v0.9.4 nvcc reference for the paged-attention ports

The gates of `mistralrs-paged-attn-a` and `-b` link the real C launchers from
`../../reference/mistralrs-paged-attn-094/libmistralrspagedattention.a`, and the kernel-level gate loads `flashinfer_decode.cubin`
from the same place. Both are untracked artefacts in `~/titan-engine/oxide-kernels/reference/` (symlinked into a worktree).

## Where the reference comes from
- **Objects.** The 15 objects are the nvcc build of mistral.rs's `mistralrs-paged-attn/src/cuda/*.cu`. It was made by cudaforge
  0.1.6 through `mistralrs-paged-attn/build.rs` into `~/titan-engine/cuda-build-094/paged-attn/kernels` (the titan-094 nvcc build).
  The flags are `-std=c++17 -O3 -U__CUDA_NO_HALF*… --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -DENABLE_FP8`,
  plus `-gencode=arch=compute_120a,code=sm_120a --default-stream per-thread`.
- **Archive and cubins.** The objects are re-archived as `libmistralrspagedattention.a`. Each cubin is extracted with
  `tools/cuobjdump -xelf`.
- **Sources match.** cudaforge's `content_hash` of every `.cu` (kept as `cudaforge_cache.json`) equals the sha256 of the same file
  at model-swap 2c4b140e3, the base of the `pattn` branch.

## Rebuilding with nvcc (`nvcc_094.sh`, `/usr/local/cuda/bin/nvcc` 13.3, `-ccbin g++-15`)
- **copy_blocks_kernel.cu.** `nvcc_094.sh <mistralrs-paged-attn> out copy_blocks_kernel` gives `copy_blocks_kernel.cubin`
  **byte-identical** to the cudaforge object's. So the script's flags are cudaforge's flags.
- **flashinfer_decode.cu, full translation unit.** cicc segfaults under the 4 GB memory cap that CPU work outside a GPU window
  must use.
- **flashinfer_decode.cu, reduced unit.** `reduced/nvcc_reduced.sh` compiles a reduced copy instead. It keeps head dim 128,
  (bf16, bf16) and (f16, fp8) decode, and all six reshape / gather pairs. **All 164 of its kernels are SASS-identical** to the
  same entries of the cudaforge `flashinfer_decode` object. Its PTX (`reduced/fi_reduced.ptx`) is where these were read:
  - the FP8 conversions: write `div.approx.ftz.f32` then `cvt.rn.satfinite.e4m3x2.f32 d, 0, x`; read `cvt.rn.f16x2.e4m3x2`,
    `cvt.f32.f16`, `mul.ftz.f32 scale, x`, `cvt.rn.{f16,bf16}.f32`;
  - the v0.9.4 `v_scale` placement: `(o * d_rcp) * v_scale`, two `mul.ftz`, `BatchDecodeParams` offset 156.

## What changed in v0.9.4 (vs the 84b53bf reference in `reference/mistralrs-paged-attn`)
| file | change | oxide port |
|---|---|---|
| flashinfer_decode.cu | `reshape_and_cache_flashinfer<DType, CacheType>` (+k_scale, v_scale, cache_dtype; FP8 E4M3 write) | crate b, all 6 pairs |
| flashinfer_decode.cu | `gather_kv_cache_flashinfer<CacheType, OutType>` (+num_seqs bound, zero-fill past `cu_seq_lens[num_seqs]`, FP8 read) | crate b, all 6 pairs |
| flashinfer_decode.cu / default_decode_params.cuh / utils.cuh | `flashinfer_decode` +k_scale (folded into sm_scale on the host), +v_scale (OutputTransform), +cache_dtype; GQA groups 5, 6, 7 | crate b: same-dtype caches, all groups; **FP8 E4M3 cache decode (864 instances) not ported: the launcher returns cudaErrorNotSupported** |
| copy_blocks_kernel.cu | `copy_blocks_u8` | crate a |
| gather_kv_cache_kernel.cu | `if (batch_id >= num_seqs) return;` | crate a |
| pagedattention.cuh, cuda_compat.h | host-side attribute check only; SASS of every other object unchanged | none needed |
