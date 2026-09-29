# Upstream mistral.rs v0.9.4 (2370966) on titan: Qwen3.6-35B-A3B, as-is

Date: 2026-09-28. Worktree: `~/titan-engine/mr-upstream` (detached at upstream/master 2370966bb).
Binary: `target-upstream/release/mistralrs`, built with `--features cuda` (default feature set, no cutile),
CUDA_COMPUTE_CAP=120, NVCC_CCBIN=g++-15, CUDA_NVCC_FLAGS=-Wno-template-body, 2 jobs and 1 nvcc thread. It built cleanly the first time,
**28m43s** from a seeded target dir. No source changes were needed.
Model: `~/ai/models/Qwen3.6-35B-A3B-MTP/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (22 GB).
Two GPU windows were used: `window-up1.sh` (build plus first benches) and `window-up2.sh` (workaround ladder).
The logs are `window-up*-*.log` and `out/*.server.log`. The harness is `collect-up.sh` plus `ubench.py`: the m3/collect.sh prompts and sampling, with configurable server flags,
4k/13k prefill, and prefix-cache repeats.

## Result: upstream cannot serve this model on a 16 GB card

Every configuration loads, but **every request fails**. As a result there are no decode, prefill, prefix-cache or greedy-identity numbers to report.

| config (all MTP off) | load / split | first request | tok/s | identical vs q35-prof |
|---|---|---|---|---|
| auto map, `--max-seq-len 16384` (default bf16, paged auto) | **refused**: "does not fit on cuda[0] 15661 MB + cpu 20349 MB, exceeds total capacity by 798 MB" | – | – | – |
| same with `--mtp --mtp-n-predict 2` | refused (same device-map error, so MTP was never reached) | – | – | – |
| same with `MISTRALRS_CUDA_GRAPHS=0` | refused (same) | – | – | – |
| m3/collect.sh flags as-is (`--paged-attn off --max-seq-len 4096`, bf16) | 25 s, layers 0-15 GPU / 16-39 CPU | **HTTP 500**: `moe experts forward: dtype mismatch in matmul, lhs: BF16, rhs: F32` | – | 0 |
| `--dtype f32`, 4k map, paged auto | 20 s, layers 0-19 GPU / 20-39 CPU, RSS 17.6 GB; "mix of GPU and CPU … disabling PagedAttention" | **500**: `causal_conv1d_cuda only supports f16/bf16, got F32` | – | 0 |
| `--dtype f32 --paged-attn off` | 20 s, same split | **500**: same causal_conv1d error | – | 0 |
| `--dtype f16 --paged-attn off` | 22 s, RSS 18.7 GB | **500**: `dtype mismatch in matmul, lhs: F16, rhs: F32` | – | 0 |
| **ours**: tiered + MTP=2 | – | – | ~100 decode, ~230 prefill @13k | 40/40 gate |
| ours: tiered, MTP off | – | – | 82–84 | 40/40 |
| llama.cpp `--n-cpu-moe 18` | – | – | ~61 | – |

The explicit-split (`-n 0:24`), graphs-off, MTP and paged-off prefix-cache runs in window 2 were never reached, because the ladder stopped once every dtype failed.

## Why (code read of 2370966, consistent with the errors above)

1. **CPU MoE for GGUF on x86 is broken and would be unusably slow even if fixed.** `mistralrs-quant/src/gguf/cpu.rs::qtensor_indexed_moe_forward`
   first tries candle's `QTensor::indexed_gemv`, which is implemented only for `target_arch = "aarch64"`. On x86 it falls back to
   `qtensor.dequantize(device)`, which dequantizes the **whole expert tensor (all 256 experts) to F32** on every call. It then does
   a gather matmul against BF16/F16 activations, which is where the dtype mismatch comes from. Casting the activations would make it run, but it would
   still write about 1 GB of f32 per projection per CPU layer per token.
2. **f32 activations are rejected by the GDN CUDA conv** (`causal_conv1d_cuda only supports f16/bf16`). So the one dtype that
   would satisfy the CPU matmul breaks the GPU layers.
3. **Offload granularity is whole layers only.** Topology/`-n` place a layer (attention, GDN, router and all experts) on one device.
   There is no expert-level offload (no `--n-cpu-moe` equivalent), no hot/cold expert tiering, and no expert cache. Topology
   regex *device* overrides only apply on the immediate-ISQ path, not to pre-quantized GGUF experts.
4. **A GPU+CPU mix disables PagedAttention** ("no CPU support for PagedAttention"). Upstream CUDA decode graphs require
   PagedAttention metadata (`try_cuda_decode_graph_forward` → `PagedAttentionUnavailable`), so **any offloaded model runs
   without CUDA graphs**.
5. **A text-only `qwen35moe` GGUF resolves to `models/qwen3_next.rs`** (log: "Qwen3Next: 10 full attention layers, 30 linear
   attention (GDN) layers"). That model has **no MTP code** (0 matches for "mtp") and does not override `supports_cuda_decode_graphs`
   (the default is `false`). Upstream's Qwen3.5/3.6 MTP (c75e5a4, 552ddf7) and MTP CUDA graphs (98ca082) live in
   `vision_models/qwen3_5{,_moe}`, which GGUF only reaches through the multimodal (mmproj) binding. The GGUF path also subtracts the
   `nextn_predict_layers` from the block count and never binds `blk.40.nextn.*` to `mtp.*`. **So from this GGUF, upstream gets
   neither its MTP nor its CUDA graphs, even on a card big enough to hold the model.**
6. The auto device map over-estimates. It needed more than 36 GB at 16k context for a 22 GB file, and put only 16-20 of 40 layers on the GPU at 4k.

## Feature overlap: theirs vs ours

| feature | upstream (2370966) | ours (ktrace) | verdict on titan |
|---|---|---|---|
| MTP | Built-in-head MTP for safetensors Qwen3.5/3.6 via the multimodal model; quantized drafter, device verify, stash-based GDN rollback, DFlash2 drafting, CUDA-graphed. **Not reachable from a text GGUF** (qwen3_next has none). | GGUF nextn head in `quantized_qwen35_moe.rs`, MTP=2 about 100 tok/s, 40/40 byte-identical gate | **ours** (theirs cannot run here; design ideas such as stash rollback and graph capture of verify are worth reading) |
| CUDA graphs | Whole-step decode graphs with batch buckets, needs PagedAttention and all layers on the GPU; qwen3_next opts out | Piecewise graphs for batch-1 decode + MTP verify that work with tiered/CPU experts, +2.6-2.9% | **ours** for titan; theirs is better for multi-sequence serving on a card that fits the model |
| GDN kernels | In-place pooled recurrent state kernels, fused MLP, f32 gate params (7d8fa17); bf16/f16 only | Ours plus cuda-oxide kernels, verified bit-identical | **take theirs as the base** in the shared `gdn/` and `cuda/gdn.rs` (it is the upstream direction, and the conflicts are in these files) and re-apply our MTP-verify row changes on top. Speed not measurable here. |
| Prefix cache | Paged block-hash prefix cache with **recurrent (GDN) state checkpoints** plus usage `cached_tokens` (d71ab1f); sequence-level cache when unpaged | Sequence-level, with our fix forbidding partial reuse on hybrid models (46053a3) | **theirs** conceptually (proper hybrid checkpoints), but it only works under PagedAttention, which offload disables. Keep our fix for the unpaged path and adopt their `cached_tokens` usage field. |
| Expert offload / tiering | None (whole-layer only; x86 CPU MoE broken as above) | Tiered experts, AVX2 CPU twin, P1 prefetch, profile placement | **ours**, no upstream equivalent |
| Chunked prefill | Scheduler `--max-prefill-chunk-tokens` for recurrent batching (paged path) | Unpaged chunked prefill with KV reserve | **ours** for titan (unpaged); revisit if we ever run paged |
| GGUF loaders | GGUF for all models via bindings into native models (1fdb451); **deleted `quantized_qwen3_moe.rs` and every `quantized_*` model except llama**; GGUF `Model` enum keeps only the X-LoRA variants | `quantized_qwen35_moe.rs` (2.9k lines), `quantized_gpt_oss.rs`, qwen3next via the qwen35moe loader, all on the old quantized path | conflict: see below |

## What upstream cannot do on titan

- Run Qwen3.6-35B-A3B (or any GGUF MoE larger than 16 GB) at all. The CPU-layer MoE path fails on x86, and a dtype that fixes it breaks GDN.
- Offload experts instead of layers, tier hot experts, or overlap CPU misses with GPU work.
- Use CUDA graphs or PagedAttention (and so its hybrid prefix checkpoints) whenever any layer is on the CPU.
- Use its MTP from a GGUF with a nextn head.

## Recommendation for the sync

The 93 upstream commits are dominated by 1fdb451, which touches 248 files (+43k/-7k lines) and moves GGUF into native models. 51 files are
changed on both sides. Rebasing our 58 commits naively would re-add the deleted `quantized_*` dispatch in `pipeline/gguf.rs`.

1. **Keep ours**: MTP, piecewise CUDA graphs, tiered experts + CPU twin + P1 prefetch, chunked prefill, and the
   `quantized_qwen35_moe` / `quantized_gpt_oss` models. Upstream has no working substitute on titan.
2. **Take theirs**: the core infrastructure we don't differentiate on: scheduler/concurrency (8010b6a, b0882bd), usage
   `cached_tokens` (d71ab1f), GDN pooled-state kernels (7d8fa17) as the base for `gdn/`, hybrid auto-device-map fixes (d1ffbe7),
   and the kernel-object cache (d348a88). Re-apply our deltas on top.
3. **Structure the rebase around 1fdb451**: keep our GGUF quantized-model path as a fork-local loader (our own `Model` variants in
   `pipeline/gguf.rs`, selected for `qwen35moe`/`qwen3next`/`gpt-oss` when `TITAN_*` tiering is in play or always), rather than
   porting tiering into upstream's `qwen3_next.rs` + `moe/experts` now. Porting is the long-term path, since it would inherit their paged/graph work,
   but it means reimplementing MTP, piecewise graphs and tiering against a model with no MTP and graphs that exclude offload.
4. Cheap upstreamable fixes we found: (a) x86 `indexed_gemv` / the per-expert dequant fallback in `gguf/cpu.rs`, plus casting activations to
   the weight dtype; (b) bind `blk.N.nextn.*` to `mtp.*` for GGUF; (c) auto-device-map size estimate for GGUF MoE.
5. For a like-for-like measurement of upstream's MTP/graphs/GDN, the next step is a model that fits entirely in 16 GB and loads through
   the qwen3_5 multimodal path (safetensors or GGUF + mmproj), compared against our fork on the same model. That needs a third
   window, which this task did not have.
