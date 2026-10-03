# orca: iq4xs merged onto the deployed tree, and OrcaSAQ-2 27B made to fit (staged, NOT deployed)

Binary **`bin/mistralrs-titan-orca`**, sha256 `6cdca99df15879fbd54786ab2406c178068672d246e8294dd45e3187fbe380a6`
(`bin/mistralrs-titan-orca.sha256`; nvcc-free `cargo build --release -p mistralrs-cli --features oxide`, window 1, CARGO_TARGET_DIR
target-orca-oxide seeded from target-devmap-oxide). Roster to deploy: **`m4/orca/models-deploy.toml`** = deploy/models.toml (caec2a5a)
+ an `orcasaq2-cyber-27b` entry. Not deployed; deploy/models.toml and bin/mistralrs-titan-swap untouched; nothing pushed.

## Branch heads (`orca` in all three repos)
| repo | worktree | head |
|---|---|---|
| mistral.rs | top-orca/mr-orca | 90707cc05 (merge 4339ffe26 of iq4xs 1f8bf6a3b into model-swap 6eaaf58f9, then the qwen35 fix) |
| titan-engine (candle + m4) | top-orca (sparse) | merge of iq4xs (4b520cf7, 98d4535f, e07157f8) into master caec2a5a + this directory |
| oxide-kernels | oxide-orca (top-orca/oxide-kernels -> ../oxide-orca) | da6602b (merge 2a3d7ba of iq4xs 91452a1 into glu-typed-kernels 85708b2, then the d128 PTX) |

## Step 1: merge
All three merges were clean: iq4xs touched fmt_mmq / iq4_xs / candle llama_fmt + k_quants / fast_mmq; devmap touched flash-prefill,
qwen3, kv_cache, candle CPU bf16 matmul. No file overlaps, so nothing had to be unioned by hand.
**Found:** the committed oxide tree did not match the embedded flash-prefill PTX. 85708b2 (devmap) added the head-dim-128
kernels to the sources and the gate, but never committed the regenerated `flash_prefill.ptx` (18 entries, no d128). mistral.rs
6eaaf58f9 embeds the 22-entry file, which existed only as an uncommitted change in oxide-devmap. **da6602b commits it.**
After CR normalisation it equals `mistralrs-core/src/attention/flash_prefill_oxide.ptx` byte for byte.

PTX regenerated with the crate build scripts (`cargo oxide build` in **oxide-orca-gate**, a reflink copy, so the concurrent
mistral.rs build only ever saw committed files), then compared with the committed and embedded copies:
| crate | regenerated vs committed / embedded |
|---|---|
| flash-prefill | 22/22 entries == the embed (ptxcmp; ptxnorm.py); 18/18 == 85708b2's file, 4 new (d128) |
| iq4_xs, iq3_s | 16/16 each (byte order differs between builds; entry code identical) |
| fmt_mmq | 141/141 (ptxcmp). export_ptx.py from the committed fmt_mmq.ptx gives iq4_xs_mmq_oxide.ptx **byte-identical** to the embed; iq4_nl / mxfp4 / nvfp4 35/35 each (the embeds predate a relabelling) |
| mistralrs-quant-a | 282/282 (ptxnorm: label numbering, trailing spaces and rustc crate hashes are the only differences) |
| mistralrs-quant-c | byte-identical |
| ptq1_0 / q1_0 exports | byte-identical to the 4 embedded mmq / mmvq modules |

## Kernel gates (window 1, regenerated PTX; logs out/gate-*.log)
| gate | result |
|---|---|
| flash-prefill (FP_BASE_PTX = 85708b2's committed file) | **PASS**: d128 bf16 / f16o32, d512 bf16 / f16 / f16o32, d256 f16, r2 bf16 / f16o32 accuracy and splits all within tolerance (worst 0.692 of tol), every mutation caught; existing kernels vs base: **0 of 53,100,544 outputs differ** |
| iq4_xs | **PASS**: 1157 launches, 15,612,608 B, 0 failing; 4/4 mutants detected |
| iq3_s | **PASS**: 1085 launches, 0 failing; 4/4 mutants |
| fmt_mmq IQ4_XS MMQ | **PASS**: 230 launches, 208,754,192 B, 0 failing; 2/2 loader mutants |
| fmt_mmq IQ4_NL MMQ | **PASS**: 330 launches, 198,296,952 B, 0 failing |
| mistralrs-quant-a | **PASS**: 7004 launcher calls, 572,862,725 B, 0 failing |
| mistralrs-quant-c (MRQC_ONLY as integ3) | **PASS**: 4940 launcher calls, 9,360,379,960 B, 0 failing |

## Step 2: OrcaSAQ-2 27B. Root causes, each measured
All runs: all 64 layers on the GPU, 8192, MTP 0 unless stated. nvidia-smi used / free after load; the pool figure is TITAN_DEVMAP_LOG's
"pool used" before the first prompt step.
| binary / setting | after load | pool used |
|---|---|---|
| bin/mistralrs-titan-iq4xs (before) | 15690 / 244 MiB; even a ~1k-token request OOMs | n/a |
| orca, TITAN_RECURRENT_SLOTS=9 (= the merge alone) | 15690 / 244 MiB | 15152 MiB |
| **orca (2 slots, the default)** | **14634 / 1300 MiB** | **14118 MiB** |
| orca, MTP 1 | 14858 / 1076 MiB | 14334 MiB |

**(a) token_embd was already on the host, for every qwen35 model including the deployed 35B.** `quantized_qwen35_moe` loads
`token_embd.weight` with `ct.tensor(.., &Device::Cpu)`. QEmbedding keeps that CPU QTensor. `QTensor::embedding` on CPU storage
runs `dequantize_rows` on the host, and only the looked-up rows go to the GPU. The emb change (a763a3f1e) did not move it.
The measurement agrees. GPU weights are layers 0-63 at 12,734 MiB plus output at 995 MiB, i.e. 13,729 MiB. Add 2 slots
(296 MiB) and ~90 MiB of RoPE tables and norms, and you get the measured 14,118 MiB. With the table on the GPU it would be
~15.1 GB. Nothing to restore.

**The ~1 GB was the recurrent state pool.** `HybridCache::new` allocates `INITIAL_POOL_CAPACITY = 9` slots up front (upstream's
CUDA-graph batch range), and the titan qwen35 path never uses those graph buckets. One slot holds 48 GDN layers × 48 heads
× 128 × 128 f32, plus conv state: **147.8 MiB**. Nine slots cost 1330 MiB, where llama.cpp `-np 1` keeps one.
- **Fix (mistral.rs 90707cc05):** `HybridCache::with_slots` and `TITAN_RECURRENT_SLOTS` (default 2). The pools still double when every
  slot is taken.
- **Measured:** −1056 MiB nvidia-smi and −1034 MiB pool (7 × 147.8 = 1034.6).
- **Other models, after load:** Bonsai-2 27B 7402 → 6346 MiB; Bonsai-27B Q1_0 5386 → 4362 MiB; Qwen3.6-35B IQ2_M 11722 → 11274 MiB.
- **Deployed 35B (service roster):** its tiered auto fraction is computed before the cache is allocated, so it is unchanged at
  **0.54** (0.60 with the gate env). Free VRAM after load went from 2365 to **2781 MiB (+416)**. Throughput: 109.8 tok/s in the
  dry run vs 110.6 deployed, and in the gate runs 115.2 / 88.6 vs 113.0 / 89.5 (MTP 2 / MTP off). To turn the freed memory into
  experts, the auto fraction would have to plan for the smaller pool; that change was not made.

**(b) KV reservation.** The chunked prefill reserved KV in 8192-position buckets (512 MiB on this model). It now reserves the
bucket only when the bucket's extra positions plus a 512 MiB margin fit in free device memory (device free + pool slack).
Otherwise it rounds the request up to 512 positions (`kv_reserve_len`; prefix-cache restores too). The 35B and the other
regression models keep their buckets (outputs identical, below).

**Prompt-length limit, measured third cause: the 2048-row big prefill chunks.** These exist to amortise streamed-expert copies.
On this dense 27B they OOM at ~4k even with 1.3 GB free (out/pb2048.json). With 512-row chunks (`TITAN_PREFILL_BIG_CHUNK=0`, like
llama.cpp's -ub 512), 4k and 8k fit with ~180 MiB of prompt transient (out/pb512.json, prefix cache off). This is set per model in
the roster, not as a code default, so every other model is unchanged.

**(c) CPU/GPU split.** The hybrid cache put every layer's KV and GDN state on the main GPU, so a CPU-mapped GDN layer multiplied
its CPU activations by a CUDA state, giving "device mismatch in mul, lhs: Cuda, rhs: Cpu". Now each layer's caches live on that
layer's own device (HybridCache already accepted a per-layer device list). The slot indices are moved to the pool's device, and
prefix-cache restores rebuild KV per layer device.
- **Gate (out/c58*):** `-n 0:58` (6 layers on the CPU) gives **G1 40/40 vs llama.cpp, median dlp 0.0000, KL max 0.00005**, with 0
  mismatch lines and 0 errors. The iq4xs binary failed every request this way (top-iq4xs window 3 a64).
- **But:** the CPU layers are slow, ~5 tok/s prefill. The CPU IQ4_XS matmul and GDN recurrence are scalar candle ops. The auto map
  offloads only above 16k (at 32768: layers 61-63 on the CPU, 60-63 with MTP 1); 8192 and 16384 map all 64 layers to the GPU.

## (d) Results (window 2, all-GPU, 512-row chunks; out/om0*, om1*, lref*)
| | titan MTP 0 | titan MTP 1 | llama.cpp (-c 12288, f16 KV) |
|---|---|---|---|
| VRAM after load | 14634 MiB | 14858 MiB | 15034 MiB (the -c 8192 run: 14760) |
| largest prompt (ladder 2k..12k, each twice at 8k / 10k) | **8k: 7983 tokens OK twice** (peak 15596 MiB); 9983 OK only after an OOM retry; 12k OOM | **8k OK twice** (peak 15916 MiB, ~0 free); 10k OOM | 8k OK; G3l prompts to 9540 tokens |
| prefill (no prefix cache) | 1054 / 1057 tok/s at 4k / 8k | ~same | 1245-1264 tok/s at 4k-9.5k |
| decode, 8 × 300 greedy | **28.8 tok/s** | **29.8 tok/s** | 36.1 tok/s |
| MTP acceptance | n/a | **89.5%** (1000 steps, 1.90 tokens/step) | n/a |
| G1 vs llama.cpp | **top-1 40/40, median dlp 0.0000 (max 0.0003), KL max 0.00005** | identical to MTP 0 (top / tokens / text 40/40) | |
| G3 (4k) vs llama.cpp's spread | top-1 4/4, dlp 0.0038 (spread 0.0069), KL 0.00018 (spread 0.00007): **OUTSIDE (KL)**, absolute PASS | same | |
| G3l (6.8k-9.5k) | top-1 4/4, dlp 0.0037 (spread 0.0015), KL 0.00043 (spread 0.00001): **OUTSIDE**, absolute PASS | not run (OOM at 9.5k) | |
- **G3 caveat:** all four g3 prompts make OrcaSAQ predict `<|im_end|>` with p ≈ 1. llama.cpp prints that token as "", so titan's files
  were scored with special-token pieces mapped to "" (out/om*-g3*-sp.json). The raw scorer said 0/4 only because of the name.
  The residual KL gap is the same pattern devmap found for qwen3: bf16 residual stream vs llama.cpp's f32 activations. It is not
  fixed here.
- **Deploy choice: 8192, MTP 0.** MTP 1 decodes only ~3% faster on this dense model (verifying a draft costs nearly a full step) and
  leaves ~0 MiB at 8k.

## Regression and identity gates (window 3 unless noted; final binary 6cdca99d)
| gate | result |
|---|---|
| 35B MTP=2 vs m6/out/h-off.json | **40/40** (115.2 tok/s; deployed 113.0) m3/out/orca-m2.json |
| 35B MTP-off (MTP file, TITAN_MTP=0) vs m6/out/mtpfile-off.json | **40/40** (88.6 tok/s; deployed 89.5) m3/out/orca-mf0.json |
| Bonsai-2 B2-off vs o-off / vs deployed run (devmap b-off2) | **8/8, 8/8** (63.4 tok/s) |
| IQ2_M 8 × 300 vs deployed run (devmap i-new2) | **8/8** (57.3 tok/s) |
| Bonsai-27B Q1_0 8 × 300 vs deployed run (devmap q-new2) | **8/8** (60.1 tok/s) |
| gemma4-12b f32 / Spark / REDCELL G1 vs deployed runs (devmap window 6 ig/is/ir-new) | **top / tokens / text 40/40 each** |
| qwen3-14b, no pin, 16384, prefix cache off (window 2) | 40/40 layers on the GPU; **16184-token prompt OK** (peak 15814 MiB, 1712 tok/s prefill); short decode 48.0 / 47.8 tok/s; **G1 identical to deployed 40/40**; vs llama.cpp 40/40, dlp 0.0000, KL max 0.00024 |
| 9B IQ4_XS e2e (window 2) | TF top-1 **507/512**, top-5 512/512, mean / max abs dlogp **0.0109 / 0.1223**: equal to the iq4xs branch; greedy texts, tf and prefixes identical to its window-3 run (only timings differ) |
| roster dry run (models-deploy.toml, port 18660, the service's from-config, MemoryMax 20G) | **13/13 replied**; orcasaq2-cyber-27b loaded in 8.6 s, 1439 MiB free, 30.4 tok/s; then 1999- and 7887-token prompts OK (peak 15616 MiB, 29.7 tok/s); table in window3 log, out/dryrun.json |

## Not done / open
- (b) has no gate of its own that failed before it: once the pool fix and 512-row chunks were in, no measured prompt was refused because of the 8192 bucket (an ~8k prompt needs a full bucket either way). It is kept as a guard. Its effect on the other models is covered by the identity gates.
- G3 outside llama.cpp's own spread on KL (absolute PASS), as for qwen3. The likely cause is the bf16 residual stream (`--dtype f32`
  would double the KV).
- No code default for dense big chunks: the roster sets `TITAN_PREFILL_BIG_CHUNK = "0"`. A code default (big chunks only for models
  with MoE layers) would be cleaner but would need its own gate run.
- The 35B's auto expert fraction does not yet use the ~416 MiB the smaller recurrent pool frees.
- CPU-offloaded qwen35 layers work but are very slow (scalar CPU IQ4_XS and GDN).

Windows: window1-20261003-2108.log (25m33s), window2-20261003-2135.log (11m37s), window3-20261003-2147.log (11m44s). Each took
`flock .gpu.lock`, stopped titan-mistral and restarted it on exit (active afterwards); titan-spark was never started.
