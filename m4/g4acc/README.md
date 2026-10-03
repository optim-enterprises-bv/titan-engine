# g4acc: gemma4 llama.cpp precision layout, GQA-2 flash kernel, no host masks

Branches `g4acc` from `integ2-20261003`, not deployed, not pushed:
| repo | worktree | head |
|---|---|---|
| oxide-kernels | `oxide-g4acc` | 58e4a75 (c1ea67a kernels + gate, 58e4a75 gate fix + regenerated PTX) |
| mistral.rs | `top-g4acc/mr-g4acc` | 492a555bc (21b59d9e3 + the F16-cache fix) |
| titan-engine | `top-g4acc` (sparse candle, m4/g4acc) | this README, scripts, logs |

Final binary `m4/g4acc/bin/mistralrs-w2`, sha256 `d8a17a3f861915e8...` (built from 492a555bc clean, embedded PTX == oxide 58e4a75).

## Verdict
- **G3 is not met** (|dlogprob| < 0.05 and KL < 0.01 at ~4k and ~8k) by the llama.cpp precision layout or by the bf16 path.
  The same G3 comparison fails **for llama.cpp against itself** when only its ubatch size changes. That is the same
  binary, the same F16 KV cache, flash attention on, the same weights:
  `-ub 256` 4k 0.0413 / KL **0.0164**, 8k top-1 **2/4**; `-ub 128` **0.1130** / **0.0195**; `-b 512` **0.0879** / **0.0259**.
  So on these prompts the gate can only pass for an implementation that is bit-identical to the one reference run.
  It cannot be met by matching precision. Evidence and per-layer comparison below.
- Sliding-layer padding fixed (`flash_prefill_r2_w8`, GQA 2 at head dim 256 with the key window). Host masks are no
  longer built for flash prompt passes. **bf16 prefill ~4k / ~8k: 3210 / 2972 tok/s** (median of the four G3
  prompts; first prompt 3269 / 3060; integ2 2727 / 2363; llama.cpp 2993 / 2978).

## 1) Precision layout (llama.cpp's) and its pieces
Implemented for gemma4 (`vision_models/gemma4/text.rs`, attention only + mask skip), switched on per model by
`--dtype f32` (`dtype = "f32"` in a roster entry):
- residual stream, norms, softcaps, router, every activation between ops in F32. This is the model dtype. Quantized
  matmuls stay MMQ on the quantized weights with F32 output; nothing is dequantized (see Q4_0 below).
- KV cache in **F16** (`TITAN_G4_KV16`, default on under F32). The engine preallocates a sequence's cache in the
  activation dtype, so an empty F32 one is dropped before the first F16 append.
- attention on F16 q / K / V with F32 accumulation and **F32 output**, prompt and decode, on the new
  `flash_prefill_r2_w8_f16o32` (sliding, window 1024) and `flash_prefill_d512_f16o32` (full) kernels.

| piece (gemma4-12b, --prefix-cache-n 0) | ~4k top-1 / dlp / KL | ~8k top-1 / dlp / KL | G1 |
|---|---|---|---|
| integ2 bf16 (flash bf16, padded sliding heads) | 4/4, 0.0517, 0.0158 | 3/4, 0.0598, 0.0106 | 40/40 |
| P0 bf16 + r2 kernel + no masks (default, `--dtype bf16`) | 4/4, 0.0311, 0.0112 | 3/4, 0.0598, 0.0106 (bit-identical to integ2) | 40/40, top-20 identical to integ2 |
| P1 F32 activations + F32 cache + eager F32 attention (`TITAN_G4_KV16=0`) | 4/4, 0.0475, 0.0103 | OOM | broken (0/40, see open items) |
| **P2 F32 activations + F16 cache + f16 attention, F32 out (llama.cpp layout)** | 4/4, **0.0748**, **0.0182** | 3/4, **0.0767**, 0.0079 | 39/40, dlp 0.0030, KL 0.00038 |
| llama.cpp vs itself, bf16 KV (hd512 window 1) | 4/4, 0.0708, 0.0201 | 3/4, 0.0844, 0.0077 | |
| llama.cpp vs itself, FA off | 4/4, 0.0995, 0.0203 | 2/4, 0.0825, 0.0116 | |
| llama.cpp vs itself, `-ub 256` / `-ub 128` / `-b 512` | 0.0413 / 0.1130 / 0.0879, KL 0.0164 / 0.0195 / 0.0259 | 2/4 / 4/4 / 2/4, 0.0413 / 0.0981 / 0.0797 | |

P0's 4k improvement over integ2 is only split noise. The r2 kernel computes every row in the same order, so 8k is bit-identical.
At 4k the different block count picks a different split count.

**Per-layer comparison**: last-token rows of P2 against llama.cpp f16-FA on G3 prompt 0 (`TITAN_G4_DUMP`,
`dump/g4dump.cpp`), next to llama.cpp bf16-KV against the same reference (window2-*.log has the full table):

| layer | P2 kqv_out | P2 attn_out | P2 l_out | llama bf16-KV l_out |
|---|---|---|---|---|
| 0 | 5.7e-4 | 1.25e-3 | 2.4e-3 | 3.1e-3 |
| 1 | 3.2e-3 | 4.8e-3 | 3.8e-3 | 5.7e-3 |
| 4 | 9.5e-3 | 1.27e-2 | 1.56e-2 | 1.58e-2 |
| 23 | 3.2e-2 | 7.4e-3 | 1.01e-2 | 9.9e-3 |
| 41 | 3.6e-2 | 4.6e-2 | 4.6e-2 | 4.8e-2 |
| 47 | 3.1e-2 | 3.7e-2 | 2.7e-2 | 2.7e-2 |

- The layout does what it should at the start. Layer 0's attention output differs from llama.cpp's by 5.7e-4 (bf16 titan: 6.3e-3). That
  residue comes from operations that are not bit-identical: RoPE from F32 tables vs llama's on-the-fly sin/cos,
  `ex2.approx` vs `expf`, a different FA split / tile order and MMQ stream-k order. Each one flips some f16 roundings of K/V.
- It then grows about 2x per layer for the first layers: kqv_out-0 5.7e-4 -> attn_out-0 1.25e-3 -> l_out-0 2.4e-3. By layer
  4 it is at the 1.5e-2 plateau that **every** perturbed run reaches, llama.cpp's own bf16-KV run included. The
  amplifier is common to both engines: every matmul re-quantizes its input activations to q8_1 (int8 per 32-value
  block). A difference far below one quantization step still flips some int8 roundings, and each flip is an error of up to half
  a step (~0.4% of the block maximum). So any perturbation at the 1e-4 level turns into the same ~1-4% at the last
  layer within a few layers, whichever op caused it. The G3 prompts' last-token distributions are flat (top logprobs
  ~ -0.9 / -2.0 / -2.4), so that shows up as dlp 0.04-0.11 and KL 0.01-0.026 for titan and for llama.cpp's own variants.
- Nothing is concentrated in one layer type or one op. No layer jumps above the llama-bf16-KV profile.

Q4_0 dequant: gemma4's prefill and decode do not dequantize Q4_0. All Q4_0 matmuls are MMQ / MMVQ on q8_1
activations. `TITAN_KTRACE` of the P2 run (out/q2.ktrace) lists one dequant kernel, `dequantize_block_q6_K_f32`
(the Q6_K token-embedding row gather). So the candle Q4_0 dequant discrepancy (`deq` agent) is not on this path and is not
part of the drift.

Cost of P2 vs the bf16 default (gemma4-12b, same binary):
| | P0 bf16 | P2 F32 / F16 KV |
|---|---|---|
| prefill ~4k (G3 median of 4) | 3210 tok/s | 3086 tok/s |
| prefill ~8k | 2972 tok/s | 2803 tok/s |
| decode at ~3.5k context (G5) | 51.6-52.8 tok/s (integ2 51.5 / 52.3) | 71.4 / 72.5 tok/s |
| decode at ~540 (G5) | integ2 62.1 | 72.1 |
| VRAM after load / after the ~8k G3 | 7.3 / 13.0 GB | 7.2 / 14.6 GB |
P2's faster decode comes from decode attention on the f16 flash kernel instead of eager bf16. The same could be done in bf16 (not done).

REDCELL (shares the text tower), G3 ~4k vs ref-red-g3, --prefix-cache-n 0:
- P2 (`--dtype f32`): two fresh servers bit-identical (4/4); 4/4, dlp **0.0888**, KL **0.0135**. G1 38/40, dlp
  0.0035, KL 0.00040. Prefill 652 tok/s, decode 109-113 tok/s at ~3.5k, VRAM 14.8 GB.
- P0 (bf16): 4/4, 0.0387, KL 0.0106. Prefill 660 tok/s, VRAM 14.2 GB. Integ2 recorded 0.0614 / 0.0264. REDCELL
  prefill is MoE-bound.

## 2) Kernels (oxide-g4acc, gate PASS: out/gate2-fp.log)
- `body` gains `REP` (query heads per KV head in the m16 warp tile: row r = head r % REP, position r / REP; REP 8 =
  the old 8 heads x 2 positions) and `O32` (f32 output). New entries:
  - `flash_prefill_r2_w8` (bf16, GQA 2, 8 positions per warp, 64 per block);
  - `flash_prefill_r2_w8_f16o32`, `flash_prefill_w8_f16o32`, `flash_prefill_d512_f16o32`;
  - `flash_prefill_combine_f32`, `flash_prefill_combine512_f32`.
- Gate (f64 reference, tolerance 2^-7 |ref| + 2^-9, splits 1 / 2 / 3 / 8, NaN-poisoned tails):
  - r2 bf16: 4.78M outputs, worst 0.604 of tol, mutations 11/11. Cases include decode rows (s = 1) and window 1024 at past 1023 / 2000 / 4608.
  - r2 f16 o32: worst 0.077, 11/11.
  - d512 f16 o32: worst 0.068, 9/9.
  - w8 f16 / o32: worst 0.344, 8/8.
  - d512 bf16 and d512 f16 unchanged: 0.666 / 0.342.
  - **existing kernels against the integ2 PTX: 292 calls, 0 of 632,881,152 outputs differ in any bit.** That covers the 256 bf16 x5, the 256 f16, the 512 bf16 x2 and the 512 f16 x2 kernels.
  - Window-1 gate run: one weak mutation. `past - 1` on a lone decode row over 3000 keys stays below tol/8 once the
    output is f32 (no bf16 rounding noise). That mutation now runs only for s >= 8; every suite still has its past-1
    cases. The fix was gate code only: the PTX was byte-identical after it.
- 255 registers. Spills 192 B for the r2 kernels and 560 B for d512. Static shared memory 32 KiB (r2) and 48 KiB (d512). FP_TIME
  (out/gate-fp.log), s = 4096 / 8192:
  - r2_w8 1.02 / 2.07 ms per sliding layer (16 heads, 8 KV heads, window 1024);
  - d512 9.4 / 35.2 ms per full layer;
  - the o32 twins within 2%.
- Launcher (`attention/flash_prefill.rs`): `attend_any(..., out32)`. GQA 2 at head dim 256 takes the r2 kernels with
  no padded heads. `launch_named` takes `krep` and `out32`, plus f32 combines.

## 3) No host masks
For a first causal prompt chunk (batch 1, CUDA, not paged) whose layers all take flash, the forward skips CausalMasker's host
(rows x keys) masks and passes `CausalFlash`. The flash kernels apply causality and the window. A layer that falls back to
eager gets its mask built on the device (`eager_attention_mask`, now pub(crate)). The marker survives the per-layer mask adjust.
Integ2 -> P0 prefill at ~8k: 2363 -> 2972 tok/s median (integ2 binary re-run in window 2), from this plus the r2 kernel.

## Gates on the final binary (window 3 unless noted)
| gate | result |
|---|---|
| 35B MTP=2 vs m6/out/h-off.json | **40/40** (107.4 tok/s) |
| 35B MTP-off, TITAN_MTP=0 vs m6/out/mtpfile-off.json | **40/40** (89.3 tok/s) |
| Bonsai-2 B2-off vs o-off | **8/8** |
| IQ2_M 8x300 vs bin/mistralrs-titan-integ2's run | **8/8** |
| Bonsai-27B Q1_0 8x300 vs integ2's run | **8/8** |
| Spark G1 / G3 / G2 (integ2 procedure) | **40/40, 4/4, 40/40 identical to integ2** (also in window 1) |
| gemma4-12b G1 bf16 (integ2 procedure) | **40/40**, dlp 0.0030, KL 0.00060, **top-20 identical 40/40 to integ2** |
| gemma4-12b G2 bf16 | **40/40 identical to the integ2 binary** (same procedure) |
| gemma4-12b P2 G1 / G2 + tie (window 2) | 39/40; G2 vs llama 3/40 identical; tie: median gap -0.055, max 0.798, > 1.0: 0/37 |
| gemma4-12b G3 | **FAIL**, both layouts (table above) |
| P2 determinism | G3 re-run identical 4/4 |
| kernel gate | PASS |

## Open
- G3 as specified: not achievable without bit-matching one llama.cpp configuration (see the llama.cpp-vs-itself rows).
  Recommendation: gate against the spread of llama.cpp's own variants, or use teacher-forced perplexity / KL over many
  positions instead of a single flat last-token distribution.
- P1 (`--dtype f32` with `TITAN_G4_KV16=0`) gives garbage on short prompts (G1 0/40), while its ~4k G3 matches the integ
  binary's earlier `--dtype f32` run (0.0475). That eager-F32 short-prompt path was never gated. Not investigated:
  P2 (the supported F32 layout) does not use it. Do not run gemma4 with `--dtype f32` and `TITAN_G4_KV16=0`.
- The default stays bf16 (P0). P2 is opt-in per model (`dtype = "f32"`). It costs ~4-5% prefill and 1.6 GB VRAM at ~8k,
  gains ~40% decode, and does not improve G3.
- Decode on the f16 flash kernel could be brought to the bf16 path for the same decode gain (not done).

## Windows (3 of 3)
1. 06:00 (12 m): kernel build + gate (one weak mutation), build, P0 / P1 (P2 failed: F32-preallocated cache), REDCELL P0, Spark.
2. 06:57 (11 m): gate re-run PASS, build with the cache fix, P2 G3 / G1 / G2 / tie / G5 / KTRACE, integ2 G5 reference, P2 dump, REDCELL P2 x2.
3. 07:14 (16 m): regression core, gemma4 bf16 identity, Spark identity, llama.cpp self-perturbation, P2 determinism.

## Follow-up (window 4, window4-*.log; binary `bin/mistralrs-w4`, sha256 2059d07907fde8be, mr-g4acc f6839fa5a, oxide 718a131)

### bf16 decode on the flash kernels
Prompt passes already ran on flash-prefill. Decode rows now do too, at any context length (`flash_prompt`; the
default bf16 path; `TITAN_G4_FLASH_DECODE=0` puts decode back on eager). There is no new kernel:
- sliding layers: `flash_prefill_r2_w8` (its gate already had s = 1 cases);
- full layers: `flash_prefill_d512`, whose gate now also has decode rows (s = 1 over 0 / 3000 / 5000 / 9000 keys).

Gate PASS (out/gate4-fp.log):
- d512 bf16: 31.7M outputs, worst 0.666 of tolerance, mutations 19/19;
- r2 bf16: worst 0.546, 11/11;
- existing kernels vs the integ2 PTX: 0 of 632,881,152 outputs differ;
- the PTX is byte-identical to window 2's (only the gate changed).

Decode tok/s, G5 (128 greedy tokens), gemma4-12b, --prefix-cache-n 0:
| context | new bf16 | new bf16, TITAN_G4_FLASH_DECODE=0 | integ2 | new `--dtype f32` (F16 KV) | llama.cpp |
|---|---|---|---|---|---|
| ~540 | **72.0 / 72.3** | 61.4 / 62.1 | 60.3 / 60.8 | 71.9 / 71.8 | 71.8 / 72.2 |
| ~3.5k | **70.5 / 71.7** | 51.7 / 52.0 | 50.2 / 51.8 | 71.0 / 72.1 | 69.3 / 69.6 |
| ~7.3k | **69.6 / 70.6** | 51.0 / 53.5 | 52.2 / 53.6 | 67.5 / 68.6 | 68.8 / 69.8 |
Prefill on the same G5 requests: ~3.5k 3190 / 3159 (integ2 2624 / 2715, llama.cpp 2980 / 2941); ~7.3k 2983 / 2944
(integ2 2280 / 2360, llama.cpp 2918 / 2926).
- bf16 G1 (integ2 procedure): 40/40, dlp 0.0030, KL 0.00060, top-20 identical 40/40 to integ2 (first token = prompt pass, unchanged).
- G2 text changes (2/40 identical to the previous binary), as expected with a different decode kernel.
  Against llama.cpp it is 5/40 identical (was 1/40). Tie check at each first divergence: median gap -0.010,
  max 0.476, **0 of 35 above 1.0**.

### `--dtype f32` with `TITAN_G4_KV16=0` "garbage": a harness bug, not a model bug
- Window 1's run_mode called `tmap gemma4 p1-g3.json p1-g3l.json p1-g1.json`. The ~8k request had OOM'd, so
  p1-g3l.json did not exist. tokmap.py stops at the first missing file, so p1-g1.json was never mapped from token ids
  to strings, and cmp.py scored ids against llama.cpp's strings (0/40, dlp 13.8).
- Mapping that same window-1 file now gives **39/40, dlp 0.0029, KL 0.00049**. Its top-20 is identical 40/40 to
  window 4's fresh-server run.
- `lib.sh` `tmap` now maps only the files that exist and names the missing ones.
- The combination itself, checked in window 4, all G1 at --prefix-cache-n 0, 12288:

  | combination | G1 |
  |---|---|
  | bf16 | 40/40 (integ2 procedure) |
  | f32 + KV16 | 40/40, dlp 0.0017, KL 0.00035 (integ2 procedure); 39/40 at pc 0 (window 2) |
  | f32 + KV32, fresh server | 39/40, dlp 0.0029, KL 0.00049 |
  | f32 + KV32, after a ~8k OOM on the same server | 39/40, identical |

  Per-layer dumps of G1 prompt 0, fresh vs after an OOM: 0 difference in every layer.
- f32 + KV32 is correct. Its limit is capacity: ~4.6k prompts run (G3 ~4k 4/4, 0.0475 / 0.01026), but ~7.3k+ prompts
  OOM on 16 GB. The cause is the F32 KV cache plus eager F32 attention. The request fails cleanly and the server
  recovers. Not made a load-time error, because it is correct wherever it fits; use F16 KV (the default) for long contexts.

### G3 against llama.cpp's own spread (reusable)
`g3spread.sh GGUF OUTDIR` (GPU, inside a window) regenerates llama.cpp's self-variation. It is the reference
server, f16 KV, FA on, with only `-ub 256` / `-ub 128` / `-b 512` changed. `g3spread.py REF VARIANTS -- TITAN...` (CPU)
scores everything against the reference with cmp.py's metrics. A titan run is "within" the spread when its top-1
agreement is no worse, and its median dlp and KL median are no higher, than the worst variant's.

| G3 | llama.cpp worst variant | titan bf16 (default) | titan `--dtype f32` (F16 KV) | integ2 |
|---|---|---|---|---|
| ~4k | 4/4, dlp 0.1130, KL 0.02591 | 4/4, 0.0311, 0.01118: **within** | 4/4, 0.0748, 0.01820: **within** | 4/4, 0.0517, 0.01576: within |
| ~8k | 2/4, dlp 0.0981, KL 0.01030 | 3/4, 0.0598, 0.01056: **outside by KL** (0.01056 > 0.01030) | 3/4, 0.0767, 0.00787: **within** | 3/4, 0.0598, 0.01056: outside by KL |
The absolute G3 limits (0.05 / 0.01) fail for every row, every llama.cpp variant included.

### Gates on bin/mistralrs-w4
| gate | result |
|---|---|
| 35B MTP=2 vs m6/out/h-off.json | **40/40** (111.3 tok/s) |
| 35B MTP-off vs m6/out/mtpfile-off.json | **40/40** (86.7 tok/s) |
| Bonsai-2 B2-off vs o-off | **8/8** |
| IQ2_M 8x300 vs integ2's run | **8/8** |
| Bonsai-27B Q1_0 8x300 vs integ2's run | **8/8** |
| Spark G1 / G3 / G2 | identical to integ2 (40/40, 4/4, 40/40); G1 vs llama.cpp 39/40, dlp 0.0124, KL 0.00371 |
| REDCELL G1 (bf16, pc 0) | 39/40, dlp 0.0084, KL 0.00078; decode at ~3.5k 113 / 117 tok/s |
| REDCELL G1 (`--dtype f32`, window 2) | 38/40, dlp 0.0035, KL 0.00040 |
| gemma4 G1 bf16 / f32 + KV16 / f32 + KV32 | 40/40 / 40/40 / 39/40 (table above) |
| kernel gate | PASS |
