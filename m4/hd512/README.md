# hd512: gemma4 prompt attention on flash-prefill (head dim 512) and the G3 drift

Branches `hd512` (all off `integ-20261002`), not deployed, not pushed:
| repo | worktree | commit |
|---|---|---|
| oxide-kernels | `oxide-hd512` | be03263 flash-prefill: head dim 512 kernels + f16-input twins, gate |
| mistral.rs | `top-hd512/mr-hd512` | 4aa8b028a gemma4 prompt attention routing, launcher, experiment knobs, dump |
| titan-engine | `top-hd512` (sparse candle, m4/hd512) | this README + scripts |

Binary: `m4/hd512/bin/mistralrs-w2`, sha256 `dff639e6a6a97f12...` (nvcc-free `--features oxide`, CARGO_TARGET_DIR
target-hd512-oxide, built from exactly the committed trees). Every gate below ran on that binary with default env.

## Result in one paragraph
gemma4's prompt attention now runs on oxide flash-prefill on both layer types (sliding: head dim 256 with the 1024-key
window; full: the new head-dim-512 kernel). Prefill at ~4k goes 1463 -> 2684 tok/s (llama.cpp 2993) and at ~8k
867 -> 2370 tok/s (llama.cpp 2978). G3 accuracy improves (4k |dlogprob| 0.0825 -> 0.0517; 8k KL 0.0187 -> 0.0106) but
**does not reach the limits (0.05 / 0.01)**: 4k KL 0.0158 and 8k |dlogprob| 0.0598 still fail. Measurement shows why:
the G3 limits at ~4k sit below the spread of llama.cpp's own legitimate configurations on this model (llama.cpp with a
bf16 KV cache: 0.0708 / KL 0.020 against its own f16 run; llama.cpp with flash attention off: 0.0995 / KL 0.020).
Per-layer dumps show titan's divergence grows evenly with depth and is the same size and shape as llama-bf16-KV vs
llama-f16; it is not concentrated in the head-dim-512 layers or in any single op. Making attention inputs F32 or
f16 (llama.cpp's precision) did not help (see the table). So: no claim that the gate is fixed.

## (1) Measurement before changing (window 1, integ binary, no build; window 2 dumps)
G3 ~4k (4 prompts, 3532-4652 tokens) and ~8k (7329-9631) vs the clean llama.cpp reference `m4/g4s/out/ref-g12-g3*.json`
(GPU, FA on, f16 KV), `--prefix-cache-n 0`:

| run | ~4k top-1 / dlp / KL | ~8k top-1 / dlp / KL |
|---|---|---|
| titan integ (eager bf16) t0 | 4/4, 0.0825, 0.0150 | 3/4, 0.0615, 0.0187 |
| titan TITAN_G4_ATTN_F32=1 t1 (prompt attention F32 on bf16 q/k/v) | 4/4, 0.0601, 0.0157 | 2/4, 0.0701, 0.0099 |
| titan `--dtype f32` t2 (everything F32) | 4/4, 0.0475, 0.0103 | OOM |
| llama.cpp FA on, bf16 KV l1 | 4/4, 0.0708, 0.0201 | 3/4, 0.0844, 0.0077 |
| llama.cpp FA off, f16 KV l2 | 4/4, 0.0995, 0.0203 | 2/4, 0.0825, 0.0116 |
| llama.cpp FA on, f16 KV l3 (= reference config) | 4/4, 0.0000, 0.0000 | 4/4, 0 |

Even titan in full F32 misses KL; llama.cpp itself fails both limits by changing only its KV cache type or turning
FA off. Logs: window1-20261002-2100.log, out/{t0,t1,t2,l1,l2,l3}-g3*.json.

Last-token dumps on G3 prompt 0 (titan `TITAN_G4_DUMP=DIR`; llama.cpp side `dump/g4dump.cpp`, a libllama cb_eval
dumper replaying the server's 512-token ubatches; comparison `dump/cmpdump.py`), relative L2 error vs llama.cpp f16-FA:

| layer (type) | titan eager e0 l_out | titan flash e1 l_out | llama bf16-KV l_out | llama FA-off l_out |
|---|---|---|---|---|
| 0 (sliding) | 7.4e-3 | 7.3e-3 | 3.1e-3 | 2.6e-3 |
| 5 (full, hd 512) | 1.6e-2 | 1.6e-2 | 1.5e-2 | 1.3e-2 |
| 23 (full) | 9.6e-3 | 9.9e-3 | 9.9e-3 | 9.2e-3 |
| 35 (full) | 3.0e-2 | 3.2e-2 | 3.3e-2 | 3.0e-2 |
| 47 (full, last) | 2.4e-2 | 2.4e-2 | 2.7e-2 | 2.9e-2 |

No jump at the full (head dim 512) layers; flash lowers the attention-output error at early layers (kqv_out L0
6.3e-3 -> 3.9e-3) and the rest tracks llama.cpp's own configuration spread. Full tables in window2-20261002-2157.log.
Root cause of the "drift": bf16-vs-f16/f32 rounding noise accumulated over 48 layers, amplified by G3's flat last-token
distributions (top logprobs -0.9 / -2.0 / -2.4); the eager path added some on top (bf16 scores), which flash removes.
Not found: any mask / -inf / softcap / scale bug in the eager path (the flash path, which has none of that machinery,
lands in the same place).

## (2) Kernel (oxide-hd512 be03263, flash-prefill/src/main.rs, gate src/gate512.rs)
- `flash_prefill_d512` (4 warps, launch_bounds(128,2)) / `_d512_w8` (8 warps): same tiling as the 256 kernels (8 heads x
  2 positions per warp m16 tile, 32-key tiles, cp.async K/V, ldmatrix, llama.cpp fattn-mma numerics). A block computes
  one 256-dim half of O (grid z = 2 x nsplit), S over all 512 dims (each half recomputes S in the same order, so both
  halves share m / l bit for bit); 128 f32 O accumulators per thread as at 256; Q fragments from L1 (no room in
  registers). Query-head groups of 8 map to KV head `group / gq` (gemma4-12b: 16 heads on 1 KV head, gq 2; REDCELL
  2 KV heads, gq 1). `flash_prefill_combine512` merges splits. Key window `win` as the 256 kernels.
- Shared memory 48 KiB static per block (K tile 32 x 1024 B + V half-tile 32 x 512 B): 2 blocks / SM on sm_120
  (100 KiB) for the 4-warp kernel = 8 warps / SM; 255 registers, 560 B local spill (the 256 kernels spill 176-536 B).
- f16-input twins `flash_prefill_d512_f16`, `_d512_w8_f16`, `flash_prefill_w8_f16` (f16 mma, f16 P, bf16 out), via a
  `F16` const generic on the bodies; used only by the off-by-default experiment.
- Gate (out/gate-fp.log, PASS): d512 bf16 31.5M outputs vs f64 worst 0.666 of tol (2^-7 |ref| + 2^-9), splits/variants
  worst 0.692, d512 vs d512_w8 bit-identical at nsplit 1, mutations (past - 1, wrong KV head, win + 1) 18/18; f16 d512
  worst 0.348, 8/8; f16 d256 worst 0.343, 8/8; existing 256 gate unchanged (0.589 / 0.602 / 8/8 / 6/6, vs v1 0 of 53.1M
  differ); **head-dim-256 bf16 kernels vs the integ PTX: 0 of 253.8M outputs differ** (148 calls, every variant,
  nsplit 1/2/3/8, windowed and not).

## (3) Routing (mr-hd512 4aa8b028a)
- `attention/flash_prefill.rs`: `attend_any` / `supported_any` (head dim 256 or 512, bf16 or f16; 256 bf16 goes
  through the unchanged `attend`); `mod.rs`: `flash_prefill_any[_supported]`.
- `vision_models/gemma4/text.rs` (attention only; no MoE edits): `flash_prompt` takes prompt passes (rows > 1,
  causal flash params, >= TITAN_ATTN_FLASH_PREFILL_MIN keys, default 1024) of both layer types after the KV-cache append,
  window = sliding_window on sliding layers. Short prompts (< 1024 keys) keep the eager path and its exact bits.
  Knobs: `TITAN_G4_ATTN_FLASH=0` (eager, = integ bit for bit), experiments off by default `TITAN_G4_ATTN_IN=f32`
  (projections / norms / RoPE in F32 with F32 norm weights) and `TITAN_G4_ATTN_KV=bf16|f16|f32`, debug
  `TITAN_G4_DUMP=DIR`.

Precision experiments (window 2, gemma4-12b, ~4k / ~8k dlp / KL):
| mode | ~4k | ~8k | prefill 4k |
|---|---|---|---|
| E0 flash off (identical to integ t0) | 0.0825 / 0.0150 | 0.0615 / 0.0187 | 1463 |
| **E1 flash bf16 (default)** | **0.0517 / 0.0158** | **0.0598 / 0.0106** (3/4) | **2729** |
| E3 F32 inputs, f16 flash (llama-like) | 0.0950 / 0.0234 | 0.0530 / 0.0080 (4/4) | 2640; G1 38/40 |
| E2 F32 inputs, F32 eager | 0.0943 / 0.0164 | 0.0939 / 0.0109 | 1122 |
| E4 F32 inputs, bf16 flash | 0.0992 / 0.0212 | 0.0693 / 0.0096 | 2632 |
None meets both limits on both lengths; E1 is the best overall and the default.

## Gates (window 3, final binary, default env) - window3-20261002-2330.log
| gate | result |
|---|---|
| gemma4-12b G1 (8192, default cache, integ config) | 40/40, dlp 0.0030, KL 0.00060; **top-20 identical 40/40 to integ** (recorded and re-run) |
| gemma4-12b G2 256 tok | **40/40 identical to the integ binary** (same procedure); vs llama 1/40 identical (as before) |
| G2 tie check (llama, first divergence) | median gap -0.025, > 1.0: 1/39 |
| gemma4-12b G3 ~4k, pc 0 | 4/4, **0.0517 (FAIL > 0.05)**, KL **0.0158 (FAIL)**; was 0.0825 / 0.0150 |
| gemma4-12b G3 ~8k, pc 0 | 3/4, **0.0598 (FAIL)**, KL **0.0106 (FAIL, marginal)**; was 0.0615 / 0.0187; no OOM, 13.4 GB |
| G3 repeat vs window 2 | identical (deterministic) |
| prefill tok/s ~4k / ~8k | 2684 / 2370 (integ 1463 / 867; llama.cpp 2993 / 2978) |
| REDCELL G3 ~4k, flash on (3 runs) | 4/4 each; dlp 0.114 / 0.096 / 0.142, KL 0.0068 / 0.0124 / 0.0259 |
| REDCELL G3 ~4k, flash off (3 runs) | 4/4 each; dlp 0.109 / 0.139 / 0.081, KL 0.0296 / 0.0400 / 0.0340 |
| REDCELL G1 | 39/40, dlp 0.0100, KL 0.00091 (integ 39/40, 0.0098-0.0100, 0.00091) |
| REDCELL prefill ~4k | 625 vs 560 tok/s (MoE-bound; redcell agent's area) |
| Spark G1 / G3 / G2 (integ procedure) | **40/40, 4/4, 40/40 identical to integ**; G3 4/4, 0.0296 |
| 35B MTP=2 vs m6/out/h-off.json | **40/40** (110.9 tok/s) |
| 35B MTP-off TITAN_MTP=0 vs m6/out/mtpfile-off.json | **40/40** (89.4 tok/s) |
| Bonsai-2 B2-off vs o-off | **8/8** |
| IQ2_M 8x300 vs integ binary's i-new | **8/8** |
| Bonsai-27B Q1_0 8x300 vs integ binary's q-new | **8/8** |
REDCELL is nondeterministic run to run (integ note 2), so it is compared by distribution: flash-on KL median of the
three runs 0.0124 vs flash-off 0.0340; dlp spreads overlap.

## Not done / open
- G3 limits not met (see above). What would plausibly be needed is matching llama.cpp's numerics throughout (F32
  residual stream and norms, f16 KV), not attention alone; full F32 titan still gave KL 0.0103 at 4k. Recommend
  re-basing the G3 gate on a noise floor (e.g. llama.cpp bf16-KV or FA-off vs reference) rather than absolute 0.05/0.01.
- Sliding layers pad 2 query heads to 8 per KV head (4x wasted MMA on 40 of 48 layers) and the host still builds the
  full masks for prompts; both are the remaining prefill gap to llama.cpp (~10% at 4k, ~20% at 8k).
- The d512 kernel spills 560 B; FP_TIME timings for d512 were not printed (FP_TIME only timed the 256 shapes here).
- No eval-callback text dumps were used: g4dump writes full last-token rows instead (same tensor names).

## Windows (3 of 3)
1. 21:00 (5 m): diagnostics on the integ binary + llama.cpp variants (no build).
2. 21:57 (13 m): kernel build + gate, export PTX, build, precision modes, dumps, Spark identity, REDCELL.
3. 23:30 (18 m): final-binary gates and regression core (no rebuild: the window-2 binary is the committed tree).
