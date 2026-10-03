# redcell: REDCELL-26B determinism and long-prompt prefill (branch `redcell`, staged, NOT deployed)

Model: `~/ai/models/redcell-26b/REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf` (gemma4 MoE, 30 layers, 128 experts top-8,
expert FFN 704: gate_up Q3_K, down Q4_0 in 10 layers / IQ4_NL in 20). Server args everywhere:
`--max-model-len 8192 --max-seq-len 8192 -n 0:30 --dtype bf16 --paged-attn off`.

## Branch heads (all `redcell`, off `integ-20261002`)
| repo | worktree | head | contents |
|---|---|---|---|
| oxide-kernels | `oxide-redcell` | 2695cb5 | deterministic `moe_gemv_down_aggregate`, stable `moe_dispatch` scatter, gate checks |
| mistral.rs | `top-redcell/mr-redcell` | 8830aa217 | grouped unaligned-K prompt route (opt-in), IQ4_NL MoE MMQ launch, gemma4 MoE probes |
| titan-engine | `top-redcell` (sparse: candle, m4/redcell) | this README | windows, scripts, gate outputs |

Binaries (scratch, not deployed): `m4/redcell/mistralrs-redcell` = window-3 build of 5cab9fa4a (sha256 ad18cbc6860f5877).
8830aa217 (final) was NOT built (no window left); it only flips the unaligned-K prompt default back to the decode kernels
(= the window-3 binary run with `TITAN_MOE_UNALIGNED_DECODE=1`) and reverts a disproven J=128 tweak.

## Problem 1, nondeterminism: FIXED
Root cause: `moe_gemv_down_aggregate` (oxide quant-a, twin of mistralrs-quant `indexed_moe.cu`) summed the top-k slots of a
token into the output with float atomics (grid y = topk), so the order of the 8 adds, hence the bits, was the hardware's.
REDCELL runs every decode step and every prompt shorter than 32 tokens (all of G1/G2) and, since a7ea678f9, every longer
prompt through it. The 35B / IQ2_M / Bonsai use other expert paths, which is why they were byte-deterministic.

Evidence (`TITAN_G4_MEMLOG=2`, G1 prompt 0, 8 tokens, two fresh servers, `firstdiff.py`):
- integ binary: layer outputs identical for the first 168 probes (prefill + 4 decode steps), first difference L18 of decode
  step 5 (sum -56.809975 vs -56.810463), everything after differs (window1 log).
- fixed binary (window 2, with probes of shared MLP, router weights, router ids and expert output per layer):
  1350 / 1350 probes identical.

Fix (oxide 2695cb5): `down_aggregate` grid ((n+3)/4, 1, batch), one row per warp, the slots added in slot order with `.ftz`
adds = exactly what the atomics give when they arrive in slot order (so the reference cases with one weighted slot stay
bit-identical). `moe_dispatch` gets a stable scatter (one warp per expert, warp prefix sum, assignment order) instead of the
atomic one, so grouped MMQ prefill sees one column order per input (tile membership and stream-k splits no longer vary).
The C/.cu reference (nvcc builds) is unchanged and still atomic.

| gate | result | log |
|---|---|---|
| quant-a kernel gate vs libmistralrsquant.a | **PASS 7004 launcher calls, 572862725 bytes, 0 failing** (was 6882): + per format/shape a slot-order check (all 8 slots weighted: full sum == host `.ftz` fold of the one-slot results, bit for bit, and == a rerun) and a stable-dispatch check | out/gate-qa.log |
| determinism, 3 fresh servers, G1 -> G3 -> G2 256 (default cache), window 2 build | **G1 top-20 40/40, G2 text 40/40** (a vs b, a vs c) | window2 log, out/rd-*.json |
| same, window 3 build (final kernels) | **G1 top-20 40/40, G2 text 40/40** | window3 log, out/fd-*.json |
| integ, same procedure (integ README) | G1 30/40, G2 1/40 | m4/integ |
| G1 vs llama.cpp ref-red-g1 | 39/40, median dlp 0.0140, KL median 0.00085 (integ 39/40, 0.00091); the miss (prompt 39) is an exact titan tie 'To'/'A' | window3 |
| G2 vs llama.cpp ref-red-g2 | 2/40 identical, common prefix median 31.5 tokens (integ ri-g2a 2/40, 28.5) | window3 |

Baseline re-frozen: `m4/g4s/out/baseline-redcell-g2.json` (+ `.md`, procedure and order) = fd-a-g2 from the window-3 build;
the old one is `baseline-redcell-g2-w5.json` (3/40 texts equal). All 40 G2 prompts are 23-28 tokens, so they run the decode
kernels on every routing setting.

## Problem 2, long-prompt prefill: profiled; batched route built but WRONG, left opt-in (not fixed)
Profile first (ncu, one 3547-token prompt, prefix cache 0, `kern.py`):
| | integ (a7ea678f9 route) | grouped route (window 2 binary, wrong output) |
|---|---:|---:|
| GPU time | 5485 ms | 1259 ms |
| MoE experts | 4567 ms (83%): `moe_gemv_fused_gate_up_q3k` 3126 (57%), `down_aggregate_iq4_nl` 867, `_q4_0` 398 | 354 ms |
| attention (eager hd-512 / cuBLAS, mask add, softmax, casts) | 581 ms (11%) | 562 ms (45%) |
So the per-token fused path is the whole gap; once it is batched, attention is the next item (left to the hd-512 agent).

Built (mr 460b1ee89..5cab9fa4a): unaligned-K prompts run `forward_grouped` in <= 2048-token pieces: packed Q3_K gate/up MMQ,
Q4_0 down via the existing grouped MMQ (K=704 is fine for the tile loop: the last 256-wide K step reads the next row's blocks
against zero activations, as llama.cpp does; the tensor tail is zero-padded by candle's `load_quantized`), IQ4_NL down via a new
`fast_mmq::grouped_llama_from_glu_packed` (D4 GLU quantize + the oxide `iq4_nl_mmq_j*` port in its MoE mode + fixup).
Speed: G5 3.5k prefill 1947 / 1708 tok/s vs 558 / 566 integ (decode unchanged 79-80; G2 decode 112-113 = integ 112).
Correctness: **broken**. G3 vs llama.cpp 0/4, top-20 KL median 2.75 (window3 fp-g3); a 547-token prompt gave NaN expert rows
from L2 on; with only the IQ4_NL layers on the decode kernels the NaN is still at L2 (a Q4_0 layer), with only the Q4_0 layers
on decode it appears at L10 (IQ4_NL), so the fault is upstream of both down formats: the packed Q3_K gate/up pair
(`grouped_pair_packed`, never exercised by a gated model before) or the shared GLU activation quantization. Forcing tile width
J=128 did not change it. The non-finite guard (recompute that forward on the decode kernels) fired in 25 of 30 layers.
Shipped state (8830aa217): unaligned-K prompts on the decode kernels by default (integ behaviour, now deterministic); the
grouped route only with `TITAN_MOE_UNALIGNED_GROUPED=1` (+ `TITAN_MOE_UNALIGNED_DECODE=1|iq4_nl|q4_0` per-format).
Next step: gate `grouped_pair_packed` (Q3_K, K=2816, GELU) against `forward_decode`'s gate/up on the device, row by row.
REDCELL G3 / G5 on the default route were not re-measured on the final build: same kernels as window 3 with
`TITAN_MOE_UNALIGNED_DECODE=1` (3547-token prompt 582 tok/s, 547 680 tok/s, outputs fine); integ G3 4/4, KL 0.019.

## Regression core and shared-code gates (window-3 binary; also all passed on the window-2 binary)
| gate | result |
|---|---|
| 35B MTP=2 vs m6/out/h-off.json | **40/40** (m3/out/redf-m2.json) |
| 35B MTP-off file, TITAN_MTP=0 vs m6/out/mtpfile-off.json | **40/40** (m3/out/redf-mf0.json) |
| Bonsai-2 B2-off vs m4/bonsai2/out/o-off.json | **8/8** |
| IQ2_M 8x300 vs integ binary's run | **8/8** |
| Bonsai-27B Q1_0 8x300 vs integ binary's run | **8/8** |
| gemma4-12b G1 vs integ g12-g1 | **40/40** logprob-identical |

## Not done / caveats
- Batched REDCELL prefill is not correct (above); the 580 tok/s long-prompt prefill is unchanged by default.
- G3 under the default prefix cache still OOMs at its first ~3.5k request (pre-existing; fd-a log).
- mistralrs-quant `moe_grouped` `grouped_store` (non-MMQ grouped GEMM with routing weights) still uses float atomics;
  REDCELL does not reach it.
- Window 1 was cut at 8 min (a `Tensor::copy` kept the narrow offset and panicked the engine; fixed with
  `force_contiguous`); windows 2 and 3 full. 3 of 3 used; titan-spark restarted after each, titan-mistral never started.

Scripts: lib.sh, window{1,2,3}.sh, one.py, diag.py, firstdiff.py, nanprobe.py, kern.py, same.py, b2client.py.
ncu reports (out/*.ncu-rep, 360 MB) are left uncommitted in the working directory.
