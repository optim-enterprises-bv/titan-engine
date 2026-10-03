# GGML IQ4_XS (type 23) as a dense weight type, 2026-10-03

Branch `iq4xs` in all three repos (base: top-level master 6d9dc9b6, mistral.rs model-swap 86c8f605a, oxide-kernels
glu-typed-kernels 64d8b5a). Worktrees `top-iq4xs` (sparse: candle, m4/iq4xs), `top-iq4xs/mr-iq4xs`, `oxide-iq4xs`
(`top-iq4xs/oxide-kernels -> ../oxide-iq4xs`). Binary **`bin/mistralrs-titan-iq4xs`**, sha256
`3ab0f100e5e0f2728a9ec44deb7fa07eaf540528bfabe63922c30201e8ee3e32` (window 3; nvcc-free `--features oxide`,
CARGO_TARGET_DIR target-iq4xs-oxide seeded from target-integ3-oxide). Not deployed; deploy/models.toml untouched.

| repo | commit | what |
|---|---|---|
| oxide-kernels | 91452a1 | iq4_xs: CPU oracle + 4 mutants + 27B/9B shapes in the gate; fmt_mmq: `mul_mat_q<IQ4_XS>` (32 entries) + 2 loader mutants, gate, export of `iq4_xs_mmq_oxide.ptx` |
| titan-engine (candle) | 4b520cf7 | IQ4_XS in `llama_fmt` (GPU dequant f32/f16, mmvq decode incl. small_k), test `cuda_iq4_xs` |
| titan-engine (candle) | 98d4535f | CPU `k_quants::matmul`: lhs scratch in VecDotType blocks (i-quant CPU layers panicked) |
| mistral.rs | 1f8bf6a3b | `fast_mmq`: IQ4_XS dense prefill through the MMQ port (fi 4, IQ4_NL's quantizer) |

Prefill path: **MMQ** (llama.cpp `mul_mat_q<IQ4_XS>`, ported), not dequant + cuBLAS. Decode: mmvq (batch <= 8).

## Gates
| gate | result | evidence |
|---|---|---|
| iq4_xs dequant f32/f16/bf16 + mmvq 1..8 (+small_k) vs llama.cpp acecd56 cubins | **PASS 1157 launches, 15612608 B, 0 failing**; CPU oracle (libggml-base `dequantize_row_iq4_xs`, 4096 blocks) equal; **4/4 mutants detected** | out/gate-iq4xs.log |
| fmt_mmq IQ4_XS MMQ (+fixups, all J / fallback, stream-k grids, channels, 27B shapes) vs `mmq-instance-iq4_xs.cu` cubin (built in window 1, ref/nvcc_mmq_iq4_xs.sh) | **PASS 230 launches, 208754192 B, 0 failing; 2/2 loader mutants detected** | out/gate-fm-iq4xs.log |
| existing PTX | fmt_mmq: **105/105 existing entries identical** to the committed PTX after normalising label / shared-array numbers and Rust symbol hashes (ptxcmp.py); mistral.rs keeps the committed iq4_nl/mxfp4/nvfp4 modules **byte-identical** (export restored them); the new iq4_xs module's 35 entries == fmt_mmq.ptx. iq4_xs.ptx (not embedded anywhere before) differs in instruction order only; its gate is the evidence | window1 log |
| negative control: deployed binary on the 9B IQ4_XS file | `unsupported dtype for quantized matmul IQ4XS` on every request | window2 log |
| 9B e2e (Qwen3.5-9B-Claude-Distill-v2 Q8_0 -> llama-quantize IQ4_XS, token_embd/output q8_0; qwen35 dense), e2e.py vs llama.cpp | greedy 4/8 identical, **TF top-1 507/512 (99.0%)**, top-5 512/512, mean / max abs dlogp **0.0109 / 0.122**; same numbers on the window-3 binary | out/iq4xs_cmp.json, iq4xs3_* |
| Q8_0 control, same harness | 6/8, 510/512 (99.6%), 512/512, 0.0095 / 0.099 (= m4/iq3s's control) | out/q8ctl_cmp.json |
| 35B MTP=2 vs m6/out/h-off.json | **40/40** (w1 116.0, w3 111.7 tok/s) | m3/out/iq4xs-m2.json, iq4xs3-m2.json |
| 35B MTP-off (MTP-dir file, TITAN_MTP=0) vs m6/out/mtpfile-off.json | **40/40** (w1 88.0, w3 86.7 tok/s) | m3/out/iq4xs-mf0.json, iq4xs3-mf0.json |
| Bonsai-2 B2-off vs o-off; IQ2_M and Bonsai-27B Q1_0 8x300 vs the deployed binary's runs (integ3 i-new / q-new) | **8/8, 8/8, 8/8** (both windows) | out/b-off*.json, i-new*.json, q-new*.json |
| REDCELL G1 (pc 0) and gemma4-12b G1 vs the deployed binary's runs (integ3 rz-a-g1, g12-g1) | **top-20 / tokens / text 40/40** each (both windows) | out/rz-g1, rz3-g1, g12-g1, g12-3-g1 |

"Deployed binary" = bin/mistralrs-titan-swap at the start of this work = integ3 (sha 1de00581...), whose recorded outputs
are the references; the main session replaced swap with devmap2 at 19:45 (during window 3).

## OrcaSAQ-2 27B (qwen35 dense, 65 blocks, IQ4_XS 439 / Q5_K 65 / Q6_K 2 tensors, 15.68 GB)
- llama.cpp reference (`-ngl 99 -c 8192`, **F16 KV fits**: 14760 MiB used): G1 out/ref-orca-g1.json; prefill 1154-1167 tok/s
  at ~500 tokens, 1250 tok/s at ~3.9k tokens. (Its g5 runs stopped after 1 token, so no llama decode figure.)
- titan, auto device map (`--max-seq-len 65536`): layers 53-63 on CPU. Window 2: every request panicked in the CPU
  IQ4_XS vec_dot (`iquant.rs:349 index out of bounds: the len is 20 but the index is 20`): candle's CPU matmul sized
  the quantized lhs in 256-value blocks while IQ4_XS dots against 32-value Q8_1 (fixed, 98d4535f). Window 3: no panic,
  but the request then fails with `device mismatch in mul, lhs: Cuda, rhs: Cpu` in the qwen35 dense path - a CPU/GPU
  layer split of this model is not supported there (not in this task's scope; devmap / qwen35 loader).
- titan, all on GPU (`-n 0:64`):
  | | MTP 0 | MTP 1 |
  |---|---|---|
  | VRAM after load | **15690 MiB used / 244 free** | 15882 / 52 free (blk.64 loaded as draft head) |
  | G1 vs llama.cpp | **top-1 40/40, median abs dlogp 0.0000 (max 0.0003), top-20 KL max 0.00005** | identical to MTP 0 (40/40 top/tokens/text) |
  | prefill | 1167 / 1050 tok/s at 502 / 484 tokens (llama 1167 / 1154) | OOM already at ~500 tokens |
  | decode (8 x 300 greedy, chat prompts) | **29.5 tok/s** | requests fail (OOM): 6/8 empty, no MTP stats |
  | largest prompt served | ~500 tokens; 1.5k / 2.2k / 3k / 3.5k / 3.9k-token prompts OOM at the prompt step (recovered, no panic), also with TITAN_PREFILL_BIG_CHUNK=0 | none useful |
  Cause: titan keeps token_embd + output (Q6_K, 2 x 1.04 GB) on the GPU; llama.cpp keeps token_embd on the host and
  had 1.17 GB free at -c 8192. The chunked prefill reserves KV in 8192-token steps (KV_RESERVE_STEP), more than the
  ~240 MiB left.
- Draft roster entry (NOT applied; only usable for short prompts until token_embd can live on the host):
  ```toml
  # OrcaSAQ-2 27B Cyber Uncensored (qwen35 dense, IQ4_XS). All 64 layers on the GPU leave ~240 MiB: prompts up to
  # ~500 tokens; >= ~1.5k tokens OOM (recovered). MTP does not fit. Decode 29.5 tok/s. Needs bin with IQ4_XS dense.
  [[models]]
  name = "orcasaq2-27b"
  kind = "text"
  model_id = "~/ai/models/orcasaq2-cyber-27b"
  max_model_len = 1024
  [models.format]
  format = "gguf"
  quantized_file = "OrcaSAQ-2-27B-Uncensored.gguf"
  [models.device]
  max_seq_len = 1024
  device_layers = ["0:64"]
  [models.titan]
  idle_ttl_secs = 1800
  [models.titan.env]
  TITAN_MTP = "0"
  ```
  (`device_layers` as in the 541d3c36 roster; prefix_cache_n = 0 advisable.)

## Not done
- candle `cuda_iq4_xs` test written (mirrors cuda_iq3_s) but **not run**: candle-core is not a member of the mistral.rs
  workspace ("cannot be tested because it requires dev-dependencies"), and a separate candle cuda build was not
  attempted in a window. The kernel gates + 9B e2e cover the same paths.
- 27B: no usable context, no MTP acceptance figure (MTP=1 does not fit), no llama.cpp decode figure.

Windows: window1-20261003-1919.log (14m44s), window2-20261003-1940.log (1m33s), window3-20261003-1946.log (15m09s);
titan-mistral stopped after taking the lock and restarted on exit each time (active after each); titan-spark never started.
