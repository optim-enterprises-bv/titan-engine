# devmap / devmap2: automatic device map that tracks real VRAM, native qwen3 on flash-prefill (staged, NOT deployed)

Binary: `bin/mistralrs-titan-devmap2`, sha256 `32e23b8bc0ab93f34559e7298036f1ff20e0f05e4fbf9fd53575830859547edd`
(nvcc-free `cargo build --release -p mistralrs-cli --features oxide`, window 5). Roster to deploy: **`m4/devmap/models-deploy.toml`**
(= deploy/models.toml @ 6d9dc9b6 with every `device_layers` pin removed; qwen3-14b keeps `prefix_cache_n = 0`, reason below).
Not deployed; `deploy/models.toml`, `bin/mistralrs-titan-swap` untouched; nothing pushed.

## Branch heads (`devmap` in all three repos)
| repo | worktree | head |
|---|---|---|
| mistral.rs | top-devmap/mr-devmap | 6eaaf58f9 (from 86c8f605a) |
| titan-engine (candle + m4) | top-devmap (sparse) | candle dc2bf0bb + this README commit (from 541d3c36) |
| oxide-kernels | oxide-devmap (top-devmap/oxide-kernels -> ../oxide-devmap) | 85708b2 (from 64d8b5a) |

mistral.rs commits: bbdae0ad0 GGUF tensor inventory / Gemma 4 F16 KV / estimate log / TITAN_DEVMAP_LOG; 38b8a6b7d + c3e40cbae +
81e990176 prefill-activation estimate, half-device cap, no act reserve on the CPU, devmap_* unit tests; 546f1cb1c qwen3 flash-prefill
prompt attention + MLP passes, kv_cache set_none_cache / error propagation, d128 dispatch; 6eaaf58f9 PTX, MoE CPU f32 rows, qwen3
flash-path estimate.

## Root causes
1. **Weights** (devmap): native GGUF models were sized by the loader's F16 formula at ONE integer pack factor for the whole model
   (Q5_K/Q6_K -> 2, i.e. 8 bits/weight): qwen3-14b 12601 MiB of layers vs 8902 real. Now: bytes of every GGUF tensor the bindings
   read, once (fused tensors bound as slices, tied heads). Gemma 4's embedding was charged dequantized F32 (3840 MiB vs 787).
2. **KV dtype**: Gemma 4 under f32 keeps F16 KV; the map charged F32 (7392 vs 3696 MiB).
3. **Activations**: charged one 1024x1024 attention chunk (qwen3 80 MiB) or a full heads x S x S matrix (gemma4 7744 MiB) instead
   of what one unchunked prompt allocates. Now per model, measured (TITAN_DEVMAP_LOG pool peaks) with modest margins, capped at half
   the device; none reserved on the CPU fallback.
4. **qwen3 prompt path** (devmap2): eager attention with an S x S mask built through an F32 `Tensor::full` (7 B/element; 16k: ~1.8 GB
   transient) and 1024-row score chunks, MLP over the whole prompt, plus a prompt-sized "preallocated" KV template per layer held on
   the main GPU by every sequence (set_none_cache then zeros_like'd it: 1.5 x KV). Pinned all-GPU, a 16k prompt OOM'd.
   Now: CUDA layers run the cuda-oxide flash-prefill kernel (new head-dim-128 variant; 40 q heads zero-padded to 8 per KV head;
   4096-row launches), no mask; CPU layers get a device-built mask; MLP in 4096-token passes; the KV template is a 1-element broadcast
   and set_none_cache allocates on the layer's device. Measured transient beyond weights + KV: 2063 MiB at 16184 tokens.
5. **kv_cache:857 panic**: `zeros_like().unwrap()` (and Tensor::cat unwraps, to_device expects) on allocation paths; now `?` into
   the existing OOM recovery.
6. **CPU offload broken**: gemma4 f32 CPU layers: preallocated KV template F32 (main device) vs CPU append converting to F16 ->
   "dtype mismatch in slice-set"; fixed by sizing the cache per layer device/dtype. REDCELL bf16: candle CPU gemm has no bf16
   ("unsupported dtype BF16 for op matmul") -> f32 matmul; then the GGUF CPU indexed-MoE path takes f32 rows only ("moe experts
   forward") -> rows converted.

## Estimate vs measured (MiB)
| model | term | old | new | measured |
|---|---|---|---|---|
| qwen3-14b @16k | weights (layers + embd/head) | 12601 + ~1484 | 8902 + 1118 | 10143 pool after load |
| | KV | 2560 | 2560 | 160 KiB/token |
| | activations | 80 | 2288 | 2063 at 16184 tok (flash path) |
| gemma4-12b f32 @11264 | weights | 5940 + 3840 | 5850 + 787 | 6710 |
| | KV | 7392 (F32) | 3696 (F16) | 336 KiB/token |
| | activations | 7744 | 4379 | 1852 at 8k |
| redcell-26b @8192 | weights | 11680 | 11076 + 577 | 11703 |
| | activations | 2048 | 1444 | 1289 at 7998 |

## Gates (logs: window4-6*.log, out/)
| gate | result |
|---|---|
| flash-prefill kernels (window 4, out/gate-fp.log) | **PASS**: d128 bf16 worst 0.592 of tol, d128 f16 o32 0.074, splits within 0.667, mutations 11/11 each; every existing kernel vs the committed PTX: 416 calls, **0 of 883556352 outputs differ** |
| qwen3-14b, no pin, 16384, prefix cache off (window 6 fq) | **40/40 layers on the GPU**; 16184-token prompt OK (nvidia-smi peak 15814 MiB, 1745 tok/s prefill); short decode **48.2 / 48.5 tok/s** |
| qwen3 G1 vs llama.cpp (F16 KV, ngl 99) | **top-1 40/40, median dlp 0.0000, KL max 0.00024** |
| qwen3 G3 ~4k vs llama.cpp spread | **WITHIN** (4/4, dlp 0.0417, KL -0.0009) |
| qwen3 G3 ~8k (g3l) | **OUTSIDE** the spread (top-1 3/4 on a near tie: titan ':' -1.406 vs ':\n' -1.531; dlp 0.0353 vs llama worst 0.0259); absolute PASS. The deployed (eager) binary on the same prompts: G3 OUTSIDE, G3l 3/4, dlp 0.0648, KL 0.0068 (absolute FAIL): the new path is closer to llama.cpp; the remaining gap is the F16 residual stream (llama.cpp runs F32 activations) |
| kv_cache:857 (window 5 p857, v3 binary) | provoked: 8k then 16k, 40 layers, prefix cache off -> **panicked at kv_cache/mod.rs:857**, engine dead, next request 500 |
| same, new binary (window 4 pnew, prefix cache on) | 16k and 36k fail **cleanly** (`titan: CUDA out of memory: the request failed ... pools trimmed`), the next request is served, unit active |
| gemma4 f32, 4 CPU layers (-n 0:44) G1 vs all-GPU | **40/40, dlp 0.0002, KL max 0.00135** (deployed binary: warm-up fails, dtype mismatch) |
| REDCELL bf16, 3 CPU layers (-n 0:27) G1 vs all-GPU | **39/40, dlp 0.0021, KL mean 0.0006 max 0.0064** |
| gemma4 / spark / REDCELL G1 vs deployed binary | **top / tokens / text identical 40/40** each |
| G1 near-max, unpinned (window 5) | gemma4 f32 11194 tok, bf16 12146, REDCELL 8134, spark 16160: all GPU, no OOM |
| REGRESSION CORE (window 5, this binary) | 35B MTP=2 **40/40**, MTP-off **40/40**, Bonsai-2 B2-off **8/8**, IQ2_M **8/8**, Bonsai-27B Q1_0 **8/8** |
| quantized-path mappings | dry run: tiered fractions and Layers lines identical to the integ3 dry run |
| unit tests | 4/4 `devmap_*` |
| dry run, models-nopins2.toml (no pins, qwen3 prefix cache default) | **12/12**; qwen3-14b 48.5 tok/s, 10620 MiB; table in window6 log |
| qwen3-14b @65536 | maps 0-12 GPU / 13-39 CPU; short prompt 4.4 tok/s; **8k prompt did not finish in ~20 min**: CPU-layer prefill (27 layers) is the limit, a ~30k prompt is not servable in practice. NOT MET |
| DeepSeek-R1-Qwen3-8B @65536 (second case) | not loadable: "Standalone Qwen3 config cannot reconstruct this GGUF's RoPE scaling" (unrelated loader limit). NOT RUN |

## qwen3 prefix cache
With the default prefix cache the 16k prompt fails at the first completion step (clean OOM): the kept sequence's KV stays with
the prefix cache while the decode step copies it (2 x 2.5 GB). models-deploy.toml therefore keeps `prefix_cache_n = 0` for
qwen3-14b; short prompts with the cache on are fine (dry run).
