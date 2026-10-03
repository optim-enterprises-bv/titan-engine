# emb: one row-only quantized embedding lookup, and the i-quant CPU dequantize fixes (staged, NOT deployed)

Binary: `m4/emb/mistralrs-emb`, sha256 `3d56f35be915eca2a1b91ccfca53da1df6b9e40a0ba38a64517ac84019a4b62d`.
It is an nvcc-free `cargo build --release -p mistralrs-cli --features oxide` with CARGO_TARGET_DIR
`target-emb-oxide` and TITAN_OXIDE_DIR `oxide-integ` (kernels unchanged: `top-emb/oxide-kernels -> ../oxide-integ`).
Nothing was deployed or pushed. `bin/mistralrs-titan-integ`, `bin/mistralrs-titan-swap` and `deploy/models.toml` are untouched.

## Commits (branch `emb`, based on `integ-20261002`)
| repo | commit | what |
|---|---|---|
| titan-engine (candle), `top-emb` | 3f389945 | IQ3_XXS: the scale word is at `qs[64 + 4*ib]`, not `qs[8*ib + 64]` (was an out-of-bounds panic for ib >= 4) |
| | 6a386409 | IQ2_XS: rewritten from `dequantize_row_iq2_xs`. It had the wrong scale formula, read two 9-bit indices per u16 and applied no signs. |
| | b307cad0 | `#[cfg(feature = "cuda")]` on the cudarc `DeviceRepr` / `ValidAsZeroBits` impls (BlockIQ4xs, BlockIQ2xs) |
| | 2dfb7483 | IQ2_XXS (**found by the audit**): it indexed the `[u8; 64]` `qs` as if it held u16 units, so every group read the wrong bytes |
| | 990abe9a | IQ4_XS (**found by the audit**): it interleaved the nibbles. ggml puts the low nibbles in values 0-15 of a group and the high nibbles in 16-31. |
| | e55afc23 | `QTensor::embedding` dequantizes only the looked-up rows, on CPU and CUDA |
| mistral.rs, `top-emb/mr-emb` | a763a3f1e | gemma4's CUDA-only copy (`qtensor_embedding_rows`, from 810d78900) is removed. `quantized_qwen35_moe::QEmbedding` (also used by gpt-oss) now calls `QTensor::embedding`. |

## Task 1: row-only embedding lookup
**Where it lives.** Every GGUF embedding that is backed by a QMatMul goes through candle `QTensor::embedding`. The call
path is `GgufMatMul::embedding_forward_raw -> QMatMul::embedding -> QTensor::embedding`, and it covers Spark, gemma4
(`embed_tokens`, and the PLE table when it is quantized), REDCELL and every native-path GGUF model. The old code
dequantized the whole table to F32 and then ran index_select on every call. The new code:
- **CPU tables:** the new trait method `QuantizedType::dequantize_rows` runs the dtype's `to_float` on each looked-up
  row's blocks. This is the same routine that `dequantize` runs over the whole table.
- **CUDA tables:** the new `QCudaStorage::gather_rows` copies the rows' block bytes on the device. It uses candle's
  `is_u32_u8` index-select kernel, which is in the oxide PTX, so there is no host sync and no change to the kernels.
  The bytes land in a zero-padded buffer laid out like `zeros()`, and then the usual `dequantize` runs on it. That is
  the same kernel, or, for dtypes whose CUDA dequantize runs on the host (PTQ1_0, IQ2_XXS/XS/S, IQ3_XXS, IQ4_XS, Q8_1,
  F16/BF16/F32), the same host routine.
- **Fallback to full dequantization:** only Metal tables, and CUDA tables of 4 GiB or more (the index-select kernel uses
  32-bit offsets; no titan model is near that). The fallback is decided by device and size. Every dtype can be
  row-sliced, because `QTensor::new` requires whole blocks per row and every format dequantizes block by block.
- **The titan qwen35 path** (and gpt-oss) does not use `QMatMul`. It uses its own host `QEmbedding`, which already
  gathered rows and called `dequantize` on the CPU. It now holds the CPU `QTensor` and calls `QTensor::embedding`.
  The rows are the same bits (35B gates below), there is one implementation, and the load keeps one host copy of the
  table instead of two.
- **gemma4:** the special case in `embed_tokens` and `mistralrs_quant::qtensor_embedding_rows` are deleted. That copy
  staged the gathered rows through the host, which cost a sync per call.

### Gates (window 1, `window1-20261002-2132.log`, 16m41s; titan-spark stopped after taking the lock and restarted on exit)
| gate | result | log |
|---|---|---|
| (a) row-only == whole-table dequantize + index_select, bit for bit, all 26 GgmlDTypes. Tables were 257 rows of random bytes; ids were 3 x 41 (ends, a consecutive run, duplicates, random). Each dtype ran on CPU, CUDA, and CUDA with host ids, with a wide row (1024) and a narrow one (96/192/384/512: ragged 256-value groups for the 32/64-value formats). | **PASS**, 0 differing in every case. The negative control (reference shifted by one row) differs every time. Empty ids give [1,0,1024]. Q8_1 is not comparable: its `to_float` panics ("no support for vec-dot on Q8_1") in the whole-table path as well, same as before. | out/embcheck.log |
| (b) 35B MTP=2 vs m6/out/h-off.json | **40/40**, 116.7 tok/s (integ 116.1) | window log, m3/out/emb-m2.json |
| (b) 35B MTP-off, MTP-dir Q4_K_XL, TITAN_MTP=0 vs m6/out/mtpfile-off.json | **40/40**, 87.0 tok/s (integ 87.0) | m3/out/emb-mf0.json |
| (b) Bonsai-2 B2-off vs m4/bonsai2/out/o-off.json | **8/8**, 63.7 tok/s | out/b-off.json |
| (b) IQ2_M new vs bin/mistralrs-titan-integ (its recorded run m4/integ/out/i-new.json), 8x300 | **8/8**, 57.6 tok/s (integ 58.0) | out/i-new.json |
| (b) Bonsai-27B Q1_0 new vs integ (m4/integ/out/q-new.json), 8x300 | **8/8**, 60.5 tok/s (integ 61.1) | out/q-new.json |
| (c) Spark Q4_K_M G2 vs integ's sp-g2, same request order (g5 x3, g3, g1, g2) | **tokens and text identical 40/40** | out/sp-g2.json |
| (c) Spark G1 vs integ / vs llama.cpp refc-spark-g1 | top-20 identical to integ 40/40. Against llama.cpp: 39/40, median dlp 0.0124, KL median 0.00371, the same as integ. | out/sp-g1.json |
| (c) Spark G3 vs integ | top identical 4/4 | out/sp-g3.json |
| (d) gemma4-12b G1 (8192, 0:48, bf16) vs integ / vs llama.cpp | **40/40 top-20 identical to integ**. Against llama.cpp: 40/40, dlp 0.0030, KL 0.00060. | out/g12-g1.json |
| (e) REDCELL G1, prefix-cache-n 0, vs llama.cpp ref-red-g1 | 39/40, median dlp 0.0084, KL median 0.00104 (integ 39/40, 0.00091; branch 0.00068). First tokens equal integ 40/40, top-20 34/40, which is within its known run-to-run spread (30-36/40 on one binary). | out/red-g1.json |

### Throughput, decode tok/s (same window, same requests, new binary then bin/mistralrs-titan-integ)
| model / prompt | integ (before) | emb (after) |
|---|---|---|
| Spark-X2.5 Q4_K_M, ~550 tokens (g5 2000) | 95.7 / 93.3 | **122.8 / 127.1** |
| Spark, ~4.1k (g5 15000) | 96.1 / 104.3 | **123.1 / 133.3** |
| Spark, ~9k (g5 33000: 8943 / 9291 tokens) | 91.7 / 91.2 | **117.8 / 119.4** |
| Spark G2 (40 x 256 tokens, short prompts), median | not run on integ (llama.cpp's recorded G2 run: 99.8) | 139.5 |
| gemma4-12b, ~540 tokens | 61.9 / 62.2 | 61.5 / 62.3 |
| gemma4-12b, ~7.3k | 51.9 / 53.2 | 38.1 / 53.3 |

- **Spark** decode is up about 30% at every depth: +31% short, +29% at ~9k. That matches removing 2.3 ms from a token
  of about 10 ms. Prefill is unchanged (5.7-6.4k tok/s).
- **Spark against llama.cpp:** it is still below the 145-165 tok/s llama.cpp reference. Short prompts reach 123-127
  through the server and 139.5 in G2.
- **gemma4** is unchanged. Integ already carried 810d78900's row gather for gemma4, so this commit only moves that fix.
  The ~7.3k first request on the new binary (38.1) is a single outlier; the second (53.3) matches integ.
  VRAM after G5 was 12.4 GB on the new binary and 13.9 GB on integ.
- **Microbenchmark** (embcheck, Q6_K, synchronised):

  | table | ids | row-only | whole table |
  |---|---|---|---|
  | Spark 131072x2560, CUDA | 1 | 0.024 ms | 2.305 ms |
  | Spark 131072x2560, CUDA | 512 | 1.23 ms | 3.38 ms |
  | gemma4 262144x3840, CUDA | 1 | 0.024 ms | 7.76 ms (and the 3.75 GiB transient) |
  | Spark 131072x2560, CPU | 1 | 0.002 ms | 383 ms |
  | gemma4 262144x3840, CPU | 1 | 0.003 ms | 1148 ms |

## Task 2: i-quant CPU dequantize
**Method.** `cpucheck/ggml_deq.c` links llama.cpp's own `libggml-base` and dumps random blocks together with
`ggml_get_type_traits(t)->to_float` (dequantize_row_*): 1,048,576 values per type, xorshift seeded by the type id.
- llama.cpp is acecd56 at `~/ai/llama.cpp/build`. PTQ1_0 (type 143) comes from `ref/sudoingx-bonsai2/build`.
- `cpucheck/iqcheck.rs` compiles candle's **own** `iquant.rs` through `#[path]` (not a copy). Stand-ins replace its two
  imports. It compares every `to_float` bit for bit, counting NaN-in-both as equal (from random f16 scales; the counts
  are listed).
- `build.sh stub` links a cudarc stand-in so that the unfixed file builds. `build.sh plain` builds without cudarc.

| type | before (`before-iqcheck.log`) | after (`after-iqcheck.log`, and embcheck on the real candle-core) |
|---|---|---|
| IQ2_XXS | 997021 of 1048576 differ | 0 |
| IQ2_XS | 1017501 differ | 0 |
| IQ3_XXS | panic: index 96 out of bounds | 0 |
| IQ3_S | 0 | 0 |
| IQ2_S | 0 | 0 |
| IQ4_XS | 887280 differ | 0 |
| IQ4_NL (iq4_nl.rs) | (not in iquant.rs) | 0 (embcheck) |

- **Bug 3:** before the fix, `build.sh plain` fails with `E0433 cannot find module or crate cudarc` at iquant.rs:258-315.
  After it, the plain build works, and so does a plain CPU `cargo check` of candle-core (`cpuonly/`, out/cbuild-cpuonly.log).
- **Routines Task 1 uses.** embcheck checked every dtype's CPU `dequantize` against llama.cpp. That is the exact routine
  the CPU row lookup and the CUDA host-side fallback call. **All PASS** with 0 differing: F16, BF16, Q4_0, Q4_1, Q5_0,
  Q5_1, Q8_0, Q2_K-Q6_K, Q1_0, IQ4_NL, MXFP4, NVFP4, PTQ1_0, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ2_S, IQ3_S, IQ4_XS.
- **Not checked against llama.cpp:** F32 (a plain copy) and Q8_K (llama.cpp acecd56 exports no `to_float` for either).
  Q8_1 has neither a llama.cpp `to_float` nor a candle `to_float`; candle panics.
- **Gemma4 / REDCELL / Spark / 35B / Bonsai** have Q6_K, Q8_0, Q5_K or Q1_0 token_embd and no i-quant tables. The
  i-quant fixes therefore do not change any gate above. They matter for i-quant token_embd files (for example the
  IQ3_S-embedding 9B) and for the CPU fallback dequantize.

## Found, not fixed (outside this task)
- **CUDA dequantize kernels differ from llama.cpp's CPU dequantize** on random blocks (embcheck, informational):
  - Q4_0: 31835 of 1048576 differ.
  - MXFP4: 8611 differ.
  - NVFP4: 524730 differ, about half; this looks like a convention difference in the scale and deserves a look.
  - Every other CUDA kernel matches exactly.
  - This does not affect Task 1's identity: rows and whole table use the same kernel. It does mean a CPU table and a
    GPU table of those three dtypes give different embeddings.
- **Load-time dequantization:** `quantized_llama` and `quantized_qwen3_moe` still dequantize the embedding table once
  at load into a dense `Embedding`. That is a one-off and not per-forward, but it keeps an F32 table resident.
- **Two warnings** (`unused import half::f16` in mxfp4.rs and nvfp4.rs) appear in the CPU-only build. They were there
  before and those files are not touched.

## Files
- `window1.sh`, `lib.sh`: the window. `cpucheck/`: the C reference dumper and the iquant checker. `ref/`: the
  reference dumps. `embcheck/`: gate (a). `cpuonly/`: the plain CPU check. `out/`: gate outputs and logs.
- Windows used: 1 of 3.
