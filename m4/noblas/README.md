# noblas: no cuBLAS / cuBLASLt / cuRAND in the titan binary

The `noblas` branch exists in three repos. It is **not deployed and not pushed**.

| repo | worktree | head | base |
|---|---|---|---|
| mistral.rs | `top-noblas/mr-noblas` | 3c1a54107 | model-swap ddfe94d8f |
| titan-engine (candle) | `top-noblas` (sparse checkout: candle, m4/noblas) | 48b983ce + this README | master 1790c6fe |
| oxide-kernels | `oxide-noblas` | d2e7cd1 | glu-typed-kernels e5949f1 |

**Staged binary:** `bin/mistralrs-titan-noblas`, sha256 `9a7d7d66ed0f79b37f4697834a6bfe9a046536ed0466698c1925dc3ebc90ef9f`. It is a nvcc-free `--features oxide` build of mr 3c1a54107 plus candle 48b983ce. The embedded PTX is byte-identical to oxide d2e7cd1's `gemm.ptx`.

## Verdict

- **Unlinking is done.**
  - `ldd` lists only libcuda (the driver), libc, libm, libgcc_s, libpthread, libdl and librt.
  - `nm -D` shows 0 cuBLAS or cuRAND imports. The deployed binary has 26.
  - `strings` finds 0 `libcublas*` / `libcurand*` names and 0 cuBLAS/cuRAND API names. The deployed binary has 3 and 47.
  - The only remaining "cublaslt" strings are Rust paths of the `mistralrs_quant::cublaslt` module, which now runs the oxide kernels.
  - One nsys trace (cuda + cublas) covered a swap server running all 13 roster models: 480,190 kernels, no cuBLAS NVTX ranges (the trace has no NVTX table at all), and 0 kernels with cuBLAS names.
- **Speed is unchanged.**
  - Prefill and decode are within ±3% for every model (table below), apart from noise on the MTP / tiered models.
  - The cuBLAS calls were 0.2-6.5% of GPU time in the deployed binary.
- **Accuracy: the gates fail on near-tie flips.** Nearly every output changes, because the summation order differs from cuBLAS. Against llama.cpp:
  - **35B G3** has top-1 3/4. The flip is a 0.16-nat tie in llama.cpp. Its dlp 0.0015 and KL 0.0004 are far inside the spread.
  - **REDCELL G1** is 37/40. The deployed binary gets 39/40.
  - **IQ2_M G1** is 36/40 and **MXFP4 G1** is 37/40. The deployed binary already fails both, at 38/40.
  - **35B G1 dlp** is 0.069. The deployed binary already fails it, at 0.056.
  - The KL and dlp distributions of new and deployed have the same size, and some models improved (qwen3-next 38→39, gpt-oss-120b 39→40).
  - Per the brief, **the 35B baselines were not re-frozen**: G1/G3 did not pass.

## 1. Inventory (phase 1, window 1)

Method:
- `TITAN_GEMM_LOG` logging in candle's matmul, the cuBLASLt wrapper and cuRAND (commits ea37ff86 / 39f1e51db).
- An nsys cuda+cublas trace of the deployed roster block of each model (CUDA graphs off).
- Requests: a warm-up, a short chat (32 tokens) and a ~4k raw prompt (16 tokens).
- Outputs: `out/inv-*.glog`, `out/inventory-raw.txt`, and `out/shapes-all.txt` (621 distinct problems with counts).

GPU share was measured from nsys traces of the deployed binary with the deployed config, 100 ms flush (`out/tim-*.sqlite`, `nsysshare.py`):

| model | cuBLAS share of GPU time | main shapes |
|---|---|---|
| gpt-oss-20b | 6.45% | f32 GEMV (m=1): q/k/v/o projections, 201088x2880 lm_head; prefill f32 GEMMs m x {512,4096,2880} |
| bonsai2-27b | 5.55% | f32 1x48x5120 GEMV; bf16 decode-attention GEMVs (cuBLASLt, K broadcast over 6 heads) |
| qwen3.6-35b (and iq2m / mxfp4) | 2.41% | f32 router GEMV 1x256x2048, 1x1x2048 shared gate, 1x32x2048 GDN; prefill m x {32,256,1} x 2048; bf16 decode attention (kv < 1024) |
| redcell-26b | 2.17% | bf16 router 1x128x2816; eager attention batches |
| qwen3-next-80b | 1.19% | f32 1x512x2048, 1x1x2048; prefill 2048x512x2048 |
| orcasaq2 | 0.79% | bf16 decode attention; eager prefill attention b24 |
| gpt-oss-120b | 0.18% | f32 router 1x128x2880 |
| gemma4-12b | 0 at run time | 2 f32 k=1 products at load |
| qwen3-14b, spark | small | eager attention for prompts under 1024 keys |

cuRAND is created at device init and on `set_seed`. No roster model draws from it: there are no `R uniform` / `R normal` lines. It is only reachable through diffusion and vision code.

## 2. Kernels (oxide-kernels/gemm)

All kernels use one problem form: `D = alpha A B + beta C + bias`, with any strides and batch in grid z.

**Kernel families:**
- **tensor cores** (f16 / bf16): mma.sync m16n8k16 with f32 accumulation, BM {128, 64, 32} x 128 x 32, 3-stage cp.async, swizzled ldmatrix (.trans for m- or n-contiguous operands).
- **SIMT**: any type and any layout, f32 FFMA, 128 or 64 tiles, register-staged double buffering.
- **GEMV** (m <= 8): one warp per column, or one block per column with split k; 16-byte loads.
- **split-K** with a deterministic in-order merge.
- **Philox4x32-10** uniform / normal (f32 / f64).

**Planner:** `src/plan.rs`, copied verbatim into candle. It folds broadcast batches into rows (the GQA decode calls) and transposes skinny problems into a GEMV.

**Gate** (`out/gate-w3-4.log`; it matches the final PTX): **PASS**.
- 2695 launches and 13.0M outputs were compared with an f64 reference. Tolerance: `r_out|ref| + (k/4+16) 2^-24 S + epilogue`.
  - Worst error/tolerance per family: tc bf16 0.995, tc f16 0.987, simt bf16 0.995, simt f16 0.989, simt f32 0.151, gemv bf16 0.984, gemv f16 0.987, gemv f32 0.049.
  - The bf16/f16 ratios near 1 are output rounding at half an ulp.
- Coverage: in-place C, strided and column-major D (canary checked), misaligned bases, broadcast A/B, bias by row or column, forced split-K.
- Mutations: 45/45 caught (k−1 or a dropped chunk, a wrong batch stride, alpha×1.01, the opposite layout kernel, a dropped bias, a missing split partial).
- Philox: bit-identical to a host replica, counter continuity, seed sensitivity, moments.
- All 115 entries were launched.
- The gate also caught a bug: the unrolling rewrite had cut Philox to 8 rounds. That was fixed in d2e7cd1.

**Speed vs cuBLAS** (`out/bench-w3-2.log`, every inventoried shape, best of 3, cublasGemmStridedBatchedEx 32F). These are count-weighted per-model totals over the inventory runs (oxide / cuBLAS):

| model | ratio | model | ratio |
|---|---|---|---|
| bonsai-27b | 0.68 | qwen3.6-35b | 0.84 |
| bonsai2-27b | 0.80 | qwen3.6-35b-iq2m | 0.86 |
| gpt-oss-120b | 1.25 | qwen3.6-35b-mxfp4 | 0.86 |
| gpt-oss-20b | 0.93 | qwen3-14b | 0.68 |
| orcasaq2 | 0.87 | qwen3-next-80b | 1.04 |
| redcell-26b | 0.72 | spark-x2.5 | 0.88 |

The shapes that carry time:

| shape | oxide / cuBLAS | note |
|---|---|---|
| gpt-oss-20b lm_head 1x201088x2880 f32 | 2677 / 3200 µs | 0.84 |
| decode GEMVs (all routers, gates, GQA attention) | 0.3-0.9x | |
| gpt-oss-20b 3335x4096x2880 and 3335x2880x4096 f32 | 1.00 and 1.08x | |

**Slower than 15%** (prefill SIMT f32 at mid sizes):

| shape | oxide / cuBLAS | ratio |
|---|---|---|
| 35B 2048x256x2048 (x40 per 2k chunk) | 187 / 129 µs | 1.44 |
| 2048x32x2048 | 64 / 37 µs | 1.73 |
| 217x256x2048 | 37 / 20 µs | 1.86 |
| qwen3-next 2048x512x2048 | 369 / 248 µs | 1.49 |
| gpt-oss-20b 77x4096x2880 | 360 / 180 µs | 2.00 |
| gpt-oss-20b 77x2880x4096 | 289 / 173 µs | 1.67 |
| gpt-oss-120b 3335x128x2880 | 280 / 174 µs | 1.61 |
| gpt-oss-20b 1x2880x4096 GEMV | 19.1 / 16.6 µs | 1.15 |

**Slower than 15%** (bf16, eager prefill attention):

| shape | oxide / cuBLAS | ratio |
|---|---|---|
| orcasaq2 b24 512x256x512 tensor cores | 94 / 47 µs | 2.0 |
| P·V with k not a multiple of 8 (SIMT) | ~7 / 3 µs | 2.4 |

- None of these shows up in model prefill: G3/G5 prefill tok/s stays within ±1% (table below).
- The worst case in absolute terms is gpt-oss-20b on a 77-token prompt: about 4 ms more per prompt.
- Every shape is listed in `out/bench-w3-2.log`.

## 3. Wiring and unlinking (phase 2)

- **candle-core:**
  - New `cublas` feature: cudarc `cublas` / `cublaslt` / `curand` are on only with it, and the workspace cudarc no longer enables them.
  - Without it, `matmul` and `rand_uniform` / `rand_normal` run `cuda_backend/oxide_gemm.rs`: the PTX is embedded and loaded through `get_or_load_custom_func`.
  - The Philox counter continues across fills, and `set_seed` restarts it.
  - f64 matmul is an error without cuBLAS. No roster model uses it.
- **mistral.rs:**
  - `mistralrs-quant/cublas`, `mistralrs-core/cublas`, and the cli's `cuda` pull cuBLAS. **`oxide` no longer implies the cli's `cuda`.** The nvcc build (`--features cuda`) is unchanged.
  - Without `cublas`, `cublaslt/oxide.rs` implements the cuBLASLt wrapper on the oxide GEMM. It keeps the same semantics (alpha, beta*C on a copy of `out`, bias), so the attention and unquantized call sites are unchanged.
  - GPTQ's hgemm now uses the oxide GEMM (f16).
- **Fail loudly in the oxide build** (non-roster):
  - the cuBLASLt FP8 batch matmul (fp8 / blockwise / vector-fp8 safetensors)
  - the cuBLASLt Relu/Gelu epilogue
  - f64 matmul

## 4. Model gates (phase 2)

Setup: one from-config server per binary with `deploy/models.toml` (port 18690), `gates_m.py`. Outputs are `out/{new,old}-MODEL-{g1,g3,g5}.json`, scored by `newold.py`, `tm1.sh` and `tm2.sh` (token ids mapped to pieces, with control tokens printed as llama.cpp prints them: `sp.py`).

llama.cpp references:
- Existing ones: spark, qwen3-14b, redcell, orca, gemma4.
- New clean ones (f16 KV, temp 0, no LoRA, MoE experts on the CPU where needed): `out/lref-*` on `plain40.json` for the other 8.
- Spreads: `out/sp-*-{ub256,ub128,b512}-{g1,g3}.json`, scored by `m4/g4acc/g3spread.py`.

**New vs deployed** (first-token top-20): "same" = top-20 identical, then top-1 agreement and KL median.

| model | G1 same / top-1 / KL med | G3 same / top-1 / KL med |
|---|---|---|
| gemma4-12b | 40/40 identical | 4/4 identical |
| qwen3-14b | 0/40, 40/40, 0.00000 | 4/4 identical |
| orcasaq2 | 3/40, 40/40, 0.00000 | 4/4 identical |
| bonsai-27b | 14/40, 40/40, 0.00000 | 4/4 identical |
| spark-x2.5 | 1/40, 39/40, 0.0017 | 4/4 identical |
| bonsai2-27b | 0/40, 40/40, 0.0001 | 0/4, 4/4, -0.0001 |
| redcell-26b | 0/40, 38/40, 0.0003 | 2/4, 4/4, 0.0027 |
| qwen3.6-35b | 0/40, 40/40, 0.0027 | 0/4, 3/4, 0.0003 |
| qwen3.6-35b-iq2m | 0/40, 36/40, 0.0077 | 0/4, 4/4, 0.0004 |
| qwen3.6-35b-mxfp4 | 0/40, 39/40, 0.0017 | 0/4, 4/4, -0.0001 |
| gpt-oss-20b | 0/40, 40/40, 0.0002 | (deployed run missing) |
| qwen3-next-80b | 0/40, 39/40, 0.0033 | 0/4, 4/4, -0.0035 |
| gpt-oss-120b | 0/40, 39/40, 0.0006 | not run (48 tok/s prefill) |

**G1 vs llama.cpp** (thresholds: top-1 >= 39/40, median |dlogprob| < 0.05, KL median < 0.01). Each entry is "new | deployed":

| model | top-1 | dlp | KL | verdict new |
|---|---|---|---|---|
| spark-x2.5 | 39 \| 40 | 0.011 \| 0.012 | 0.0037 \| 0.0035 | PASS |
| qwen3-14b | 40 \| 40 | 0.000 \| 0.000 | 0 \| 0 | PASS |
| orcasaq2 | 40 \| 40 | 0.000 \| 0.000 | 0 \| 0 | PASS |
| gemma4-12b | identical to deployed | | | as deployed |
| redcell-26b | **37** \| 39 | 0.010 \| 0.008 | 0.0009 \| 0.0008 | **FAIL** (top-1) |
| qwen3.6-35b | 39 \| 39 | **0.069** \| 0.056 | 0.0074 \| 0.0053 | FAIL (dlp; deployed fails too) |
| qwen3.6-35b-iq2m | **36** \| 38 | 0.063 \| 0.068 | 0.011 \| 0.013 | FAIL (deployed fails too) |
| qwen3.6-35b-mxfp4 | **37** \| 38 | 0.049 \| 0.031 | 0.0027 \| 0.0021 | FAIL (deployed fails too) |
| bonsai-27b | 40 \| 40 | 0.014 \| 0.014 | 0.0017 \| 0.0017 | PASS |
| bonsai2-27b | 40 \| 40 | 0.004 \| 0.006 | 0.0008 \| 0.0010 | PASS |
| gpt-oss-20b | 39 \| 39 | 0.020 \| 0.020 | 0.0000 \| 0.0003 | PASS |
| qwen3-next-80b | 39 \| 38 | 0.029 \| 0.035 | 0.0028 \| 0.0034 | PASS (deployed fails) |
| gpt-oss-120b | 40 \| 39 | 0.015 \| 0.010 | 0.0008 \| 0.0006 | PASS |

Every flipped prompt is a near-tie in llama.cpp. For example, redcell #17 'In' -0.664 vs '###' -0.868 and #22 'To' -0.742 vs 'In' -1.145. llama.cpp's own -ub / -b variants leave G1 unchanged, because these short prompts fit in one ubatch, so G1 has no spread to compare against.

**G3** (RULES-agents spread rule):

| model | result |
|---|---|
| spark, qwen3-14b, gemma4, orca, bonsai-27b | G3 output identical to the deployed binary, so the earlier verdicts stand |
| redcell | 4/4, dlp 0.065, KL 0.019: within llama.cpp's spread (deployed 0.067 / 0.022) |
| iq2m | 4/4, 0.0025 / 0.0008: PASS |
| mxfp4 | 4/4, 0.0012: PASS |
| gpt-oss-20b (6000 chars: ~3.3k prompts OOM in both binaries) | 4/4, 0.068 / 0.023: within its spread (worst 0.198 / 0.046) |
| bonsai2 | 4/4, 0.0009 / 0.0005: absolute PASS (llama's own spread is narrower, 0.0014) |
| **qwen3.6-35b** | **top-1 3/4**, dlp 0.0015, KL 0.0004. **Outside the spread on top-1 only**: the variants keep 4/4. Prompt 3 is '' -1.120 vs '\n\n' -1.278 in llama.cpp; new has '\n\n' -1.222 vs '' -1.319 |

- The iq2m / mxfp4 spread rows contain an inflated variant (dlp 5.4 on one control-token prompt), so use the absolute numbers for those two.
- qwen3-next and gpt-oss-120b G3 were not compared with llama.cpp: CPU-expert llama.cpp runs at 2.5 tok/s prefill.

**35B 40 x 256 greedy:**
- MTP=2 vs `m6/out/h-off.json`: 1/40 identical. MTP-off file vs `mtpfile-off.json`: 1/40.
- Common prefix median 339 characters (min 57).
- New MTP=2 vs new MTP-off: 40/40, so MTP stays lossless.
- `m3/out/noblas-m2.json` and `noblas-mf0.json` are **not** frozen as baselines (G1/G3 did not pass).

**Swap-roster dry run:**
- All 13 models loaded and served G1/G3/G5 through one deployed-semantics server per binary (paged attention on for Spark and qwen3-14b).
- Bursts on the noblas server: Spark 4/4 (7.6-9.8k prompts, peak 13.6 GB) and qwen3-14b 4/4 (6.5-9.2k prompts, peak 14.8 GB).
- The same 13 models also ran under nsys.

**Speed** (noblas / deployed, same swap server conditions):

| model | prefill G5 ~2k | prefill G3 ~4k | decode (2 x 128, chat) |
|---|---|---|---|
| qwen3.6-35b | 2319 / 2249 (+3.1%) | 2321 / 2281 (+1.8%) | 135/106 vs 93/110 (MTP noise) |
| spark-x2.5 | 6665 / 6625 | 6304 / 6219 | 157.2 / 157.3 |
| qwen3-14b | 2559 / 2563 | 2396 / 2385 | 52.0 / 52.1 |
| gemma4-12b | 3123 / 3156 (-1.0%) | 3040 / 3045 | 77.3 / 77.3 |
| redcell-26b | 4954 / 4954 | 5052 / 5077 | 137.1 / 136.8 |
| orcasaq2 | 1080 / 1090 (-0.8%) | 1086 / 1094 | 29.5 / 29.5 |
| bonsai-27b | 1201 / 1208 | 1205 / 1201 | 62.0 / 61.8 |
| bonsai2-27b | 1109 / 1119 (-0.9%) | 1105 / 1114 | 83.8 / 81.3 |
| qwen3.6-35b-iq2m | 101 / 101 | 101 / 102 | 58.6 / 58.5 |
| qwen3.6-35b-mxfp4 | 502 / 449 | 504 / 504 | 103.1 / 102.3 |
| gpt-oss-20b | n/a | n/a | 84.4 / 82.1 |
| qwen3-next-80b | 158 / 154 | 148 / 149 | 29.1 / 29.8 (-2.4%) |
| gpt-oss-120b | 44.8 / 44.7 | n/a | 6.3 / 5.8 |

## 5. Left on NVIDIA libraries

Only `libcuda.so.1` remains: the driver API, including the PTX JIT. NVRTC is not linked (0 imports).

## Logs

- Windows: `window1..6-*.log`.
- Kernel gate and bench: `out/gate-w3-*.log` and `out/bench-w3-*.log` (round 2 = final kernels; round 3 = a rejected launch-bounds trial).
- Builds: `out/build-noblas-w*.log`.
- 35B runs: `m3/out/noblas-*.json`.

## Round 2: REDCELL (windows 7-8) -> `bin/mistralrs-titan-noblas2`

**Staged binary:** `bin/mistralrs-titan-noblas2`, sha256 `7dcba9fb446953852398a59c8265163e0c2035634bd24a1fd134ab7bd3a2890a`. It is built from mr e623480f0, candle af7c5e99 and oxide bd9992d. The embedded PTX is byte-identical to the crate's gated `gemm.ptx`.

**Localisation** (`TITAN_GEMM_REF`, window 7):
- New f64-accumulating reference kernels `gemm_ref_{f32,f16,bf16}`, selected per call class: `all`, `b1` (unbatched), `bN` (batched), a dtype, or `kNNN`.
- They are gated with the crate: 3289 launches, worst ratio 0.994, mutations 45/45, all entries launched (`out/gate-w7.log`).
- REDCELL's GEMMs at G1 are two classes:
  - the MoE router: `b1`, m x 128 x 2816;
  - the eager attention batches: `bN`, b16, QK^T and PV.
- Root cause: titan cast the router to the model dtype (bf16). The GGUF's F32 `ffn_gate_inp` weight, the router norm input and the logits were all rounded to bf16.
  - llama.cpp keeps all three in F32.
  - The 2^-8 rounding of the logits flips near-tie expert choices.
  - cuBLAS and the oxide kernels then land on different sides of those ties (deployed 39/40, noblas 37/40).

**Fix** (mr e623480f0, `vision_models/gemma4/text.rs`):
- The gemma4 MoE router now runs in F32. The GGUF's router weight and norm scale are re-read at F32 (not rounded through bf16), the norm runs in F32, and the logits are F32.
- `TITAN_G4_ROUTER_F32=0` restores the old path.
- It applies to every gemma4 MoE model whenever the model dtype is not F32, so it is not REDCELL-specific. Gemma4-12B (dtype F32) and every other architecture are untouched.
- The router GEMM itself runs on the gated SIMT f32 kernel (f32 FMA accumulation, split-K summed in a fixed order). In round 1 the same shape class ran at 0.66x cuBLAS (1x128x2880 GEMV); m x 128 x 2816 runs at prefill speed (below).

**REDCELL G1 vs llama.cpp** (`ref-red-g1.json`):

| build | top-1 | median dlp | KL median / mean | vs deployed: top-1, KL median |
|---|---|---|---|---|
| deployed (cuBLAS, bf16 router) | 39/40 | 0.0084 | 0.00078 / 0.0053 | - |
| noblas round 1 (bf16 router) | 37/40 | 0.0101 | 0.00088 / 0.0050 | 38/40, 0.0003 |
| **noblas2 (F32 router)** | **39/40** | **0.0069** | 0.00118 / 0.0035 | 40/40, 0.00072 |
| noblas2 + f64 router (`TITAN_GEMM_REF=b1`) | 38/40 | 0.0079 | 0.00068 / 0.0036 | 39/40 |
| noblas2 + f64 attention (`bN`) | 39/40 | 0.0113 | 0.00104 / 0.0040 | 40/40 |
| noblas2 + f64 everywhere (`all`) | 38/40 | 0.0071 | 0.00059 / 0.0033 | 39/40 |

Per-prompt evidence (llama.cpp top-1 > second with its gap in nats, then titan's order and gap):
- deployed: #39 llama 'A' > 'To' 0.325 | titan 'To' > 'A' 0.186
- noblas round 1:
  - #17 'In' > '###' 0.204 | '###' > 'In' 0.117
  - #22 'To' > 'In' 0.403 | 'In' = 'To' 0.000
  - #39, as deployed
- **noblas2**: #39 only, 'To' > 'A' at 0.000 (an exact tie in titan)
- f64 variants: #17 (titan gap 0.059) and #39 (0.06-0.12)

Every remaining flip is a llama.cpp near-tie (0.2-0.4 nats) that even f64 accumulation does not settle. The F32 router removes the round-1 regression (#17, #22) and lowers mean KL vs llama from 0.0053 to 0.0035.

**REDCELL G3:** 4/4, dlp 0.0741, KL 0.0166. That is **within llama.cpp's spread** (worst variant 0.1638 / 0.0328); deployed was 0.067 / 0.022.

**Other models:**
- All 12 other roster models give G1/G3 top-20 **identical** to noblas round 1 (`out/new2-*` vs `out/new-*`), so their verdicts above stand.
- 35B G3 is still top-1 3/4: **outside the spread on top-1 only**. The flipped prompt is the 0.16-nat tie described in section 4; dlp 0.0015 and KL 0.0004 are far inside.

**Speed** (noblas2 / deployed):
- REDCELL prefill G3 4990 / 5077 (-1.7%), G5 4885 / 4953 (-1.4%).
- REDCELL decode 136.2 / 136.8.
- Every other model is within noise of round 1 (`out/dec-new2-*`).

**Proof:**
- `ldd` / `nm -D`: 0 cuBLAS/cuRAND entries.
- nsys of REDCELL on noblas2: 5505 kernels and no cuBLAS NVTX table (`out/proof2-red.sqlite`).

## Round 3: the other MoE routers, and the 35B G3 flip (windows 9-10) -> final `bin/mistralrs-titan-noblas3`

**Final binary:** `bin/mistralrs-titan-noblas3`, sha256 `2616ce6a66b89e5f2c0283bcc22fd3db5a78d81ce71899e4154fbb0c09aaa7d5`. It is built from mistral.rs e623480f0, candle e1ce8b5b and oxide-kernels bd9992d (the embedded PTX is byte-identical to its `gemm.ptx`).
- The only change since noblas2 is a grid guard in the diagnostic `TITAN_GEMM_REF` switch.
- Its default output is identical to noblas2: 35B G1 40/40 and G3 4/4 top-20 identical.
- Proof: `ldd` 0, `nm -D` 0, cuBLAS/cuRAND API strings 0.
- nsys of the 35B on noblas3: 57,990 kernels and no cuBLAS NVTX table (`out/proof3-q35.sqlite`).

**No router rounding on the titan MoE paths.** Router input, weight, logits and top-k are all F32, as in llama.cpp's `build_moe_ffn`:
- **qwen35moe / qwen35 / qwen3next** (35B, IQ2_M, MXFP4, qwen3-next-80b; `models/quantized_qwen35_moe.rs`):
  - The residual is F32 (`x_attn = (h + residual)`, :2418) and the router input is the F32 `post_attention_norm` output (:2423).
  - `flat ... .to_dtype(F32)` (:770 / :818 / :898).
  - The router is `QMatMul::from_qtensor(ffn_gate_inp)` (:2005; F32 in the GGUF; the inventory logs it as an F32 GEMM).
  - The MTP head's BF16 routers are dequantized to F32 (`host_f32`, :1563 / :1820).
  - Logits are F32 (`route`, :881-882), and softmax / top-k are F32 (`ops.rs` :291).
  - The shared-expert gate `ffn_gate_inp_shexp` is loaded as F32 (:1621-1630) and applied in F32 with an F32 sigmoid (:825 / :968).
- **gpt-oss** (`models/quantized_gpt_oss.rs`):
  - The residual is F32 (:108 / :172 / :458-462).
  - The router is `lin("ffn_gate_inp")` (:333) on the F32 `flat`, giving F32 logits (:191).
  - Raw top-k then softmax in F32 (:193; `ops.rs` :291).
- Only gemma4-MoE (REDCELL) rounded its router to the model dtype; that was fixed in round 2. So the 35B's outputs, MTP greedy, tiering decisions and speed are those of round 1 / noblas2. No re-run was needed: noblas3 is output-identical.

**The 35B G3 flip is reduction-order noise.** On noblas3, `TITAN_GEMM_REF` routes the 35B's F32 products (router, shared-expert gate, GDN projections; selectors `b1` / `f32` / `all`) through f64 accumulation:

| run | G3 top-1 | dlp / KL med | prompt 3 top-2 (nats) | G1 top-1 / dlp / KL |
|---|---|---|---|---|
| llama.cpp reference | - | - | '' -1.120, '\n\n' -1.278 | - |
| noblas3 default | 3/4 | 0.0015 / 0.00044 | '\n\n' -1.222, '' -1.319 | 39/40 / 0.069 / 0.0074 |
| f64 F32 products (`b1`, `f32`, `all`: identical) | 4/4 | 0.0011 / 0.00024 | '' -0.926, '\n\n' -1.533 | 38-39/40 / 0.048-0.061 / 0.0065 |
| f64 attention batches (`bN`) | 3/4 (identical to default) | | | 39/40 |

- Prompt 3's order follows the accumulation order alone; the f64 run overshoots llama.cpp's margin in the other direction.
- f64 accumulation also moves a G1 near-tie the other way (38/40).
- By the user's decision, the 35B ships as is.

## Final gate table (noblas2 / noblas3 output; RULES-agents.md)

G1 needs top-1 >= 39/40, median |dlp| < 0.05 and KL < 0.01. G3 needs to be within llama.cpp's own spread; the absolute dlp < 0.05 / KL < 0.01 is shown too. "dep" is the deployed binary on the same prompts.

| model | G1 top-1 / dlp / KL (dep) | G1 | G3 | G3 verdict |
|---|---|---|---|---|
| qwen3.6-35b | 39 / 0.069 / 0.0074 (dep 39 / 0.056) | FAIL dlp (dep fails too) | 3/4, 0.0015 / 0.0004 | outside spread on top-1; reduction-order noise; ships (user) |
| qwen3.6-35b-mxfp4 | 37 / 0.049 / 0.0027 (dep 38) | FAIL top-1 (dep fails too) | 4/4, 0.0012 / -0.0002 | PASS (within; abs pass) |
| qwen3.6-35b-iq2m | 36 / 0.063 / 0.011 (dep 38 / 0.068) | FAIL (dep fails too) | 4/4, 0.0025 / 0.0008 | PASS (within; abs pass) |
| qwen3-next-80b | 39 / 0.029 / 0.0028 (dep 38) | PASS | no llama.cpp G3 (CPU experts at 2.5 tok/s); vs dep 4/4 top-1 | not gated vs llama.cpp |
| gpt-oss-120b | 40 / 0.015 / 0.0008 (dep 39) | PASS | not run (48 tok/s prefill) | not gated |
| gpt-oss-20b | 39 / 0.020 / 0.0000 (dep 39) | PASS | 4/4, 0.068 / 0.023 (6000-char prompts) | PASS (within) |
| bonsai-27b | 40 / 0.014 / 0.0017 (dep 40) | PASS | identical to dep | as deployed |
| bonsai2-27b | 40 / 0.004 / 0.0008 (dep 40) | PASS | 4/4, 0.0009 / 0.0005 | abs PASS; outside llama.cpp's very narrow spread (KL worst 0.0002); dep same |
| qwen3-14b | 40 / 0.000 / 0.0000 (dep 40) | PASS | identical to dep: 4/4, 0.039 / 0.0004 | PASS (within) |
| spark-x2.5 | 39 / 0.011 / 0.0037 (dep 40) | PASS | identical to dep: 4/4, 0.030 / 0.0056 | abs PASS; outside the narrow c13500 spread, as dep |
| gemma4-12b | identical to dep: 39 / 0.003 / 0.0004 | PASS | identical to dep: 4/4, 0.075 / 0.018 | PASS (within) |
| redcell-26b | 39 / 0.0069 / 0.0012 (dep 39) | PASS | 4/4, 0.074 / 0.017 | PASS (within) |
| orcasaq2-cyber-27b | 40 / 0.000 / 0.0000 (dep 40) | PASS | identical to dep: 4/4, 0.0038 / 0.0002 | abs PASS; outside the narrow spread, as dep |
