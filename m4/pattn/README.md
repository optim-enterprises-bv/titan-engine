# pattn: PagedAttention in the nvcc-free (oxide) build (staged, NOT deployed)

## pattn3: per-model serving settings in the swap server (staged, NOT deployed)

Binary **`bin/mistralrs-titan-pattn3`**, sha256 `a4f16c82a7574a6cfb8e8fae2bd459db636c780d6f92800ef040878e08c4c651`
(`bin/mistralrs-titan-pattn3.sha256`), built from mistral.rs `ddfe94d8f` (branch `pattn`, which already contains
model-swap 2c4b140e3: `git log pattn..model-swap` is empty). Roster to deploy: **`m4/pattn/models-deploy.toml`**
(= deploy/models.toml + 4 lines for qwen3-14b, 5 for Spark; every other entry unchanged).

New `[models.titan]` keys (absent = the global value, so an unchanged roster builds the same configs as before):

| key | global it overrides | goes into |
|---|---|---|
| `paged_attn = "on" / "off" / "auto"` | `[paged_attn] mode` | the model's PagedAttention config |
| `pa_cache_type = "auto" / "f8e4m3"` | `[paged_attn] cache_type` | KV pool dtype |
| `pa_context_len` or `pa_memory_mb` (not both) | `[paged_attn]` pool sizing | KV pool size |
| `max_num_batched_tokens` | `[runtime] max_num_batched_tokens` | paged scheduler step + device-map activation rows |
| `max_seqs` | `[runtime] max_seqs` (default 32) | scheduler only (paged `max_num_seqs` / default scheduler fixed batch) |

- What limits concurrency today: `[runtime] max_seqs` (default 32) in the scheduler; the deployed roster (paged off)
  runs the default scheduler, which batches only sequences of equal length. `[models.device] max_batch_size` (default 1)
  only sizes the auto device map. Per-model `max_seqs` never reaches the device map, so it cannot offload layers.
- Code: `TitanModelSettings` (mistralrs-core titan_swap.rs) carries the values; `build_titan_swap` (server-core)
  resolves them per model into the loader config and, for registered models, `SchedulerConfig::titan_swap_template`
  (paged: placeholder geometry the load's realized pool replaces).
- Unit tests: CLI `titan_serving_settings_are_per_model`, `titan_serving_settings_reject_bad_values` (+ the pcache ones);
  core `titan_swap_template_*`.

### 35B repeat TTFT after a full swap chain, A/B (window C2, same order and port, out/c2-*.pc.json)
Chain: 35B pc -> Spark, qwen3-14b, gemma4, REDCELL, OrcaSAQ (one ~4k prompt each) -> 35B pc x3.

| binary + roster | before | after 1 / 2 / 3 (median of 7 repeats) | decode | 35B fraction |
|---|---|---|---|---|
| mistralrs-titan-swap + deploy/models.toml | 0.51 s | 0.53 / 0.36 / 0.29 s | 140 tok/s | 0.54 |
| mistralrs-titan-pattn3 + models-deploy.toml | 0.45 s | 1.01 / 0.30 / 0.52 s | 140-141 tok/s | 0.54 |

The first rounds after the reload are noisy on both binaries (page cache refill after OrcaSAQ: the 35B reload took
37.6 s old / 46.4 s new vs ~25 s cold); both settle around 0.3-0.5 s. Free VRAM after every unload is the same on
both (15507 MiB; 15409 after OrcaSAQ on both).

### Swap gate (window C1, one from-config server, models-deploy.toml on port 18670, out/sw.server.log)
35B -> Spark -> qwen3-14b -> gemma4-12b -> REDCELL -> OrcaSAQ -> 35B. No OOM, no panic, no error.

| step | load | VRAM free after load | work | result |
|---|---|---|---|---|
| start | | 15661 MiB before any model | | |
| 35B | 26.6 s | 2781 MiB | pc (repeat TTFT) | 0.41 s median, 143 tok/s |
| Spark (paged, 8522 MB pool, 60544 tokens) | 1.3 s | 4467 MiB | 4 concurrent ~8k (7601-9822 tokens), 64 tokens | 4/4 ok, peak 12714 MiB |
| | | | 4 x 128 / 4 x 256 concurrent | **x1.98 / x2.00** (257 / 255 tok/s) |
| qwen3-14b (paged, 3196 MB pool, 20416 tokens) | 3.8 s | 2289 MiB | 4 concurrent ~16k (11245-14804 tokens; 53k > pool: admitted as blocks free) | 4/4 ok, 63 s, peak 14764 MiB |
| | | | 4 x 128 concurrent | **x1.71** (82.9 tok/s) |
| gemma4-12b (paged off) | 3.8 s | 8817 MiB | 2 x ~8k, prefix cache 0 | 2/2 ok, max 12268 MiB |
| REDCELL | 6.4 s | 3825 MiB | 3 x ~4k | 3/3 ok, max 13998 MiB |
| OrcaSAQ | 8.4 s | 1423 MiB | 1 x 27000 chars | 1/1 ok, max 15246 MiB |
| 35B | 57.0 s (page cache) | 2625 MiB | pc x2 | 0.66 s, then 0.51 s median; 140 tok/s |

- **Paged pool fully released:** free after unloading Spark (paged) 15505 MiB, qwen3 (paged) 15505 MiB, the 35B 15507 MiB,
  gemma4 15505 MiB. gemma4 loaded after qwen3 with 15505 MiB free, as after any model.
- **35B tiered fraction unchanged:** 0.54 on both loads (the second plans with the first load's 15565 MiB).
- **Spark / qwen3 numbers through the swap server = pattn2:**
  - qwen3-14b G1 40/40, G3 ~8k 4/4, G3 ~16k 4/4 **bit-identical** (top-20) to q7-on; G5 decode 49.3/49.6 (49.3/50.0).
  - Spark G1 40/40 bit-identical to s7-big. G3 with Spark's roster prefix cache (16) is served from cached blocks
    of the preceding burst (prefill 250k tok/s): 4k tokens 4/4 and dlp 0.025 / KL 0.0024 (within llama's spread);
    8k tokens 3/4 identical, dlp 0.095 / KL 0.0027. With `prefix_cache_n = 0` (out/models-gate-pc0.toml) G3 4k and 8k
    are **bit-identical** to s7-big (= paged off). The difference is the cached KV's chunking (computed in a 4-way
    concurrent 16384-token step), not the swap. Decode G5 137.9/139.4 (139.4/138.8).

### Regression core, paged off, bin/mistralrs-titan-pattn3 vs bin/mistralrs-titan-swap (window C1)
35B MTP2 40/40, MTP0 40/40; Bonsai-2 8/8 (o-off and deployed); IQ2_M 8/8; Bonsai-27B Q1_0 8/8; Spark / gemma4 / REDCELL
G1 40/40 identical each (top, tokens, text).

## pattn2 (phase A + B): the five blockers fixed

Binary **`bin/mistralrs-titan-pattn2`**, sha256 `b01d0c57a4a0f2537fb62d3bbad37fcef295c75c93f8fdb9ad60877226aa5034`
(`bin/mistralrs-titan-pattn2.sha256`).
- Built in window B3 from clean commits: mistral.rs `00d75bfda`, oxide `e5949f1`.
- Not deployed. `deploy/models.toml` and `bin/mistralrs-titan-swap` are untouched. Nothing was pushed.
- Six GPU windows were used: A1-A3 and B1-B3 (logs `window_{a1,a2,a3,b1,b2,b3}-*.log`).

| repo (branch `pattn`) | new commits |
|---|---|
| mistral.rs (top-pattn/mr-pattn) | a03026bf9 Spark/gemma2 sliding-mask fix, titan flash prefill for paged prompts, planner/admission, TITAN_PATTN_TRACE/DUMP, qwen3/spark2_5 `model_config`; db23d75f3 device map sizes activations for one paged prompt chunk (`prefill_rows`); 00d75bfda unstub the v0.9.4 graph helpers |
| oxide-kernels (oxide-pattn) | e5949f1 FP8 E4M3 KV decode (432 stage-2 flashinfer_decode instances) + v0.9.4 CUDA-graph helpers |
| titan-engine (top-pattn) | this directory |

### 1. Spark ~8k wrong answers with paged ON: root cause proven, fixed
- **Cause:** on Spark's sliding layers, `spark2_5.rs` passed the *full* causal mask to the paged forward. The first prompt
  chunk (up to 4096 rows) applies the model mask as given, so sliding layers attended outside their window.
- **Proof:** A1 tensor dumps (`TITAN_PATTN_DUMP`, `pdump.py`/`cmpdump.py`, g3l prompt 1, 7719 tokens), layer-0 attention
  output, ON vs OFF:

  | binary | first row >5% rel | rows 512-4095 (median rel) | rows >=4096 |
  |---|---|---|---|
  | old mask | row 602 | 0.13-0.66 | 0.003 |
  | fixed mask | none (L0 max 0.0054) | | |
  | flash prefill path | bit-identical (0.0000) | | |

  The same fix is applied in `gemma2.rs`.
- **Spark, one chunk per prompt** (`--max-num-batched-tokens 16384`): G1, G3 4k and G3 8k are **bit-identical to paged
  OFF** (top-20, tokens, text; s6-big, re-confirmed s7-big in B2).
- **Default 4096-token chunks:** chunking alone changes the numerics (stream-k / row-count-dependent kernels; layer 3 ~1.4%
  rel).
  - 8k: dlp 0.097, KL 0.0099. Paged OFF: dlp 0.044, KL 0.0068.
  - llama.cpp's worst self-spread: dlp 0.052, KL 0.0060.
  - G1 is 40/40 vs llama, identical to OFF.

### 2. Paged prefill uses titan's flash-prefill kernels
`try_titan_flash_prefill` (paged_attention.rs):
1. Writes the chunk's K/V.
2. Gathers each sequence's keys with `gather_kv_cache_flashinfer`. Sliding layers start from the window's first block.
3. Runs `flash_prefill_any` with the window.

It falls back to the gather+SDPA path when the shape is unsupported. Prefill tok/s:

| | paged ON | paged OFF |
|---|---|---|
| Spark ~8k (g3l) | 5911 / 5924 / 5691 / 5697 | 5825 / 5799 / 5660 / 5645 |
| qwen3-14b ~8k | 2206 / 2226 / 2131 / 2098 | 2150 / 2169 / 2059 / 2010 |
| qwen3-14b ~16k | 2023 / 1935 / 1862 / 1857 | 1939 / 1858 / 1778 / 1778 |

Before this change, Spark was at ~1.3k tok/s and qwen3 at ~1.4k tok/s.

### 3. qwen3 2.92 GB scratch 503 removed
- The prompt workspace is sized to the flash-prefill workspace for the planned chunk (`plan.rs`).
- qwen3 and spark2_5 now report prefix-prefill features (`model_config`). Without them the plan was `None` and every
  long prompt took the gather path.
- The device map sizes activations for one paged prompt chunk (`prefill_rows` = max_num_batched_tokens), so the default
  pool (3352 MB, 21408 tokens) leaves room.
- qwen3 ~8k and ~16k now succeed with the default pool. VRAM after ~16k is 14916 MiB.

| qwen3-14b G3 | ON | OFF | llama self-spread (worst) |
|---|---|---|---|
| ~8k | top-1 3/4, dlp 0.0322, KL 0.00064 | top-1 3/4, dlp 0.0353, KL 0.00238 | top-1 4/4, dlp 0.026, KL 0.0024 |
| ~16k | top-1 4/4, dlp 0.0510, KL 0.00495 | top-1 3/4, dlp 0.0616, KL 0.00405 | top-1 3/4, dlp 0.050, KL 0.0046 |

- ON is at least as close to llama as OFF.
- At 8k both are outside the spread only on top-1 (one near-tie prompt).
- At 16k, ON's dlp/KL is 0.001/0.0004 over the spread.
- G1: ON is 40/40 identical to OFF, and vs llama KL is 0.00001.

### 4. Upstream CUDA decode graphs (item 3)
- Kernels that the graph path needed, all ported in `mistralrs-core-cuda`:
  - `pad_decode_input_u32` (input_packing.cu);
  - `pack_completion_input_u32` (input_packing.cu, 64-pointer staged rows);
  - `cuda_graph_copy_2d_bytes` (graph.cu, a cuMemcpy2DAsync wrapper; needed the `cuMemcpy2DAsync_v2` binding in
    titan-oxide-ffi `cu.rs`).
- **Gates:** they match bit for bit against the cudaforge v0.9.4 objects. The objects are linked under renamed symbols
  (`v094_*`, build.rs). Their SASS was first checked against `ref094/nvcc_094.sh` rebuilds.
  - B1 gate: 60 launcher calls, 262516 bytes, 0 failing.
  - Mutants `core-pad-lastrow` and `core-pack-shift` were both DETECTED.
- **Running it:** qwen3-14b paged ON with `MISTRALRS_CUDA_GRAPHS` unset loads and captures 18 decode graphs (batch buckets
  through 32). A new binary's first start pays a one-time ~74 s PTX JIT; later starts take ~1.4 s.
  - G1, graphs on vs off: 40/40 identical (top, tokens, text). Graphs on is also 40/40 identical to paged OFF.
  - Decode: 49.3/50.0 tok/s with graphs on, 49.4/50.0 with graphs off. Paged OFF is 39.1/39.5. titan's own graphs
    already cover batch-1 decode.
- REDCELL paged ON had died at load on `pad_decode_input_u32`. It now loads and serves.
- No further nvcc-only stub was hit (abort mode).

### 5. FP8 E4M3 KV-cache decode (item 5)
- **Port:** `batch_decode<T, KV>` gained an `E` element type: one byte, decoded per element with the nvcc sequence
  `cvt.rn.f16x2.e4m3x2` → f32.
  - 432 FP8 stage-2 instances were generated. The 432 FP8 stage-1 instances are unreachable on cc 12, which selects
    `NUM_STAGES_SMEM=2`.
  - The launcher dispatches on `cache_dtype` 3 (tile 2 for group 1). The 801 refusal was removed.
- **Build fix found in B1:** 891 mangled kernel names overflowed the 128 KiB argument limit of `opt`'s internalize list.
  cargo oxide then printed "continuing with unoptimized IR", and every crate-b kernel ran unoptimised (qwen3 decode
  20.9 tok/s).
  - The 864 decode entries now use short oxide names `bd<n>`. `Inst.ox` maps each reference name to its oxide name, and
    `ptxrename.py` maps them back for PTX comparisons.
  - The rebuilt PTX matches the committed PTX instruction for instruction, apart from the dynamic-smem symbol name.
- **Gate (B2, optimised PTX):** bit-identical vs the v0.9.4 nvcc objects.
  - 8610 launcher calls and 1.78 GB compared, 0 failing.
  - Decode family: 4320 calls. Coverage: 883 of 883 reachable instances.
  - Mutants DETECTED: FP8 byte order (`b-fp8dec-byteorder`) and the v_scale multiply reorder, now on the **f32** instance
    (`b-vscale-reassoc`). The weak f16 mutant has been redone.
- **Spark FP8 KV end to end** (`--pa-cache-type f8e4m3`; uncalibrated unit scales, as upstream warns):

  | | bf16 KV | FP8 KV |
  |---|---|---|
  | VRAM after load, fixed 16384-token pool | 5156 MiB | **4004 MiB (-1152 MiB, half the pool)** |
  | VRAM after ~8k prompts, same pool | 5894 MiB | 4742 MiB |
  | default pool (10504 MB) context | 74656 tokens | 149344 tokens |
  | decode tok/s (G5) | 139.7 / 139.6 | 140.2 / 140.2 |
  | prefill tok/s ~8k | 5911 / 5924 / 5691 / 5697 | 5911 / 5915 / 5691 / 5687 |
  | G1 vs llama (top-1, KL median) | 40/40, 0.0035 | 40/40, 0.0035 (identical to bf16) |
  | G3 4k vs bf16 KV | | top-1 4/4, dlp 0.057, KL median 0.0118 |
  | G3 8k vs bf16 KV | | top-1 3-4/4, dlp 0.078-0.107, KL median 0.028-0.038 |
  | decode KL vs bf16 KV (`decodekl.py`, 16 prompts x 128) | | 937 positions to first divergence, top-1 924/937, KL median 0.00029, p99 0.017 |

  - G1 is identical because short prompts never read the cache. They take the regular first-chunk path below 1024 keys.
  - llama's `-ctk q8_0` is a different format, so no llama comparison is claimed.
  - G3 vs llama: FP8 is outside the spread (4k KL 0.015, 8k KL 0.015-0.028). That is the cost of uncalibrated FP8.

### Full paged-ON table (final binary or identical B2 build; paged OFF in brackets)
| model | G1 | G3 | prefill tok/s ~8k | decode tok/s | VRAM |
|---|---|---|---|---|---|
| Spark, `--max-num-batched-tokens 16384` | 40/40 identical to OFF | 4k + 8k identical to OFF (8k: dlp 0.044, KL 0.0068) | 5758 (5825) | 139.4 (121.0) | 11524 MiB load, 13318 after 8k |
| Spark, default chunks | 40/40 identical to OFF | 4k dlp 0.035, KL 0.0092 (PASS); 8k dlp 0.097, KL 0.0099 (outside spread) | 5911 (5825) | 139.7 (121.0) | 13380 / 14086 MiB |
| Spark FP8 KV | identical to bf16 | see above | 5911 | 140.2 | -1152 MiB per 16k pool |
| qwen3-14b (graphs on) | 40/40 identical to OFF | 8k dlp 0.032, KL 0.0006; 16k dlp 0.051, KL 0.005 | 2206 (2150) | 50.0 (39.5) | 13860 load, 14916 after 16k |
| REDCELL 26B (`-n 0:30`) | 40/40 identical to the deployed binary | vs OFF: top-1 4/4, dlp 0.149, KL median 0.010 | 4k: 2616 (4978) | 125.0 (120.6) | 14062 (12164) MiB |
| gemma4-12b (f32) | tokens 40/40 = OFF (KL 0.00006) | **fails**: OOM, then `FlashInfer decode failed ... head_size=512` | - | - | 14436 MiB (f32 pool 7392 MB) |

- **REDCELL** pays a prefill penalty. Its prompts take the regular first-chunk path (`TITAN_PATTN_TRACE`: "regular prompt
  (chunk K/V, model mask)") because it does not report prefix-prefill features. Its G3 also differs from OFF.
- **gemma4-12b** cannot run paged: upstream FlashInfer has no head_size-512 decode instances (its global layers), and its f32
  KV pool leaves no headroom.

### Batching table (`--max-seqs 4 --prefix-cache-n 0`, 4 concurrent vs the same 4 sequential, greedy)
| model | paged | single tok/s | concurrent aggregate | speedup | identical to sequential |
|---|---|---|---|---|---|
| Spark | ON | 129.6 | 255.3 (128 tok), 253.3 (256 tok) | x1.97 / x1.99 | 1/4, 0/4 |
| Spark | OFF | 117.8 | 191.9 (128); 256 tok served serially (106.5) | x1.63 / x0.92 | 1/4; 4/4 when serial |
| qwen3-14b | ON | 49.0 | 84.1 | x1.72 | 0/4 |
| qwen3-14b | OFF | 45.9 | 73.5 | x1.60 | 2/4 |

Concurrent and sequential outputs diverge only at near-ties:
- At the first divergent token, the gap between the top-1 and top-2 logprobs is 0.0-0.33 nats (`conc.py`).
- The batched decode reduces in a different order than batch 1. Both paged ON and OFF show this.

### Regression core: paged OFF on the final binary vs bin/mistralrs-titan-swap (B2 and B3)
| check | result |
|---|---|
| 35B MTP2 vs m6/out/h-off.json | 40/40 |
| 35B MTP0 vs m6/out/mtpfile-off.json | 40/40 |
| Bonsai-2 B2-off vs o-off / deployed | 8/8 / 8/8 |
| IQ2_M 8x300 vs deployed | 8/8 |
| Bonsai-27B Q1_0 8x300 vs deployed | 8/8 |
| Spark / gemma4 / REDCELL G1 vs deployed | 40/40 identical each (top, tokens, text) |

### Roster recommendations (paged ON, `--max-seqs` > 1)
- **Spark:** `--paged-attn on --max-seqs 4 --max-num-batched-tokens 16384`.
  - Prompts up to 16k are one chunk, so they are bit-identical to OFF.
  - Decode is +16%, and 4 concurrent requests give x1.97 aggregate. The concurrency run used default chunking; the
    combination with 16384-token chunks was not measured.
  - Add `--pa-cache-type f8e4m3` only where context matters more than G3 accuracy (2x tokens per MB).
- **qwen3-14b:** `--paged-attn on --max-seqs 4` (default pool). Decode is +26%, and ~16k prompts fit. Upstream graphs may
  stay on.
- **REDCELL:** keep paged OFF. It works, but prefill is ~2x slower and G3 drifts from OFF. Enabling it needs prefix-prefill
  features in its model.
- **gemma4-12b:** keep paged OFF (no hd512 FlashInfer decode).

---

# pattn (v1, superseded by pattn2 above)

Binary **`bin/mistralrs-titan-pattn`**, sha256 `5734ae969f4a26aba0ab3aabf6624eef57e3c72f0ae1a7de036f5976692ebbf3`
(`bin/mistralrs-titan-pattn.sha256`). Built in window 3 with `cargo build --release -p mistralrs-cli --features oxide`
(CARGO_TARGET_DIR target-pattn-oxide, seeded from target-orca-oxide).
- Not deployed. `deploy/models.toml` and `bin/mistralrs-titan-swap` are untouched. Nothing was pushed.

## Branches (`pattn` in all three repos)
| repo | worktree | commits on top of BASE |
|---|---|---|
| oxide-kernels | oxide-pattn (top-pattn/oxide-kernels -> ../oxide-pattn) | abf90f5 v0.9.4 ABI for paged-attn-a/-b + titan-oxide-ffi regen; ddf5eca `cuFuncGetAttribute` binding (on glu-typed-kernels da6602b) |
| mistral.rs | top-pattn/mr-pattn | f44b88fcc unstub the launchers; 0709064c1 gqa_grouped_sdpa paged-V fix (on model-swap 2c4b140e3) |
| titan-engine | top-pattn (sparse: candle + m4/pattn) | this directory (on master bc486050; candle unchanged) |

## What was missing, and what changed
Paged attention died on load in the oxide build. The cause was nvcc-only stubs for the v0.9.4 ABI of
`reshape_and_cache_flashinfer`, `gather_kv_cache_flashinfer` and `copy_blocks_u8`.

**Next missing piece found: `flashinfer_decode`.** Its ABI also changed in v0.9.4 (+k_scale, v_scale, cache_dtype), and every
paged decode step on a FlashInfer cache calls it. It was ported too:
- `mistralrs-paged-attn-b`:
  - reshape / gather for all six (activation, cache) pairs, FP8 E4M3 included. The write is `div.approx.ftz` then
    `cvt.rn.satfinite.e4m3x2.f32 d, 0, x`. The read is `cvt.rn.f16x2.e4m3x2`, `cvt.f32.f16`, `mul.ftz`, `cvt.rn`.
  - gather's `num_seqs` bound, and the zero-fill past `cu_seq_lens[num_seqs]`.
  - decode: v0.9.4 folds `k_scale` into sm_scale on the host and applies `v_scale` in OutputTransform as
    `(o * d_rcp) * v_scale` (BatchDecodeParams offset 156).
  - GQA groups 5/6/7: 144 new stage-2 instances (qwen3-14b is group 5).
  - **FP8 E4M3 KV-cache decode (864 reference instances) is NOT ported.** With cache_dtype 3 the launcher prints why and
    returns cudaErrorNotSupported (801). The caller bails with "flashinfer_decode failed with status 801".
- `mistralrs-paged-attn-a`:
  - `copy_blocks_u8`.
  - v0.9.4's `gather_kv_cache` `batch_id >= num_seqs` guard. This is a semantic change in an existing kernel; the old twin
    read the block table out of bounds.
  - v0.9.4's host error paths: pagedattention.cuh `CUDA_CHECK(VLLM_EnsureMaxDynamicSharedMemorySize)`, which exits before the
    launch, and the reshape_and_cache line 139.
- `titan-oxide-ffi`: regenerated with gen.py. The example gate, make_ref.sh and check_symbols.sh now use the v0.9.4 archive.
- mistral.rs:
  - the three `link_name = "titan_nvcc_only_*"` redirects are removed from `mistralrs-paged-attn/src/cuda/ffi.rs`;
  - four entries are removed from `titan-nvcc-only/nvcc_only.txt`;
  - the binary exports the real `flashinfer_decode`, `gather_kv_cache_flashinfer` and `reshape_and_cache_flashinfer`.
  - `copy_blocks*` is dead-stripped from both this binary and the deployed one: nothing in the serving path reaches it.

**Reference.** `oxide-kernels/reference/mistralrs-paged-attn-094/` (untracked) holds the nvcc 13.3 objects of the same v0.9.4
`.cu` files, from the titan-094 nvcc build root `cuda-build-094`. Their content hashes equal the sources at 2c4b140e3.
Provenance and checks are in `mistralrs-paged-attn-b/ref094/README.md`:
- `ref094/nvcc_094.sh` rebuilds copy_blocks_kernel.cubin **byte-identically** with `/usr/local/cuda/bin/nvcc`.
- The full flashinfer_decode.cu TU makes cicc segfault under the 4 GB CPU cap. The reduced TU (`ref094/reduced/`) is used
  instead. Its 164 kernels are **SASS-identical** to the reference object, and its PTX is where the FP8 conversions and the
  v_scale placement were read.

## Gate 1: kernels, bit-identical vs the v0.9.4 nvcc launchers (windows 1-2)
| crate | result | log |
|---|---|---|
| paged-attn-b | **PASS**: 5154 launcher calls, 1,277,949,568 B, 0 failing; coverage **451/451** reachable instances; families mla 142, reshape 222 (all 6 pairs; FP8 scales incl. 2e38, denormal, 0, -1.5, inf, NaN; -1 / i64::MIN / -7 padding slots; key/value strides ≠ nh*hs), gather 190 (6 pairs; padding tokens past cu_seq_lens, empty sequences), decode 2592 (f16/bf16/f32 × 64/128/256/512 × groups 1-8,16 × SW × softcap, split-KV, fused-qkv strides, random k/v scales), edges 216, errors 28 (bad dtype pairs, groups 9/10/12, smem limit), fp8refuse 24, kernel-level 1740 (the by-value Params struct with v_scale at 156) | out/gate-b.log |
| paged-attn-a | **PASS**: 2538 launcher calls, 5,436,402,887 B, 0 failing (copy_blocks u8 incl. 4097/4099/3/7-byte blocks; gather with 1-3 padding tokens; all 7 exit paths identical incl. `pagedattention.cuh:719`) | out/gate-a2.log (window 1's out/gate-a.log failed exactly the 2 exit-path messages that window 2 fixed) |

**PTX vs the committed PTX** (`ptxkind.py`: per-entry, comments and per-function numbering normalised):
- **a**: 372 entries identical. The 6 `gkv` entries differ (the guard), and `copy_blocks_kernel_u8` is new. The window-2 host
  fix left the PTX byte-identical.
- **b**: MLA 3 and merge 12 entries identical. All 288 non-FP8 decode entries differ (the v0.9.4 `v_scale` parameter and
  multiply). The 6 cache kernels are renamed by their two-type template; 156 entries are new.

**Mutation checks** (`mut.sh`, reduced kernel builds):
| mutant | result |
|---|---|
| FP8 write with IEEE `/` | DETECTED (22 cases) |
| FP8 read without ftz | DETECTED (6) |
| gather without the zero-fill | DETECTED (60) |
| copy_blocks_u8 as i16 | DETECTED (7) |
| gather_kv_cache without the guard | DETECTED (the Rust side faults with an illegal address) |
| exit line 140 instead of 139 | DETECTED |
| `o * (d_rcp * v_scale)` | **NOT DETECTED** |

The v_scale mutant was built with only the f16 head-128 group-4 instance. Its 10 calls compared equal: f16 output rounding
masks a 1-ulp f32 reassociation. A version with an f32 instance was not run (no window left). The full gate does exercise the
f32 instances with v_scale ∈ {0.5, 1.7, 1e-3, …}, so equality there is shown, but sensitivity to the placement is not.
- Stage-1 (`NUM_STAGES_SMEM=1`) instances: 435 are unreachable on sm_120 and were not separately gated this time.

## Gate 2: end to end, same binary, paged attention ON vs OFF (windows 2-3)
VRAM is nvidia-smi used. G1 compares the first token's top-20 over 40 prompts. Llama references:
- Spark: m4/g4s/out/refc-spark-g1.json, plus `ref-spark-cpu-g3l.json` (CPU llama.cpp at 28000 chars, made here, no LoRA).
- qwen3-14b: m4/devmap/out/lref-g1.json.

| model | G1 ON vs OFF | G1 vs llama | ~8k prompts (G3, 28000 chars) | 2k prompts prefill / decode tok/s | VRAM after load |
|---|---|---|---|---|---|
| Spark-X2.5 Q4_K_M (native spark2_5) | top-1 39/40, median dlp 0.0034, KL med 0.0011 (text 39/40) | ON 40/40, dlp 0.0118, KL 0.0035; OFF 39/40, 0.0124, 0.0037 (OFF == deployed 40/40) | **OFF vs llama 4/4, KL 0.0027. ON vs llama 2/4, dlp 0.47, KL med 0.27: WRONG** (window 2: HTTP 500 before the fix) | ON 3370/3513, 141.0/144.0; OFF 7010/6726, 121.0/123.7 | ON 5156 MiB (+2304 MB paged pool) / OFF 2916; after ~8k ON 6150 / OFF 8454 |
| qwen3-14b Q5_K_M (native qwen3), MISTRALRS_CUDA_GRAPHS=0 for ON | **40/40 identical** (top, tokens, text) | ON = OFF: 40/40, dlp 0.0000, KL 0.00001 | ON: **HTTP 503**, admission refused 2.92 GB of gather workspace with the 2.5 GB pool at `--pa-context-len 16384`; OFF vs llama 3/4, KL 0.0024 | ON 1397/1386, 49.0/49.2; OFF 2605/2513, 39.1/39.5 | ON 12964 / OFF 10500 MiB |
| Qwen3.6-35B UD-Q4_K_XL (titan qwen35 path) | 40/40 identical: paged attention is **disabled** (below) | n/a | ON == OFF 4/4 identical | same path | 13306 / 13306 MiB |

The 35B does not support paged attention at all. `mistralrs-core/src/pipeline/gguf_titan.rs:131` warns "titan GGUF models
run without PagedAttention (MTP, hybrid prefix cache and chunked prefill are unpaged); disabling it" (logged in
out/t-on.server.log). The flag was not forced.

## Gate 3: concurrency (Spark, `--max-seqs 4`, window 3, out/c3-*.json)
| | 1 request (wall) | 4 concurrent aggregate | per-request decode | identical to sequential (tokens) |
|---|---|---|---|---|
| paged ON, 4 x 128 tok | 141.3 tok/s | **269.6 tok/s (x1.91)** | 72.3-72.5 | 1/4 (common prefix 117, 110, 128, 53) |
| paged ON, 4 x 256 tok | 138.6 | **268.1 (x1.93)** | 71.3-71.4 | 0/4 (63, 83, 21, 134) |
| paged OFF, 4 x 128 | 128.5 | 200.1 (x1.56) | 53.2-53.5 | 1/4 (118, 61, 128, 72) |
| paged OFF, 4 x 256 | 126.3 | 115.4 (x0.91, served one after another) | 128.6-130.2 | 3/4 (256, 38, 256, 256) |

**Why the concurrent texts differ.** Two things change the numerics, and this probe cannot separate them:
- Batch-4 decode runs batched matmul kernels instead of the single-row ones. Greedy near-ties then diverge after 20-130 tokens.
  OFF shows the same pattern when it does batch.
- The burst re-sends prompts the sequential pass had just put in the prefix cache, so it starts from cached KV. OFF diverged on
  prompt 9 at token 38 even when it served the burst one request at a time.

A clean check would use `--prefix-cache-n 0`; it was not run (no window left).
- Window 2's probe, which also passed `--max-batch-size 4`, is invalid. That flag made the device map put layers 23-35 on the
  CPU, which also turned paged attention off.

## Gate 4: REGRESSION CORE, paged attention OFF (window 3) vs bin/mistralrs-titan-swap (6cdca99d)
| check | result |
|---|---|
| 35B MTP=2 vs m6/out/h-off.json | **40/40** (110.0 tok/s) |
| 35B MTP off vs m6/out/mtpfile-off.json | **40/40** (87.6 tok/s) |
| Bonsai-2 B2-off vs m4/bonsai2/out/o-off.json | **8/8** (and 8/8 vs the deployed binary's run) |
| IQ2_M 8x300 vs the deployed binary (m4/orca/out/i-new) | **8/8** |
| Bonsai-27B Q1_0 8x300 vs the deployed binary (q-new) | **8/8** |
| Spark / gemma4-12b / REDCELL G1 vs the deployed binary | **40/40 identical** each (top, tokens, text) |
| qwen3-14b OFF G1 (window-2 binary) vs devmap fq-g1 | 40/40 identical |

## Not done / next missing pieces (precise)
1. **Spark long-prompt accuracy with paged attention ON.** ~8k prompts disagree with llama.cpp and with OFF (KL ~0.27).
   - OFF agrees with llama, and ON short prompts agree, so the error is in the paged GatherSdpa prefill once the context
     outgrows one 512-token chunk.
   - Spark's sliding layers (window 512) go through `block_aligned_window_len_for_query` and `prefix_gather_causal_mask`.
   - The upstream nvcc build takes FlashAttentionPaged there (feature `flash-attn`), so this fallback is rarely exercised.
   - Root cause not found (no window left). qwen3, which has no sliding window, could not be measured at ~8k (item 3).
2. **qwen3 with upstream CUDA decode graphs** (MISTRALRS_CUDA_GRAPHS unset) aborts at load on the nvcc-only
   `pad_decode_input_u32` (`mistralrs-core/src/cuda/input_packing.cu`, called from `pipeline/cuda_graph.rs:551`).
   - The TITAN_NVCC_ONLY=warn discovery run reached only that stub, then faulted with garbage input (device-side assert), so
     later stubs on that path (likely `cuda_graph_copy_2d_bytes`) are unconfirmed.
   - With `MISTRALRS_CUDA_GRAPHS=0` qwen3 paged ON serves.
3. **qwen3 ~8k with paged ON: memory admission.** The 2.5 GB pool at `--pa-context-len 16384` leaves 2.97 GB free after load.
   The GatherSdpa prefill needs a 2.92 GB transient workspace, so the request gets 503. A smaller pool or a non-gather prefill
   would be needed.
4. **FP8 E4M3 KV-cache decode** (`--pa-cache-type f8e4m3`): not ported (864 instances); fails loudly with 801. The FP8
   reshape / gather / copy paths are ported and gated.
5. **Prefill is slower with paged ON.** The GatherSdpa eager fallback replaces titan's flash prefill:
   - Spark 2k prompts ~3.4k vs ~7k tok/s, ~8k prompts ~1.3k vs ~5.8k tok/s;
   - qwen3 1.4k vs 2.6k tok/s.

   **Decode is faster:** Spark +16%, qwen3 +25%.

## Logs
- Windows: window1-20261004-0435.log, window2-20261004-0446.log, window3-20261004-0532.log (+ *.nohup).
- Gates and mutants: out/gate-{a,a2,b}.log, out/mut-*.log.
- Server logs and gate JSON: out/*.server.log and out/*.json.
- CPU llama reference: out/ref-spark-cpu-g3l.json.
- 35B runs: m3/out/pattn-{m2,mf0}.json.

Window 2 lost ~30 min. Its rebuild waited on two fixes in titan-oxide-ffi: `cuFuncGetAttribute` was not bound and then not
re-exported.
