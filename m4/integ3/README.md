# integ3-20261003: integ2 + deq + g4acc + redcell2 + kv_cache OOM fix (staged, NOT deployed)

Binary: `bin/mistralrs-titan-integ3`, sha256 `1de00581740d5523af109644d450c4a5e5c4368e8e5a2af2606eaa2b0ee2a496`
(nvcc-free `cargo build --release -p mistralrs-cli --features oxide`, CARGO_TARGET_DIR `target-integ3-oxide` seeded from
target-integ2-oxide, TITAN_OXIDE_DIR `oxide-integ3`, built in 4m40s in window 1). `bin/mistralrs-titan-swap`,
`bin/mistralrs-titan-integ*` (other than integ3), `deploy/models.toml` and titan-mistral are untouched. Nothing pushed.
The file to deploy is **`m4/integ3/models.toml`** (below).

## Branch heads (all `integ3-20261003`, from `integ2-20261003`)
| repo | worktree | head | contents |
|---|---|---|---|
| oxide-kernels | `oxide-integ3` | 64d8b5a | merge g4acc 718a131 (6c1ef65), merge redcell2 7251efd (64d8b5a) |
| titan-engine (candle + m4) | `top-integ3` (sparse: candle, m4) | this README commit, on merges e3ef098f (deq a38bfb2b), aaf1e62d (g4acc 17b3fdb9), 7518a89c (redcell2 98a980c1) | |
| mistral.rs | `top-integ3/mr-integ3` | 86c8f605a | merges deq 1cdd8dee4 (3b18d9634), g4acc f6839fa5a (886a57d10), redcell2 39719cbdc (a564f8dac), then 86c8f605a (the OOM fix) |

`top-integ3/oxide-kernels -> ../oxide-integ3` is an untracked symlink (as in integ2).

### Merge notes
- **No conflicts in any repo.** g4acc and redcell2 touch disjoint oxide files (flash-prefill vs mistralrs-quant-c / fmt_mmq /
  titan-oxide-ffi `quant_c.rs`); redcell2 does not touch `vision_models/gemma4/text.rs` (its change is
  `moe/experts/backends.rs`), so no union was needed there; deq touches `mla/weights.rs` + `mistralrs-quant/src/gguf/mod.rs`.
- Generated ffi: `python3 titan-oxide-ffi/gen.py` on the merged tree regenerates all 7 modules **byte-identical** to the
  committed files (`git status` clean), so `quant_c.rs` is exactly redcell2's regenerated file.
- Paths: `git grep 'top-deq|top-g4acc|top-redcell2|oxide-g4acc|oxide-redcell2|top-integ2|oxide-integ2|top-emb|top-hd512|top-pcache'`
  in mr-integ3: **0**. `cargo metadata --offline --all-features`: candle-core / candle-nn / candle-kernels ->
  `top-integ3/candle`, titan-oxide-ffi -> `top-integ3/oxide-kernels/...` = `oxide-integ3/titan-oxide-ffi` (window1 log).
- Embedded `flash_prefill_oxide.ptx` == oxide-integ3 committed `flash-prefill/flash_prefill.ptx` (CR-normalised) = g4acc's.

## The fix: prefix-cache KV OOM no longer panics the engine (86c8f605a)
`NormalCacheManager::clone_in_cache` (`kv_cache/mod.rs`) built the batched per-step KV copy with
`Tensor::zeros(..).unwrap()` (the `:542` panic), `slice_set(..).unwrap()` and `contiguous().unwrap()`. The function already
returns `Result`, and its caller (`pipeline/mod.rs` step: `CacheInstruction::In => self.clone_in_cache(input_seqs)?`) runs
under `handle_pipeline_forward_error!`, which fails the sequences, resets the caches, evicts the prefix cache and calls
`titan_swap::recover_from_oom` on a CUDA OOM - the same path that already recovered the prompt-step OOM. The 8 unwraps are
now `?` (`.map(|x| x.contiguous()).transpose()?` for the Option ones). Nothing else changed.

**Gate (window 1, same window, same procedure on both binaries):** REDCELL, `--max-model-len 8192 --max-seq-len 8192 -n 0:30
--dtype bf16`, **default prefix cache (16)**, default MoE piece (2048), G1 -> G3 (~3.5k-4.7k) -> G2 256, every request sent on
its own and a failure recorded (`oomgate.py`; gates.py aborts on the first 500), then a probe request:
| binary | requests ok | OOMs | engine | after |
|---|---|---|---|---|
| **integ3** | **82/85**: G1 40/40, G3 3/4, G2 38/40 | 3 (G3[2] prompt step; G2[15], G2[32] **completion step**, i.e. the clone_in_cache path), each `titan: CUDA out of memory: the request failed ... pools trimmed (21 -> 3253 / 2837 MiB free)` | **0 panics**; the failed request gets HTTP 500 `model_error`, the next request succeeds | probe ok, unit active |
| redcell2 (`m4/redcell2/mistralrs-redcell2`, negative control) | 81/85 | same 3 points | **2 panics at `kv_cache/mod.rs:542:92`** (CUDA_ERROR_OUT_OF_MEMORY), `Engine ... is dead, rebooting` x2, G2[33] HTTP 503 | |
Logs: out/oom-new.server.log, out/oom-old.server.log, out/oom-{new,old}-oom.json, window1 log.
- Requests served after a recovered OOM are not corrupted: of the G2 rows that succeeded, the share whose tokens equal a fresh
  prefix-cache-0 server's (rz-a-g2) is the same before the first G2 OOM (6/15) and after it (11/23); the remaining differences
  are the default-cache vs cache-0 difference (redcell2 saw the same: dd-g1 vs dz-a-g1 top 0/40, tokens 40/40).
- The probe ("The capital of France is", raw completion) gives junk on REDCELL even on a fresh server (rz-a/b/e: `likely로...`),
  so it is only a liveness probe.
- The OOMs themselves remain (REDCELL at ~3.5k with a filled prefix cache does not fit 16 GB); the roster serves REDCELL with
  `prefix_cache_n = 0`, where none occur (pcache test and dry run: 0 OOM).

## Gate table
Windows: window1-20261003-1023.log (29m39s), window2-20261003-1053.log (15m30s), window3-20261003-1109.log (12m04s), with
`.nohup` copies. titan-spark stopped after taking the lock and restarted by the EXIT trap each time (active after each);
titan-mistral never started. No TITAN_* in the window environment.

| gate | integ3 result | reference | evidence |
|---|---|---|---|
| flash-prefill (regenerated and committed PTX) | **PASS**: vs v1 0 of 53100544 differ, splits worst 0.667 of tol; d512 / f16 / r2 / decode-row suites PASS; **existing kernels (256 bf16 x5, 256 f16, 512 bf16 x2, 512 f16 x2) vs the integ2 PTX: 292 calls, 0 of 632881152 outputs differ** | g4acc: same | out/gate-fp.log, gate2-fp.log |
| mistralrs-quant-a | **PASS 7004 launcher calls, 572862725 bytes, 0 failing** | 7004 / 0 | out/gate-qa.log |
| mistralrs-quant-c, all 10 MMQ types incl. packed-stride MoE | **PASS 4940 calls, 9360379960 bytes, 0 failing** | redcell2 4940 / 0 | out/gate-qc.log |
| fmt_mmq iq4_nl incl. MoE mode (K = 704) | **PASS 330 launches, 198296952 bytes, 0 failing** on the committed PTX (window 2) | redcell2 330 / 0 | out/gate2-fm.log |
| (fmt_mmq on window 1's regenerated PTX) | not loadable (cuModuleLoadData 218). **Harness cause, not the merge:** lib.sh builds every crate with `--arch sm_120`; fmt_mmq needs `sm_120a` (its b.sh; committed PTX is `.target sm_120a`): ptxas: `mma with block scale not supported on .target sm_120`. redcell2's window 1 hit the same | | out/gate-fm.log, out/fmregen/ptxas.log |
| iq3_s (regenerated and committed) | **PASS 1085 launches, 0 failing** | same | out/gate-iq3s.log, gate2-iq3s.log |
| iq-moe (regenerated and committed) | **PASS 208 launches, 457904 B, 0 failing** | same | out/gate-iqmoe.log, gate2-iqmoe.log |
| 35B MTP=2 vs m6/out/h-off.json | **40/40**, 111.7 tok/s | 40/40 | m3/out/integ3-m2.json |
| 35B MTP-off (MTP-dir Q4_K_XL, TITAN_MTP=0) vs m6/out/mtpfile-off.json | **40/40**, 86.1 tok/s | 40/40 | m3/out/integ3-mf0.json |
| IQ2_M 8x300 vs integ2's i-new (and integ's) | **8/8** (8/8), 57.9 tok/s | | out/i-new.json |
| Bonsai-27B Q1_0 8x300 vs integ2's q-new (and integ's) | **8/8** (8/8), 61.0 tok/s | | out/q-new.json |
| Bonsai-2 B2-off vs o-off | **8/8**, 64.2 tok/s | | out/b-off.json |
| Spark G2 / G1 / G3 vs integ2 (integ order g5 x3, g3, g1, g2) | **G2 tokens + text 40/40 identical; G1 top-20 40/40; G3 4/4** identical; G1 vs llama 39/40, dlp 0.0124, KL 0.00371 | integ2 same | out/sp-*.json |
| Spark decode ~550 / ~4.1k / ~9k | 122.5 / 127.1, 122.4 / 130.4, 116.4 / 116.9 tok/s; prefill 5.6-6.3k | integ2 121.9-127.1 / 115.4-122.4 / 116.4-116.7 | out/sp-g5-*.json |
| deq: Spark + LoRA (`--lora spark=spark-lora-r16`), G6 off/on and G1 rows off/on vs integ2's recorded run (m4/deq/out/spl-old-*) | **identical** top-20 / tokens / text 24/24, 24/24, 40/40, 40/40 | deq: identical | out/spl-*.json |
| gemma4-12b G1 (8192, 0:48, bf16, default cache) vs g4acc final (m4/g4acc/out/nb-g1, bin/mistralrs-w4) | **top-20 / tokens / text identical 40/40**; vs llama 40/40, dlp 0.0030, KL 0.00060 | w4: 40/40 identical | out/g12-g1.json |
| gemma4-12b G2 vs nb-g2 | **40/40 tokens and text identical**; vs llama 5/40 identical (= w4) | | out/g12-g2.json |
| gemma4-12b G3 ~4k / ~8k (pc 0, 12288) vs g4acc p0-g3 / p0-g3l | **4/4 and 4/4 identical**; vs llama 4/4, dlp 0.0311, KL 0.01118 / 3/4, 0.0598, 0.01056 | same | out/g12-g3.json, g12-g3l.json |
| gemma4-12b G3 vs llama.cpp's own spread (g3spread.py, m4/g4acc/out lub256 / lub128 / lb512) | ~4k **WITHIN**; ~8k **OUTSIDE by KL** (0.01056 > worst variant 0.01030), exactly as g4acc's P0; absolute 0.05 / 0.01: FAIL both (as every llama.cpp variant) | g4acc same | window2 log |
| gemma4-12b prefill ~4k / ~8k (G3 rows) | 3204 / 3156 / 3152 / 3055; 3003 / 2937 / 2872 / 2829 tok/s (medians **3154 / 2905**) | g4acc 3210 / 2972 | |
| gemma4-12b decode (G5, pc 0) ~540 / ~3.5k / ~7.3k | **72.7 / 73.1, 71.3 / 71.6, 69.7 / 70.1** tok/s | w4 72.0 / 72.3, 70.5 / 71.7, 69.6 / 70.6 | out/g12-d*.json |
| REDCELL G1 vs llama | **39/40, dlp 0.0084, KL 0.00078**; top-20 logprob values **identical 40/40 to redcell2's dz-a** | redcell2 39/40, 0.0084, 0.00078 | out/rz-a-g1.json |
| REDCELL determinism, 2 fresh servers, pc 0, G1 -> G3 -> G2 256 | **G1 40/40, G3 4/4, G2 40/40 identical a vs b**; 0 guard lines, 0 OOM | redcell2: same | out/rz-{a,b}-*.json |
| REDCELL vs the redcell2 frozen run (dz-a), same procedure | G1 40/40 identical; **G3 2/4** (the 3547 / 3532-token prompts differ, max top-20 delta 0.55 / 0.43; the 4107 / 4652 prompts identical); **G2 2/40** | | see bisect |
| ... same with `TITAN_G4_FLASH_DECODE=0` (rz-e) | G1 40/40, G3 2/4, **G2 40/40 identical to dz-a** | | out/rz-e-*.json |
| **bisect**: integ3 with the g4acc merge reverted (446fedf94, branch `integ3-bisect-nog4acc`, binary out/mistralrs-bis 148b5662) | **G1 40/40, G3 4/4, G2 40/40 identical to redcell2's dz-a** | | window3 log, out/bz-*.json |
| REDCELL G3 ~4k vs llama (pc 0) | 4/4, dlp **0.0666**, KL **0.0223**: within llama.cpp's spread (worst variant 0.164 / 0.0328, redcell2 out/spread) | redcell2 grouped 0.1048 / 0.0209 | |
| REDCELL G2 vs llama | 2/40 identical (redcell2 dz-a: 0/40) | | |
| REDCELL G5 pc 0, 3547 / 3532 | **prefill 5432 / 4954 tok/s, decode 115.0 / 116.7**; ~540: 3335 / 3567, 123 / 124 | redcell2 4473 / 4286, 77.5 / 81.7 | out/rp-g5*.json |
| REDCELL OOM recovery, default prefix cache | **no panic, 3 OOM requests fail cleanly, server keeps serving** (above); redcell2 binary panics twice | | above |
| REDCELL default-cache G2 vs m4/g4s/out/baseline-redcell-g2.json | 4 of the 38 successful rows identical (g4acc's flash decode changes G2; the baseline was made with eager decode) | redcell2 dd (piece 1024): 40/40 | out/oom-new-g2.json |
| IQ3_S 9B IQ3_M and embIQ3S e2e vs integ2 | **greedy 8/8 texts and TF 8/8 byte-equal, both files** | | iq3s/out/*_mistral.json |
| pcache swap, deploy roster on port 18640: 35B -> gemma4-12b 6 x ~4k -> redcell-26b 3 x ~4k -> 35B | swaps 3.6 s / 6.6 s / 35.1 s; **gemma4 6/6, 0 OOM, max 10058 MiB**; **REDCELL 3/3, 0 OOM, max 14122 MiB**; 35B repeat TTFT median **0.43 s before, 0.35 / 0.35 s after**, decode 138.7-139.2 tok/s; `Prefix caching enabled` only on the two 35B loads; gemma4 top-20 4/4 = g12-g3, REDCELL 3/3 = rz-a-g3 | integ2: 1.20 s then ~0.3-0.4 | out/*.pc.json, g12-pc0.json, red-pc0.json, roster-p.server.log |

### What differs from a branch result, and why (bisected)
REDCELL shares gemma4's text tower, so g4acc's attention changes (r2 GQA-2 kernel for sliding layers, no host masks, bf16
decode on the flash kernels) reach it. redcell2 was gated before g4acc existed:
- **G2** (decode): with `TITAN_G4_FLASH_DECODE=0` integ3's G2 is 40/40 identical to redcell2; with the default flash decode
  2/40 - g4acc's decode kernel, as gemma4-12b's own G2 changed on the g4acc branch (2/40 identical to the previous binary).
- **G3** prompts 0-1 (3547 / 3532 tokens): integ3 minus the g4acc merge reproduces redcell2 **4/4**; integ3 2/4. g4acc's own
  README: at ~4k the r2 kernel's block count picks a different split count (gemma4-12b ~4k changed 0.0517 -> 0.0311 the same
  way). The result is closer to llama.cpp (dlp 0.0666 vs 0.1048) and inside its spread.
- Nothing else differs from its branch: gemma4-12b is identical to the g4acc final binary in G1 / G2 / G3 / G3l, Spark and the
  regression core to integ2, Spark LoRA to integ2 (deq), REDCELL G1 to redcell2. No merge was reverted in the result.

## Deploy dry run (window 2, `[D]`; window 3 `[A]`)
`m4/integ3/models.toml` = the main worktree's current `deploy/models.toml` + `m4/integ2/models-toml-draft.diff` (applied with
`patch -p2`, clean). Served as `roster-dry.toml` (= models.toml with `port = 18640`, the only difference) with the service's
exact ExecStart (`<binary> from-config --file <toml>`), `LD_LIBRARY_PATH=~/titan-engine/lib`, MemoryMax=20G,
MemorySwapMax=0, working directory `$HOME`, on a fresh server; every entry swapped in by name with one chat request ("In two or
three sentences, explain why the sky is blue.", max_tokens 400, temperature 0). **All 12 model files present; 12/12 replied
coherently; 0 OOM, 0 panics.**
| model | load (s) | VRAM free after load | VRAM used after reply | decode tok/s | reply |
|---|---|---|---|---|---|
| qwen3.6-35b (default) | 31.1 (startup) | 2365 MiB | 13896 MiB | 110.3 | thinking, coherent (400-token cap reached while thinking) |
| qwen3.6-35b-mxfp4 | 30.8 | 2709 | 13290 | 85.7 | thinking, coherent (cap) |
| qwen3-next-80b | 22.3 | 1313 | 14716 | 19.7 | coherent answer (Rayleigh, violet vs eye sensitivity), stop |
| gpt-oss-120b | 9.2 | 1953 | 14076 | 5.0 | coherent reasoning + answer, stop |
| gpt-oss-20b | 15.6 | 449 | 15484 | 88.2 | coherent reasoning + answer, stop |
| bonsai-27b | 3.2 | 10582 | 5436 | 63.2 | thinking, coherent (cap) |
| qwen3-14b | 11.1 | 6625 | 9500 | 18.5 | coherent reasoning + answer, stop |
| qwen3.6-35b-iq2m | 6.2 | 4225 | 11881 | 59.8 | thinking, coherent (cap) |
| bonsai2-27b | 5.5 | 8268 | 7838 | 83.9 | coherent reasoning + answer, stop |
| spark-x2.5 | 1.4 | 13023 | 3070 | 142.5 | coherent reasoning + answer, stop |
| gemma4-12b | 4.1 | 8863 | 7422 | 79.5 | thinking, coherent (cap) |
| redcell-26b | 6.7 | 3839 | 12318 | 142.2 | thinking, coherent (cap) |
Replies: out/dryrun.json; server log out/dry.server.log.
- qwen3-14b (18.5 tok/s) and gpt-oss-120b (5.0 tok/s) looked slow, so window 3 A/B'd them, two requests each, same roster,
  integ3 vs integ2: qwen3-14b 18.6 / 18.8 vs 18.8 / 18.7, gpt-oss-120b 4.9 / 5.3 vs 5.1 / 5.1 tok/s (out/ab-{new,old}.json).
  **Same on integ2: not a regression of this merge**, a property of those roster entries (not investigated here).

## Not done / open
- G3's absolute limits (0.05 / 0.01) still fail for gemma4 and REDCELL as on every branch; the user has not decided on the
  spread-based gate (g3spread.py). gemma4 ~8k is outside llama.cpp's spread by KL (0.01056 vs 0.01030), unchanged from g4acc.
- The REDCELL default-cache OOMs themselves are not removed (they now fail one request instead of killing the engine); the
  roster's `prefix_cache_n = 0` for REDCELL avoids them.
- lib.sh's oxide build uses `--arch sm_120` for every crate; fmt_mmq needs `sm_120a`. Fix the harness before regenerating
  fmt_mmq's PTX (the committed PTX is the gated one).
- `m4/g4s/out/baseline-redcell-g2.json` and redcell2's dz-a G2 predate g4acc's flash decode; a new REDCELL G2 baseline would be
  rz-a-g2.json if the decode change is accepted.
- Windows: 3 of 3 (29.7 + 15.5 + 12.1 min).

## Files
`lib.sh` (m4/integ2's, retargeted; `cfg_start` takes `CWD`), `window{1,2,3}.sh` + `.nohup`, `oomgate.py`, `dryrun.py`,
`pcache.py`, `b2client.py`, `same.py`, `iq3s/`, `models.toml` (to deploy), `roster-dry.toml`, `out/`.
Left uncommitted: `out/regen/`, `out/fmregen/fmt_mmq.ptx`, `out/mistralrs-bis`, `out/metadata-all.json`.
Apply with `cp m4/integ3/models.toml ~/titan-engine/deploy/models.toml` (the main session's call).
