# integ2-20261003: integ-20261002 + pcache + emb + hd512 + redcell in one build (staged, NOT deployed)

Binary: `bin/mistralrs-titan-integ2`, sha256 `bca90161bd162eab5c7e14b446a9575952d876f894197c5b1cbd4a94f6371055`
(nvcc-free `cargo build --release -p mistralrs-cli --features oxide`, CARGO_TARGET_DIR `target-integ2-oxide` seeded from
target-integ-oxide, TITAN_OXIDE_DIR `oxide-integ2`). `bin/mistralrs-titan-swap`, `bin/mistralrs-titan-integ`,
`deploy/models.toml` and titan-mistral are untouched. Nothing pushed.

## Branch heads (all `integ2-20261003`, from `integ-20261002`)
| repo | worktree | head | contents |
|---|---|---|---|
| oxide-kernels | `oxide-integ2` | 61b36fa | merge hd512 be03263 (906db68), merge redcell 2695cb5 (61b36fa). Disjoint files (flash-prefill vs mistralrs-quant-a + titan-oxide-ffi/quant_a.rs), no conflicts, committed PTX of each branch kept |
| titan-engine (candle + m4) | `top-integ2` (sparse: candle, m4) | this README commit, on merges 04d2bad6 (emb, candle 3f389945..e55afc23 + evidence 939568ec), 6e480d8e (pcache c5d1a575), 3321555f (hd512 9aa83361), 704ebe7c (redcell 12c93b7f) | auto-merged |
| mistral.rs | `top-integ2/mr-integ2` | 262364180 | merges pcache 2a300c23c (59e5f04b3), emb a763a3f1e (998f234aa), hd512 4aa8b028a (0efe3e8a8), redcell 8830aa217 (65124fe3f), then 262364180 (below) |

`top-integ2/oxide-kernels -> ../oxide-integ2` is an untracked symlink (as top-integ).

### Merge notes
- All merges were textual auto-merges, including `vision_models/gemma4/text.rs`: hd512's attention edits, emb's removal of
  gemma4's own `qtensor_embedding_rows` call in `embed_tokens`, and redcell's 6 MoE probe lines are all present (union).
  `qtensor_embedding_rows` is gone from the whole tree; `mistralrs-quant/src/lib.rs` carries emb's removal and redcell's
  `grouped_moe_llama_mmq_from_glu_packed` export.
- **Build failure found (not a merge conflict): emb's a763a3f1e as committed does not compile.** It drops
  `QEmbedding::hidden`, but the MTP-head loader (`load_mtp_block`, from e449283c5 in the base) still reads `e.hidden`
  (`error[E0609]`, out/build-integ2-window1.log, first build attempt). The emb agent's worktree `top-emb/mr-emb` still has
  the 3-line fix uncommitted (file mtime 21:34:02 = the commit time; `m4/emb/mistralrs-emb` built 21:37 from that tree),
  so the emb binary and its gates include it. 262364180 applies exactly that uncommitted hunk (`git -C top-emb/mr-emb
  diff | git apply`); nothing else. The emb branch itself still needs the same commit.
- Paths: `git grep 'top-emb\|top-hd512\|top-redcell\|top-pcache\|oxide-hd512\|oxide-redcell'` in mr-integ2 is empty.
  Candle `[patch]` is `../candle`, the three `titan-oxide-ffi` paths are `../../oxide-kernels`. `cargo metadata --offline`
  resolves candle-core / candle-nn to `top-integ2/candle` (out/metadata.json), and with `--all-features` titan-oxide-ffi to
  `top-integ2/oxide-kernels/titan-oxide-ffi` = `oxide-integ2/titan-oxide-ffi` (out/metadata-all.json).
- Embedded PTX: mr-integ2 `flash_prefill_oxide.ptx` == oxide-integ2 committed `flash-prefill/flash_prefill.ptx`
  (CR-normalised, = hd512's); quant-a PTX is incbin'd from oxide-integ2 (= redcell's committed file). Regenerating the PTX
  through the crates' `cargo oxide build` gives code that differs textually from the committed files for all four crates
  (the known worktree-path crate hash in mangled names / emission order, integ README); the gates pass on both, and the
  committed files were restored before the build.

## Gate table
Windows: window1-20261003-0023.log (23 m), window2-20261003-0047.log (10 m), window3-20261003-0057.log (7 m). titan-spark
stopped after taking the lock and restarted by the EXIT trap each time; titan-mistral never started. No TITAN_* in the
window environment (`TITAN_MOE_UNALIGNED_GROUPED` unset everywhere; the non-finite guard of the grouped route logged 0 lines).

| gate | integ2 result | branch-level reference | evidence |
|---|---|---|---|
| oxide flash-prefill (regenerated PTX) | PASS: d256 window 16187392 outputs worst 0.602 of tol, win+1 6/6, past-1 8/8, vs v1 0 of 53100544 differ, splits worst 0.667; **hd512 gate512 PASS**: d512 bf16 variants 0 of 35012608 differ, mutations 18/18, f16 d512 8/8, f16 d256 8/8; **head-dim-256 bf16 kernels vs the integ PTX: 148 calls, 0 of 253755392 outputs differ** | hd512: same counts | out/gate-flash-prefill.log |
| same, committed PTX | PASS, same numbers | | out/gate2-flash-prefill.log |
| oxide mistralrs-quant-a vs libmistralrsquant.a | **PASS 7004 launcher calls, 572862725 bytes, 0 failing** (incl. redcell's slot-order and stable-dispatch checks; summary lines identical to redcell's gate-qa.log) | redcell: 7004, 0 failing | out/gate-qa.log |
| oxide iq-moe (IQ2_XXS/XS/S, IQ3_XXS, IQ4_XS 32 each, IQ4_NL 48) on the merged quant-a PTX | PASS, 208 launches, 457904 B, 0 failing (regenerated and committed) | integ: same | out/gate-iqmoe.log, gate2-iqmoe.log |
| oxide iq3_s (dequantize x3, mmvq x9 vs llama.cpp acecd56 cubins) | PASS, 1085 launches, 15608544 B, 0 failing (committed and regenerated PTX) | integ: same | out/gate3-iq3s.log, gate3r-iq3s.log (window 1 failed to start: the untracked nvcc reference cubins `iq3_s/ref/*.cubin` are not in a fresh worktree; copied from oxide-integ, re-run in window 2) |
| config unit tests (`config::tests`) | **11/11 pass** | pcache 11/11 | out/test-cli.log |
| 35B MTP=2 vs m6/out/h-off.json | **40/40** (105.1 tok/s, see note) | 40/40 | m3/out/integ2-m2.json (copy in out/) |
| 35B MTP-off, MTP-dir Q4_K_XL, TITAN_MTP=0 vs m6/out/mtpfile-off.json | **40/40** (87.5 tok/s) | 40/40 | m3/out/integ2-mf0.json |
| Bonsai-2 B2-off vs m4/bonsai2/out/o-off.json | **8/8**, 63.7 tok/s | 8/8 | out/b-off.json |
| IQ2_M 8x300 vs bin/mistralrs-titan-integ's run (i-integ) | **8/8** (and 8/8 vs b2merge's i-old), 57.6 tok/s | 8/8 | out/i-new.json |
| Bonsai-27B Q1_0 8x300 vs integ's run (q-integ) | **8/8** (8/8 vs q-old-b2merge), 60.4 tok/s | 8/8 | out/q-new.json |
| Spark Q4_K_M G2 (integ order g5 x3, g3, g1, g2) | **tokens + text 40/40 identical to emb's sp-g2 and to integ's sp-g2**; vs llama baseline 39/40 | emb: 40/40 = integ | out/sp-g2.json |
| Spark G1 / G3 | top-20 40/40 / 4/4 identical to emb; vs llama G1 39/40, dlp 0.0124, KL 0.00371; G3 4/4, 0.0296 | same | out/sp-g1.json, sp-g3.json |
| Spark decode tok/s, ~550 / ~4.1k / ~9k | 121.9 / 127.1, 115.4 / 122.4, 116.4 / 116.7 (emb: 122.8 / 127.1, 123.1 / 133.3, 117.8 / 119.4); prefill 5.6-6.1k (one 1659 outlier at 4111 tokens, integ had a 2879 one) | ~120+ | out/sp-g5-*.json |
| gemma4-12b G1 (8192, 0:48, bf16) | **40/40, dlp 0.0030, KL 0.00060; top-20 identical 40/40 to integ and to hd512's f-g1** | 40/40 | out/g12-g1.json |
| gemma4-12b G3 ~4k, --prefix-cache-n 0 | 4/4, **0.0517**, KL **0.0158**; **4/4 identical to hd512's f-g3** | hd512: 4/4, 0.0517, 0.0158 | out/g12-g3.json |
| gemma4-12b G3 ~8k, --prefix-cache-n 0 | 3/4, **0.0598**, KL **0.0106**; **4/4 identical to f-g3l**; 0 OOM, 13.7 GB | hd512: 3/4, 0.0598, 0.0106 | out/g12-g3l.json |
| gemma4-12b prefill tok/s ~4k / ~8k (G3 rows) | 2733 / 2770 / 2623 / 2652; 2462 / 2500 / 2313 / 2255 | hd512 f-g3: 2708-2530; f-g3l: 2370-2218 | |
| REDCELL determinism, 2 fresh servers, baseline procedure (default cache, G1 -> G3 -> G2) | **G1 top-20 40/40, G2 text 40/40 identical a vs b**; G3's first request OOMs on both (as recorded for the baseline) | redcell: 40/40, 40/40 | out/fd-{a,b}-g{1,2}.json |
| REDCELL G2 vs m4/g4s/out/baseline-redcell-g2.json (recorded order) | **40/40 tokens and text, both servers** | redcell: = by construction | out/fd-a-g2.json |
| REDCELL G1 vs llama.cpp ref-red-g1 | 39/40, dlp 0.0140, KL 0.00085; top-20 40/40 identical to redcell's fd-a-g1 | redcell: 39/40, 0.0140, 0.00085 | |
| REDCELL G3 ~4k, per-model prefix_cache_n = 0 (from-config, draft roster) | 4/4, dlp 0.0614, KL 0.0264; **top-20 identical 4/4 to the same binary with CLI --prefix-cache-n 0** (rp-g3); 6/6 ~4k ok, 0 OOM, max 14.6 GB | not re-measured on redcell's final build; hd512 flash-on runs (nondeterministic then): dlp 0.096-0.142, KL 0.0068-0.0259 | out/red-pc0.json, red-pc0-g3.json, rp-g3.json |
| REDCELL G5 prefill / decode, --prefix-cache-n 0, default route | 3547 / 3532 tokens: **640 / 637 prefill**, 79.3 / 80.6 decode; ~540: 644 / 642, 97 / 98 | redcell default-route 582 (TITAN_MOE_UNALIGNED_DECODE=1); integ 558 / 567; hd512 625 | out/rp-g5.json, rp-g5s.json |
| pcache swap, combined draft roster via from-config: 35B -> gemma4-12b (6 x ~4k) -> 35B | swaps 4.8 s / 29.7 s; **gemma4 6/6, 0 OOM**, max 10154 MiB; gemma4 top-20 **4/4 identical to hd512's f-g3** and to this binary's CLI pc-0 G3; `Prefix caching enabled` only for qwen3.6-35b (load and reload), not gemma4-12b / redcell-26b | pcache 6/6, 10218 MiB | out/g12-pc0.json, roster.server.log |
| 35B repeat TTFT (pc.py, 6144 cached each) | before 0.43 s median; after the swap 1.20 s median in window 2 (last three 0.58 / 1.23 / 0.67). **Window 3 A/B, interleaved, same roster** (pcache branch binary vs integ2, twice each): after-swap medians pcache 0.34 / 0.49 s, integ2 0.64 / 0.44 s; a second round right after: pcache 0.35 / 0.35, integ2 0.30 / 0.42 s. Noise of the first few post-reload requests, same on both binaries; settles at ~0.3-0.4 s | pcache: 0.87 s, last three ~0.5 s | out/*q35-*.pc.json, window3 log |
| IQ3_S 9B IQ3_M e2e | greedy 8/8 texts and TF 8/8 byte-equal to **both** the iq3s branch and integ; cmp vs llama identical: greedy 2/8, TF top-1 500/509, in top-5 509/509, dlp mean 0.054 | same | iq3s/out/iq3m_*.json |
| IQ3_S 9B, IQ3_S token_embd | 8/8 and 8/8 equal to both; 2/8, 441/452, 452/452, dlp 0.046 | same | iq3s/out/embiq3s_*.json |

Note: the 35B MTP=2 collect throughput (105.1 tok/s) is below integ's 116.1 and redcell's 110.6 / hd512's 110.9 on the
same harness, with byte-identical outputs. Decode on the service env in the same windows is unchanged (pc.py decode median
139-141 tok/s, pcache 136-139), so this looks like run-to-run variance of the tiered host path (6 GB free RAM at window
start), not a code change; it was not investigated further.

Nothing differed from its branch-level result, so no merge was reverted / bisected.

## G3 thresholds
G3's absolute limits (median |dlogprob| < 0.05, top-20 KL < 0.01) are below llama.cpp's own bf16 noise floor on gemma4-12b
(m4/hd512/README.md: llama.cpp with a bf16 KV cache scores 0.0708 / KL 0.020 against its own f16 run, FA off 0.0995 /
0.020), so gemma4-12b ~4k (0.0517 / 0.0158) and ~8k (0.0598 / 0.0106), and REDCELL ~4k, still "fail" them exactly as on the
hd512 branch. **The user has not yet decided whether to re-base G3 on a noise floor**; the results above are reported against
the current limits and against the branch numbers.

## Combined draft deploy/models.toml diff (NOT applied)
`models-toml-draft.diff` = m4/integ/models-toml-draft.diff (bonsai2-27b, spark-x2.5, gemma4-12b, redcell-26b) +
m4/pcache/models-toml-draft.diff (`prefix_cache_n = 0` under `[models.titan]` for gemma4-12b and redcell-26b, comments).
pcache's diff already contained integ's, so the combined diff is byte-identical to m4/pcache/models-toml-draft.diff; the
35B and `[runtime]` keep the global 16. It applies cleanly (`patch -p1 --dry-run`) to the main worktree's current
deploy/models.toml. Validation:
- tomllib: 12 models; `prefix_cache_n` 0 for gemma4-12b and redcell-26b only.
- **from-config, served on a scratch port**: `roster-draft.toml` = the patched file with only `port = 18640`. This binary
  parsed it (`titan swap mode: 12 models, default qwen3.6-35b`), loaded the 35B, swapped to gemma4-12b and redcell-26b with
  the per-model 0 in effect (table above). Negative control: `roster-bad.toml` (prefix_cache_n = -1) is refused by the
  parser: `Error: Failed to parse TOML config file`, rc 1 (out/roster-bad.log).
- config unit tests 11/11 (incl. the three per-model prefix-cache tests).
Apply with `patch -p1 < m4/integ2/models-toml-draft.diff` in ~/titan-engine (the user's call).

## Not done / open
- emb branch: commit the `QEmbedding::hidden` fix on `emb` itself (see merge notes); integ2 carries it as 262364180.
- G3 limits (above); REDCELL batched long-prompt prefill stays opt-in and known wrong (TITAN_MOE_UNALIGNED_GROUPED=1,
  untouched here); pcache (b) pool fragmentation not fixed; the integ README's other known bugs stand (the i-quant CPU
  dequantize ones are fixed by emb).
- Windows: 3 of 3 used (23 + 10 + 7 min).

## Files
`lib.sh`, `window{1,2,3}.sh` (+ `.nohup` copies of the logs), `pcache.py` (m4/pcache client writing here), `b2client.py`,
`same.py`, `iq3s/` (e2e.py + llama references), `roster-draft.toml`, `roster-bad.toml`, `models-toml-draft.diff`, `out/`.
Regenerated PTX (`out/regen/`) and `out/metadata*.json` are left uncommitted (17 MB).
