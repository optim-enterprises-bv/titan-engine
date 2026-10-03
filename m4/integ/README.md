# integ-20261002: today's titan-engine branches in one build (staged, NOT deployed)

Binary: `bin/mistralrs-titan-integ`, sha256 `c423c6a4aa04a2515998f65849e97dedee2c500d4e88747edbf58c6be8045ba9`
(nvcc-free `cargo build --release -p mistralrs-cli --features oxide`, CARGO_TARGET_DIR target-integ-oxide, TITAN_OXIDE_DIR=oxide-integ).
`bin/mistralrs-titan-swap`, `deploy/models.toml` and titan-mistral are untouched. Nothing pushed.

## Branch heads (all `integ-20261002`)
| repo | worktree | head | contents |
|---|---|---|---|
| oxide-kernels | `oxide-integ` | 34f0c5a | glu-typed-kernels 12c7422 + iq3s 22df5e4 + g4s-iq4nl-moe 3ac0b9a + spark-swa 0a716a1; no conflicts (disjoint files) |
| titan-engine (candle) | `top-integ` (sparse: candle/) | cab3e51e (+ this README commit) | master 8472cb50 + ptq1_0 9a46bac8 + iq3s 7e00bc86; auto-merged, arms union |
| mistral.rs | `top-integ/mr-integ` | 23296a4b8 | model-swap 0415d217a + swap-b2 c5e810541 + iq3s 2aeedfb6a + gemma4-spark a7ea678f9 + spark-prefill e83baa8f1, then d7d752e1e and 23296a4b8 |

Merge work in mistral.rs:
- iq3s: textual conflict in `mistralrs-quant/src/gguf/mod.rs` (uqff code / label tables): both arms kept (PTQ1_0 143 and IQ3S 21).
- d7d752e1e: candle patch back to `../candle` (was `../top-iq3s/candle`), the three `titan-oxide-ffi` paths back to
  `../../oxide-kernels` (were `../../oxide-g4s`); `top-integ/oxide-kernels -> ../oxide-integ` is an untracked symlink (as top-b2).
  `21 => IQ3S` added to `GgufDType::candle_dtype` (gemma4-spark mapped the five i-quants, swap-b2 PTQ1_0, nobody IQ3_S).
  `git grep 'candle-b2\|top-iq3s\|oxide-g4s\|oxide-sparkpf'` is empty. `cargo metadata --offline` resolves candle-core/candle-nn to
  `top-integ/candle` (out/metadata.json).
- 23296a4b8: semantic conflict, not textual: swap-b2's flash-prefill GQA padding recursed with `attend(&qp, k, v, &pp)` while
  spark-prefill added a `win` argument. Fixed by passing `win` through (the padding only adds zero query heads). This build
  error ended window 1 after 5 min.
- Embedded PTX: `ptq1_0_{mmq,mmvq}_oxide.ptx` == swap-b2, `flash_prefill_oxide.ptx` == spark-prefill (equal to oxide-integ
  `flash-prefill/flash_prefill.ptx` except for CR bytes inside `//` comments that git normalises), candle `iq3_s_oxide.ptx` ==
  oxide-integ `iq3_s/iq3_s.ptx`, quant-a PTX (incbin via titan-oxide-ffi) == g4s-iq4nl-moe.

## Gate table
| gate | result | branch-level reference | log |
|---|---|---|---|
| oxide iq3_s (dequantize x3, mmvq x9 vs llama.cpp acecd56) | PASS, 1085 launches, 15.6 MB, 0 differing; 4/4 mutants caught | same | out/gate-iq3s.log, out/gate2-iq3s.log |
| oxide iq-moe (IQ2_XXS/XS/S, IQ3_XXS, IQ4_XS 32 launches each, IQ4_NL 48) | PASS, 208 launches, 457904 B, 0 failing | same | out/gate-iqmoe.log, out/gate2-iqmoe.log |
| oxide flash-prefill | PASS; window 16187392 outputs worst 0.602 of tol, mutation 6/6; win=0 vs v1 0 of 53100544 differ; splits 0.667 | same | out/gate-flash-prefill.log, out/gate2-flash-prefill.log |
| 35B MTP=2 vs m6/out/h-off.json | **40/40** (116.1 tok/s) | 40/40 | window2 log, m3/out/integ-m2.json |
| 35B MTP-off, MTP-dir Q4_K_XL, TITAN_MTP=0 vs m6/out/mtpfile-off.json | **40/40** (87.0 tok/s) | 40/40 | m3/out/integ-mf0.json |
| IQ2_M new vs old bin/mistralrs-titan-swap, 8x300 | **8/8** (and 8/8 vs b2merge's recorded old run); 58.0 vs 57.4 tok/s | 8/8 | out/i-new.json, out/i-old.json |
| Bonsai-27B Q1_0 new vs old, 8x300 | **8/8** (8/8 vs recorded); 61.1 vs 60.4 tok/s | 8/8 | out/q-new.json, out/q-old.json |
| Bonsai-2 B2-off vs m4/bonsai2/out/o-off.json | **8/8**, 64.4 tok/s | 8/8 | out/b-off.json |
| Bonsai-2 B2-mtp1 vs o-off | **8/8**, 79.8 tok/s, MTP acceptance 80.2% | 8/8 | out/b-n1.json |
| Spark Q4_K_M G1 vs refc-spark-g1 | 39/40, median dlp 0.0124, KL median 0.00371; **40/40 top-20 identical to spark-prefill's t-d0-g1** | 39/40, KL 0.0037 | out/sp-g1.json |
| Spark G3 vs refc-spark-g3 | 4/4, median dlp 0.0296; **4/4 identical to t-d0-g3** | 4/4, 0.030 | out/sp-g3.json |
| Spark G2 vs baseline-spark-x2.5-q4km-g2 | 39/40 (prompt 11); **40/40 identical to t-d0-g2** | 39/40 | out/sp-g2.json |
| gemma4-12b G1 (8192, 0:48, bf16) vs ref-g12-g1 | **40/40**, median dlp 0.0030, KL 0.00060; 40/40 identical to the clean a7ea678f9 build and to w5 | see note 1 | out/g12-g1.json, out/g12b-g1.json |
| gemma4-12b G3 ~4k, prefix-cache-n 0 | 4/4, dlp 0.0825, KL 0.0150; **identical to w6 t-g12p-g3**; no OOM, 10.0 GB VRAM | same | out/g12-g3.json |
| gemma4-12b G3 ~8k, prefix-cache-n 0 | 3/4, dlp 0.0615, KL 0.0187; **identical to w6 t-g12p-g3l**; no OOM, 14.1 GB VRAM | same | out/g12-g3l.json |
| REDCELL G1 (prefix-cache-n 0) vs ref-red-g1 | 39/40, median dlp 0.0098-0.0100, KL 0.00091 | branch build: 39/40, KL 0.00068 | out/red-g1.json, out/ri-g1a.json, out/rb-g1a.json |
| REDCELL G2, baseline procedure vs baseline-redcell-g2 | 2/40 and 4/40 (two runs) | **branch build: 1/40 and 2/40**; see note 2 | out/redb-g2.json, out/rip-g2.json, out/rbp-g2.json |
| IQ3_S 9B IQ3_M e2e (vs llama.cpp) | greedy 2/8, TF top-1 500/509, in top-5 509/509, dlp mean 0.054 | identical outputs to the iq3s branch (greedy 8/8, TF 8/8 byte-equal) | iq3s/out/iq3m_cmp.json |
| IQ3_S 9B, IQ3_S token_embd | greedy 2/8, TF top-1 441/452, 452/452, dlp mean 0.046 | identical to branch | iq3s/out/embiq3s_cmp.json |

Note 1: `bin/mistralrs-g4s-w6` (the gemma4-spark w6 binary, built 19:47) is not a7ea678f9 as committed (19:55): its tree still
had the `TITAN_G4_ATTN_F32` experiment on by default (its G1 median dlp 0.0019 is the commit message's ATTN_F32 number). So
`t-g12s6-g1`/`t-red6-g1` are not branch references. Window 3 built a7ea678f9 clean (`bin/mistralrs-g4s-a7ea`, sha256
a638a840..., scratch, not deployed) and integ equals it on gemma4-12b G1 (40/40 top-20 identical).

Note 2: REDCELL is not deterministic run to run, on the branch build as well as on integ: two fresh servers of the same binary,
prefix cache 0: G1 top-20 identical 30/40 (integ) and 36/40 (branch), first tokens 40/40; G2 256-token texts 1/40 (integ) and
1/40 (branch). baseline-redcell-g2.json (w5, before the unaligned-K routing fix) can therefore not be reproduced byte for byte by
any build; REDCELL needs a statistical gate. G3 under the default prefix cache still OOMs at its first ~3.5k request on both
builds (as recorded for the baseline); with prefix-cache-n 0 it completes (gemma4-spark w6).

## Throughput (integ binary)
- Spark-X2.5 Q4_K_M, ~9k prompts (G5 33000 chars, 8943 / 9291 tokens): prefill 5762 / 5634 tok/s, decode 90.9 / 91.3 tok/s
  (spark-prefill: 5911 / 5767, 92.0 / 91.3). ~4.1k: 2879 (one outlier) / 6029 tok/s, decode 92-103; ~550: 5979, 96.
- gemma4-12b, ~540-token prompts (G5 2000 chars): prefill 2925 / 2730 tok/s, decode 61.2 / 62.0 tok/s.

## prefix_cache_n = 0 and the 35B service (measured, window 2)
35B MTP-dir Q4_K_XL with the models.toml service env (tiered, MTP 2, PFS, doorbell, one-pass CPU, graphs), 65536, 8 chat
requests sharing a ~6.4k-token system prefix, 64 greedy tokens each (pc.py):
| prefix_cache_n | first request TTFT | 7 repeats TTFT median / sum | decode median |
|---|---|---|---|
| 16 (default) | 2.43 s | 0.53 s / 4.8 s (6144 tokens cached each) | 135.3 tok/s |
| 0 | 2.60 s | 2.93 s / 20.9 s | 133.1 tok/s |
So a global `[runtime] prefix_cache_n = 0` costs the 35B ~2.4 s per repeated-prefix request (5.5x TTFT on this workload);
decode is unaffected. Recommendation: do not set it globally; either serve gemma4/REDCELL long prompts from a separate process
or make prefix_cache_n per model (not implemented).

## Draft deploy/models.toml diff (NOT applied)
`models-toml-draft.diff` (against the main worktree's current, uncommitted deploy/models.toml): adds bonsai2-27b
(TITAN_MTP=1, TITAN_REASONING_EFFORT=medium, max_seq_len 65536), spark-x2.5 (max_model_len 16384), gemma4-12b (dtype bf16,
device_layers ["0:48"], max_model_len 12288) and redcell-26b (dtype bf16, device_layers ["0:30"], max_model_len 8192), each
idle_ttl 1800 with max_seq_len equal to max_model_len. Parses with tomllib and matches `config::ModelEntry`; not run through
from-config. No `[runtime] prefix_cache_n = 0` (see above).

## Known pre-existing bugs (not fixed here)
- candle `iquant.rs`: IQ3_XXS CPU dequantize indexes out of bounds for ib >= 4.
- candle `iquant.rs`: IQ2_XS CPU dequantize uses the wrong scale and applies no signs.
- candle `iquant.rs`: the cudarc impls lack `#[cfg(feature = "cuda")]`.
- REDCELL decode/prefill nondeterminism (note 2), present on the gemma4-spark branch.
- Regenerated oxide PTX is not byte-reproducible across worktree paths: the crate-hash in mangled names (Cs4Uqse.. vs
  Cs8iBaw..) and the function emission order change; the gates pass on both the regenerated PTX (window 1) and the committed
  PTX (window 2), and the build embeds the committed files.

## Windows (3 of 3 used; titan-spark stopped after taking the lock, restarted on exit; titan-mistral never started)
- window1-20261002-2003.log (5 m): metadata, oxide builds + gates PASS, mistral.rs build failed (note above).
- window2-20261002-2010.log (23 m): committed-PTX gates, build, every regression and model gate, prefix-cache measurement.
- window3-20261002-2034.log (16 m): clean gemma4-spark build, REDCELL determinism and baseline procedure, gemma4 G1 provenance.
