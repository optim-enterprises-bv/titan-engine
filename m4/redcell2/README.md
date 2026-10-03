# redcell2: REDCELL-26B batched (grouped MMQ) MoE prefill for unaligned K, FIXED and on by default

Branch `redcell2` off `integ2-20261003` in all three repos (staged, NOT deployed, nothing pushed).

| repo | worktree | head | contents |
|---|---|---|---|
| oxide-kernels | `oxide-redcell2` | 7251efd | quant-c MoE MMQ launcher honours `stride_col_dst` (72ce644); quant-c gate packed-stride MoE cases (7251efd); fmt_mmq gate MoE-mode arm (72ce644) |
| mistral.rs | `top-redcell2/mr-redcell2` | 39719cbdc | grouped route default for unaligned K, `TITAN_MOE_XCHECK` (4209cc1ad); non-finite guard opt-in (39719cbdc) |
| titan-engine | `top-redcell2` (sparse: candle, m4/redcell2) | this README | windows, scripts, gate outputs |

Final binary: `m4/redcell2/mistralrs-redcell2` = window-2 build of 39719cbdc, sha256 e992ce465971f828 (windows 2 and 3 ran it).
Server args everywhere: `--max-model-len 8192 --max-seq-len 8192 -n 0:30 --dtype bf16 --paged-attn off`.

## Root cause (why the grouped route returned NaN / wrong rows)
`fast_mmq::grouped_pair_packed` runs the gate and up MoE MMQs into ONE buffer, `[assignments, 2 * nrows]`: gate at column
offset 0, up at offset nrows, both with dst column stride `stride_col_dst = 2 * nrows`. The C launcher
(`mmq_gguf.cuh` `DEFINE_MMQ_MOE_LAUNCHER`) passes that stride as `nrows_dst`; its cuda-oxide twin (`mistralrs-quant-c/src/
launch.rs` `mmq_moe`, generated into `titan-oxide-ffi/src/quant_c.rs`) took the parameter as `_stride_col_dst` and used
`nrows_dst: nrows_x` (the dense launcher's rule). So in the nvcc-free build gate and up were written with stride nrows into a
2 * nrows buffer: they overwrote each other and half of the buffer was never written (uninitialised memory, hence the
intermittent NaN and the 0/4 G3 of redcell window 3). Every other MoE MMQ caller passes `stride_col_dst == nrows`, which is
why the 35B and the existing quant-c gate never saw it. Found by reading; the "Q3_K kernel" suspicion was the launcher, not
the kernel. candle's Q4_0 dequantize kernel (being fixed by `deq`) is not on this path: Q4_0 experts run MMQ / vec_dot on the
raw blocks.

Fix (72ce644): `mmq_moe` takes `stride_col_dst` and uses it as `nrows_dst`, exactly as the C launcher.

## Evidence on the model (window 1, `TITAN_MOE_XCHECK=1`: per layer, grouped vs per-token decode kernels on the same rows)
547- and 3547-token prompts, all 30 layers: max |grouped - per-token| 0.004-0.5 against max |per-token| 1.3-106 (the two
routes quantize activations differently: noise, not error), no non-finite row; the guard (window 1, on) fired 0 times in
every run. Log: window1-20261003-0535.log.

## Gates (final binary unless noted)
| gate | result | log / file |
|---|---|---|
| quant-c kernel gate, all 10 MMQ types (dense, MoE, and new: MoE with dst stride 2 * nrows vs the C result at stride nrows scattered, bit-exact) | **PASS 4940 launcher calls, 0 failing** (`mmq_*_moe_packed` 16 calls each) | out/gate-qc2.log |
| fmt_mmq kernel gate iq4_nl vs llama.cpp acecd56 `mul_mat_q` cubin, incl. new MoE-mode arm (ids_dst / expert_bounds, K = 704, 128-expert REDCELL-like routings, forced grids) | **PASS 330 launches, 0 failing** | out/gate-fm2.log |
| (window 1) the C reference itself at `stride_col_dst = 2 * nrows` | writes NaN into the gap rows (bounds rows by the stride): not usable as a packed reference, hence the scattered comparison | out/gate-qc.log |
| REDCELL G1 vs llama.cpp | **39/40, median dlp 0.0084, KL median 0.00078** (prompt 39: titan 'To' -0.643 vs 'A' -0.830, llama 'A' -0.589 vs 'To' -0.914) | window2 |
| REDCELL G3 ~4k (pc 0) vs ref-red-g3, against llama.cpp's own spread (g4acc g3spread: -ub 256 / -ub 128 / -b 512) | grouped **4/4, dlp 0.1048, KL 0.0209**; per-token (= integ2, bit-identical to its rp-g3) 4/4, 0.0614, 0.0264; llama.cpp variants 4/4, dlp 0.055-0.164, KL 0.0116-0.0328 (worst 0.164 / 0.0328): both titan routes WITHIN; grouped has the better KL, per-token the better dlp | window3, out/{tg,tq,pa,fq}-g3.json, out/spread |
| determinism, 2 fresh servers, prefix cache 0 (the roster's REDCELL setting), G1 -> G3 -> G2 256 | **G1 top-20 40/40, G3 top-20 4/4, G2 text 40/40 identical**; G3 also identical across windows 1, 2, 3 | out/dz-{a,b}-*.json |
| default prefix cache, baseline procedure (G1 -> G3 -> G2), TITAN_MOE_PIECE=1024 | G2 **40/40 = m4/g4s/out/baseline-redcell-g2.json**; G3's first request OOMs (as on every build) and the engine recovers | out/dd-*.json |
| G5 prefill / decode, pc 0, tok/s (prompt tokens) | **3547: 4473 / 77.5, 3532: 4286 / 81.7** (window 3: 4456 / 4240); 6790 / 6826: **3625 / 3678**; 547 / 535: 3335 / 3640, decode 100. Per-token route 649 / 647. **llama.cpp (same window, GPU, f16 KV): 3655 / 3703 prefill, 122 decode; 6.8k 3587 / 3594, 119** | out/pa-g5*.json, tg-g5, l-g5*.json |
| GPU profile, one 3547-token prompt (ncu) | **694 ms** (integ 5485 ms): MoE 441 (64%, Q3_K gate/up MMQ 304), attention 139 (flash-prefill), norms/copies 89 | out/ncu-final.csv |
| gemma4-12b G1 vs integ2 | **40/40 logprob-identical** | window2 |
| 35B MTP=2 vs m6/out/h-off.json | **40/40** (m3/out/rc2b-m2.json) | window2 |
| 35B MTP-off file vs m6/out/mtpfile-off.json | **40/40** (m3/out/rc2b-mf0.json) | window2 |
| Bonsai-2 B2-off vs o-off | **8/8** | window2 |
| IQ2_M 8x300 vs integ2's run (top-integ2/m4/integ2/out/i-new.json) | **8/8** | window2 |
| Bonsai-27B Q1_0 8x300 vs integ2's run | **8/8** | window2 |
(The regression core, gemma4 and REDCELL G1 also passed in window 1 on 4209cc1ad.)

## Defaults and switches (mistral.rs 39719cbdc)
- Unaligned-K prompts (>= 32 tokens) run the grouped route in <= 2048-token pieces (`TITAN_MOE_PIECE`; 1024 measured
  3881 / 3814 at 3.5k, slower). `TITAN_MOE_UNALIGNED_DECODE=1|iq4_nl|q4_0` puts all / those layers on the per-token kernels.
- `TITAN_MOE_NONFINITE_GUARD=1` (off: its F32 sum + sync was 19% of the 3.5k prefill, 0 firings), `TITAN_MOE_XCHECK=1` (debug).

## Caveats / not done
- Under the DEFAULT prefix cache with piece 2048, the G3 OOM (pre-existing at ~3.5k with a filled prefix cache) once hit an
  `unwrap()` in `kv_cache/mod.rs:542` and panicked the engine (window 1, fd-a); with piece 1024 it recovered (window 2). The
  integ2 roster serves REDCELL with prefix_cache_n = 0, where nothing OOMs. The unwrap is outside this branch.
- G3 ~9k (g3l) cannot run on REDCELL at max-model-len 8192 (prompts 9392 / 9631 tokens); llama.cpp's own g3l spread is in
  out/spread for reference.
- The 12-prompt CPU llama.cpp reference (g3x) was never completed (lock contention, then OOM at the 6 GB cap; stopped);
  titan's g3x runs for both routes are in out/{pa,pq}-g3x.json without a reference.
- The C/.cu `moe_gemv_down_aggregate` (nvcc builds) is still atomic (redcell's determinism fix is oxide-only).
- Windows: 3 of 3 (window1 24 min, window2 30 min, window3 3 min); titan-spark restarted each time, titan-mistral never started.
