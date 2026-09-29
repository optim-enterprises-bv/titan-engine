# titan-engine standing benchmark suite

**Rule: every merge to `titan-094` gets a bench run before deploy.** Run the candidate binary with
`bench/window.sh` and compare it against the latest clean run of the deployed binary with `bench/compare.py`.
Any regression over 3%, or any identity change, has to be explained before the binary replaces
`bin/mistralrs-titan-094`.

A run takes about 10 minutes (15 at most) and needs the GPU to itself.

## Running it

```
# the main session queues it under the GPU lock (gpu-queue.sh `run` line):
run bench-<label> $E/bench bash $E/bench/window.sh <binary> <label>

# inside a window you already hold (titan-mistral stopped):
bench/run.sh <binary> [label]
BENCH_SRC=~/titan-engine/mr-xyz bench/run.sh <binary> <label>   # binary built from another worktree

python3 bench/compare.py baseline-094 <label>                   # A B: run id, id prefix or label (its latest run)
python3 bench/compare.py A B --all                              # also llama.cpp times, path breakdown, GPU split
```

- `window.sh` does not take the lock itself. It waits for the 30-minute service gap (skipped while
  `~/titan-engine/.nogap` exists), stops titan-mistral, runs `run.sh`, and restarts the service on exit.
- `run.sh` refuses to start while titan-mistral is active.
- Its servers run on port 18530 under `systemd-run MemoryMax=20G`, one at a time. They take the service config
  (the environment and arguments) from `deploy/titan-mistral.service`, so the bench follows the unit.
- `BENCH_SRC` (default `~/titan-engine/mr-094`) is the source tree of the binary. The kernel tier loads its
  `mistralrs-quant/src/gguf/*.ptx`, and its git revision is recorded as mistral.rs.

## Outputs

| file | what |
|---|---|
| `bench/history.jsonl` | one JSON line per run: metrics, verdicts, revisions, hygiene |
| `bench/runs/<date>-<label>.md` | the per-run report |
| `bench/runs/<date>-<label>/` | raw data: `summary.json`, per-tier JSON, `run.log`. The nsys files and server logs stay local (gitignored). |
| `bench/history.html` | static chart page built from the JSONL: inline SVG, no scripts, no CDN. Open it as a file. |

To rebuild the page: `python3 bench/lib/mkhtml.py`.

To regenerate a run's report and its history line: `python3 bench/lib/report.py bench/runs/<id>`.

## What each number means

### 1. Kernel tier (about 2 min): `kernel.<op>/b<B>.*`

The harness is `m4/kbench/kbench.cpp`, built to `bench/bin/kbench` by `bench/kbuild.sh`. It feeds the same device
buffers to both sides:

- **llama.cpp**: `~/ai/llama.cpp` at acecd56, through ggml graphs;
- **ours**: the service's exact launch sequence (titan-oxide-ffi launchers plus the tree's PTX).

The ops are the 35B shapes:

- expert gate/up Q4_K (k2048 n512);
- expert down Q5_K and Q6_K (k512 n2048), 256 experts, top-8;
- the dense Q8_0 projections: qkv, attn_gate, ssm_out, shared-expert gate/up and down;
- lm_head (b1 and b3 only);
- llama.cpp's fused expert FFN, for scale.

Batches are 1, 3, 8, 512 and 2048.

Each op is launched 50 times after 3 warm-ups. A 256 MiB memset before every launch evicts L2, so weights are read
cold. The whole tier runs twice (rep 1 and rep 2) under one nsys session. It reports:

- `llama_us` / `ours_us`: the GPU kernel time of the whole op (activation quantize included), summed from nsys
  inside each launch's NVTX range. The median is over all 2 x 50 launches.
- `ratio`: ours / llama.cpp. Above 1 means we are slower; the report puts ≥ 1.25 in bold.
- `ours_mmq_moe_us`: our MMQ-MoE port. It is faster than llama.cpp at prefill, but **the tiered service path does
  not call it**. It is listed so the gap is visible.
- `bit_equal`: the share of output floats that are bit-identical to llama.cpp.
- `spread`: (max - min) / median of the two reps' medians.

The GDN recurrence and conv kernels are not in the harness. Their time appears in the e2e nsys split, under
attention/GDN.

**Caveat:** the kernel tier links the prebuilt `oxide-kernels/titan-oxide-ffi/target/release/libtitan_oxide_ffi.a`.
If a merge changes oxide-kernels launchers, rebuild that staticlib first. `run.sh` relinks kbench whenever the `.a`
or `kbench.cpp` is newer than the binary. The PTX always comes from `BENCH_SRC`.

### 2. Path tier (from the e2e nsys captures): `path.<class>@<profile>.span_us`

This is one layer's MoE forward as the model actually runs it, with nothing synthetic. The **MoE block** is the span
from the post-attention rmsnorm before the router top-k kernel to the next rmsnorm. It includes:

- the router;
- the tiered expert gemvs;
- the CPU-miss sync and upload gaps;
- the combine and the shared expert.

Values are the median over layers and steps:

| class | what | profile |
|---|---|---|
| `b1@off` | plain decode | MTP off |
| `b3@mtp2` | the MTP=2 verify forward (3 rows) | MTP=2 |
| `mtp_draft@mtp2` | the MTP head (stock indexed MoE) | MTP=2 |
| `b512@off`, `b512@mtp2` | one 512-token prefill chunk | both |

`busy_us` is the kernel time inside the span. `idle_us` is the span with no kernel running, mostly the tiered miss
sync. `cat_us` splits it into experts, shared expert and other. The report also divides the routed-expert kernel time
by llama.cpp's fused FFN at the same batch (kernel tier).

These numbers come from under nsys, which adds per-launch CUPTI overhead. Compare them run to run, not against
un-profiled wall time. Spread = between the two reps.

### 3. E2E tier (about 6 min): `e2e.*`

These run on the given binary with the service config.

**8 eval prompts × 256 tokens, greedy** (`bench/prompts/eval8.txt`, the first 8 of `m4/prompts-eval.txt`). They run
twice with MTP=2 (the service) and twice with MTP off (`TITAN_MTP=0`), all streamed.

- `decode_tok_s.{mtp2,off}`: Σ(completion tokens - 1) / Σ(last chunk - first chunk), measured client side. It
  excludes prefill. `decode_tok_s_server.*` is the median of the server's `avg_compl_tok_per_sec`.
- `ttft_cold_ms.{mtp2,off}`: the median time to first token of the 8 prompts, sent with a fresh nonce prefix and
  `max_tokens=1`, so each is a cold prefill. The plain second pass of the 8 prompts is prefix-cache warm, which is
  why it is not used for TTFT.

**Long prompts** (MTP=2) come from `bench/prompts/corpus-sys28k.txt`, a frozen opencode-style system text, with a
unique nonce at the start so each one is cold:

- `prompt_tok_s.{4k,13k}`: prompt tokens / the server's `total_prompt_time_sec`. The real token counts are in the
  report, about 3.9k and 12.7k.
- `ttft_s.{4k,13k}_cold`: client TTFT.
- `ttft_s.13k_warm`: the same 13k prompt sent again, i.e. the prefix-cache warm time.

**Identity** (8-prompt subsets, frozen in `bench/ref/`):

- MTP=2, both passes, vs `m6/out/h-off.json`;
- MTP off vs `h-off`;
- the pre-MTP GGUF (`~/ai/models`, MTP off, `--max-seq-len 4096` as in `m3/collect.sh`) vs `m3/out/q35-prof.json`.

A mismatch is re-checked without streaming, so a streaming artefact is told apart from an engine change.
**identity FAIL blocks a deploy.**

**GPU time split** (`split.{mtp2,off}.*_pct`): nsys (`nsys launch`, then `start`/`stop` around the requests)
captures two reps of a ~2.1k-token cold prompt plus 32 decode tokens. The decode window runs from the end of the
last prefill chunk to the last kernel of the request. Its share by category:

| category | kernels |
|---|---|
| experts | tiered/indexed MoE gemvs, q8_1 quantize, GLU, router top-k and matmul, combine |
| attention/GDN | GDN recurrence/conv/norm, rope, softmax, attention cuBLAS |
| dense | Q8_0 mmvq/mmq projections, including the shared expert |
| lm_head | the 248 320-row Q8_0 |
| copies | `ucopy_*` / `copy2d_*`. In decode these are mostly the per-step attention KV-cache append (two ~127 µs bf16 copies per attention layer at ~2.1k context in the baseline), so this share grows with context. |
| other | norms, elementwise, casts, argmax |
| idle/gaps | window time with no kernel running: host work, tiered miss syncs, launch gaps |

### 4. Hygiene and CONTAMINATED

Before and after the run, the bench records:

- uptime and load, `free -m` and the top-5 RSS;
- GPU SM/memory clocks, temperature, power, pstate and throttle reasons (nvidia-smi / NVML);
- the git revisions of mistral.rs (`BENCH_SRC`), candle (the titan-engine subtree; upstream a9667ca),
  oxide-kernels, titan-engine and llama.cpp.

It kills any rust-analyzer whose parent is PID 878468.

Every timed item runs at least twice. A run is **CONTAMINATED** when any of these holds:

- the 1-minute load is over 6;
- any non-bench process is over 4 GB RSS;
- any e2e or path item has a spread over 5%;
- more than 10% of the kernel-tier items have a spread over 5%.

The reasons are listed in the report and in the JSON. A contaminated run is kept in the history (hollow markers
on the chart), but don't base a deploy decision on it: rerun.

## Files

| file | what |
|---|---|
| `run.sh` | the suite |
| `window.sh` | the GPU window wrapper |
| `compare.py` | deltas between two runs |
| `kbuild.sh` | builds kbench |
| `lib/client.py` | the e2e client: plans service, mtpoff, ident and nsys |
| `lib/nsys_split.py` | path tier and GPU split |
| `lib/kanalyze.py` | kernel tier |
| `lib/hygiene.py` | the hygiene snapshots |
| `lib/svcconf.py` | reads the service unit |
| `lib/report.py` | summary, history line and markdown |
| `lib/mkhtml.py` | the chart page |
