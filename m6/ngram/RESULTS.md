# n-gram (prompt-lookup) drafting on titan-094: windows 1 and 2, 2026-09-28

Branch `ngram-spec-094`, worktree `~/titan-engine/mr-ngram94`, commit 1fcb4047b on titan-094 db0d27755.
The earlier port onto ktrace (`ngram-spec`, 0ecc3ea in `mr-ngram`) was never built.
Binary: `target-ngram94-oxide/release/mistralrs`, built with `--features oxide`. Not deployed.

## Design

- `speculative/ngram.rs`: at each step, look for the context's last n tokens (prompt plus generated text) earlier
  in the context, trying n = 3 first and n = 2 last. Among the 64 latest occurrences, take the one whose match
  extends furthest back (the latest on ties). The draft is up to `run` of the tokens that followed it.
  - `run` adapts: it doubles after a fully accepted draft and halves after a rejection, but never below
    accepted + 1. It stays within `[KMIN=2, K=16]` and starts at 4.
  - Environment: `TITAN_NGRAM=1`, `TITAN_NGRAM_K`, `TITAN_NGRAM_MIN` (default 2), `TITAN_NGRAM_MAXN` (3),
    `TITAN_NGRAM_KMIN`, `TITAN_NGRAM_K0`, `TITAN_NGRAM_ADAPT=0`, and `TITAN_NGRAM_POLICY`:
    - `first`: take the n-gram draft;
    - `agree`: take it only when its first token equals MTP's first draft;
    - `longer`: take the longer of the two.
- `quantized_qwen35_moe.rs` `spec_propose`:
  - Draft source: the n-gram draft when one is found (subject to the policy), otherwise the MTP draft.
  - The MTP KV cache is caught up on every step (one MTP pass). This keeps MTP valid after n-gram steps.
  - Works without MTP too: the plan is then built `without_target_hiddens`.
- Verification uses the same M6 path as MTP:
  - `forward_verify`, the row-exact batched rows;
  - `lin_rows` now splits more than 8 rows into chunks of at most 8 `mmvq_rows` rows, each still row-exact;
  - verifications longer than MTP's stay off the CUDA-graph path.
- GDN rollback:
  - Verifications of 4 rows or fewer keep M6's per-row state snapshots, which is 60 MiB per row for 35B.
  - Longer ones keep one copy of the state before the verification plus the rows' recurrence inputs.
    `titan_gdn::GdnRowReplay::replay` re-runs the kept rows with the same decode kernels on rollback.
- Driver (v0.9.4 `propose_and_stage_batch`): a proposal may be shorter than `plan.proposal_len`. The plan is
  max(MTP n, K).
- Stats: `titan ngram stats` lines in the server log, separate from `titan mtp stats`.

## Gates (window 1: 19:00:01 to 19:43:44, service down 43m43s)

| gate | result |
|---|---|
| (4) nvcc-free build links | yes, 226 s incremental |
| (1) MTP=2 + NGRAM, 40x256 vs m6/out/h-off.json | **40/40** |
| (1) no MTP + NGRAM, 40x256 vs m3/out/q35-prof.json | **40/40** |
| (1) code bench, policy `first` vs `agree` | 20/20 identical. The ngram-off run died, so there is no off reference |
| (2) 40-prompt speed, MTP=2: NGRAM off vs on (`first`) | 100.3 vs 91.3 tok/s = **-9.0%: FAIL** |
| (2) no MTP + NGRAM (`first`) | 77.5 tok/s vs 83.5 (titan-094 report, other window) = ~-7% |
| (3) code bench, `first` | 44.1 tok/s decode; n-gram acceptance 90.6%, 11.76 tokens accepted per n-gram verification (872 verifications) |
| (3) code bench, `agree` | 44.5 tok/s decode; acceptance 92.8%, 13.30 accepted per verification (770 verifications) |
| (3) code bench, off | no result: prompt 1 stalled (see below) |

The only paired off/on point is code-bench prompt 0 (a whole-file rewrite, 1024 tokens):
- off: 45.5 tok/s;
- `first`: 84.4 tok/s;
- `agree`: 83.3 tok/s.

The ratio is x1.85.

## Why the speed numbers are not final

- From about 19:09, a 7.2 GB rust-analyzer started by PID 878468 was running and host RAM was down to about
  0.4 GB free. I killed it at about 19:24, per the rules.
- From about 19:23, an unrelated `rpmbuild` (guarded-browser packaging) was running too, and the 5-minute load
  average reached 54. The tiered CPU experts compete for those cores.
- The code-bench off run stopped making progress on prompt 1: decode fell from 15 to 3.6 tok/s at 12:10:46 UTC
  and nothing followed until the 880 s runtime cap killed the run.
- In both n-gram runs, fully accepted 17-row verifications ran at only ~45 tok/s after prompt 0, about 0.3 s per
  verification. That is consistent with CPU starvation of the expert misses, though it was not proven.
- The code-bench off/on comparison and the 17-row verification cost need a clean window.

## Gate 2 failure and what to try next

On chat prompts, the `first` policy with `MIN=2` drafts from weak 2- and 3-token matches:
- acceptance was 30%, 0.75 tokens per n-gram step;
- each of those steps displaces an MTP draft, which accepts 2.5 tokens per step.

Ways to fix this without a rebuild (environment only):
- `TITAN_NGRAM_POLICY=agree`, which drafts only when MTP's first token agrees. On code it matched `first`.
- `TITAN_NGRAM_MIN=3`.

A code-side option: require the match to extend at least 8 tokens back before displacing MTP.

The flag stays off by default. It cannot be enabled until gate 2 passes with one of these settings.

Files:
- `out/ng1-*.json`: gate 1 outputs;
- `out/cb1-*`: code-bench outputs, metrics and server logs;
- `code-bench.json` (from `gen_code_bench.py`), `run_code_bench.py`, `window1.sh`, `window1.out`.

## Window 2 (22:38:21 to 22:52:46, service down 14m25s; cut short by the titan reboot at about 22:53)

Same binary as window 1, no rebuild.
- Host at the start: load average 5.0, 11 GB free, no non-titan process over 4 GB, so no settle wait was needed.
  rust-analyzer was killed before each run.
- vmstat (`out/w2-vmstat.log`) shows about 8 GB of swap in use and 24-25 runnable threads during the code runs.
- At 22:52:46 the shutdown for the reboot began:
  - it stopped cb2-agree during request 11 of 20, so the run saved no completions or metrics;
  - systemd refused every later run ("transaction is destructive": systemd-exit queued).

**The 40-prompt gate-2 runs (off, agree, first+MIN=3) and the code-bench first+MIN=3 run never started.**

| check | result |
|---|---|
| code bench, ngram off | **finished: 20/20 requests, 0 failed**, 39.9 tok/s decode (file 41.0, diff 39.5, fix 36.7), wall 494 s |
| code bench, agree | requests 0-10 only (killed by the reboot). n-gram acceptance 94.2%, 13.86 accepted per n-gram verification (556 verifications) |
| code bench speedup, agree vs off, same window, requests 0-10 | whole-file rewrites (0-6, 7168 tokens): 41.0 -> 52.0 tok/s = **x1.27**; diff->function (7-10, 1167 tokens): 34.6 -> 34.0 = x0.98; together x1.21 |
| identity: cb2-off (window 2) vs cb1-first and cb1-agree (window 1), same binary | **20/20 and 20/20** byte-identical |
| gate 2 for agree and first+MIN=3 | **not measured** (the reboot) |

**The window-1 hang.** In window 2, with the host quiet, the same run finished all 20 requests and none hit the
new 240 s per-request timeout.
- During the window-1 hang, the server used about 1.4 cores on average (20 CPU-min over 14.7 min), against about
  15 cores in a normal run.
- There was no kernel OOM, and no titan error was logged.
- At the same time, a 7.2 GB rust-analyzer and the rpmbuild were running and btrfs writeback kworkers were busy.
- Most likely the server was blocked on host memory or I/O pressure, not on anything in titan.
- The runner now has a per-request timeout (`REQ_TIMEOUT`, default 240 s). On a timeout it saves
  `<run>.diag-<id>.txt` (PSI, top RSS/CPU, the server's thread wait channels) and continues with the next request.

**Why the code-bench speedup is small despite ~94% acceptance.**
- A fully accepted 17-row verification yields about 14.9 tokens.
- On these prompts it costs about 0.3 s, against about 25 ms for a 3-row MTP step, which yields 3 tokens.
- The verification is row-exact, so its cost grows with the row count: per row it runs the router, the GDN
  recurrence and the attention, and each row misses more tiered experts, which run on the CPU twin.
- Long drafts therefore help only whole-file rewrites (x1.27). Short edits break even.
- Two directions:
  - a shorter `TITAN_NGRAM_K` (8), which trades acceptance length for a cheaper verification;
  - making many-row verifications cheaper: batching the per-row router/shared-expert GEMVs, one expert pass per
    distinct expert.
- Also, the code-bench baseline itself (about 40 tok/s with 99.7% MTP acceptance) is well below the 100 tok/s of
  the 40-prompt set. That is 64k max-seq-len (146 vs 161 experts per layer on the GPU), 3-4k contexts and swap.
  Worth a look apart from n-gram.

## Recommendation

**No policy is ready to recommend.**
- `first` (MIN=2) fails gate 2: -9.0% on the 40-prompt set.
- `agree` and `first`+MIN=3 were never measured on the 40-prompt set.
- `agree` has 20/20 code-bench identity, but no 40/40 gate-1 run of its own. `first` has 40/40 + 40/40.
- `agree` is the likeliest candidate: on chat text it only replaces an MTP draft whose first token it agrees with.
  On code, it matched `first` (window 1: identical outputs, 44.5 vs 44.1 tok/s).

`TITAN_NGRAM` stays off by default. Enabling it needs one more window with:
- 40-prompt off / agree / first+MIN=3 in the same window (the gate-2 bar is -2%);
- the full code bench with agree;
- agree's 40/40 identity against h-off.json.
