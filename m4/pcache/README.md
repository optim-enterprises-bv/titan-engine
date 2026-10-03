# pcache: per-model prefix cache size (and a KV byte bound) for the swap server

Branch `pcache` (mistral.rs worktree `top-pcache/mr-pcache`, base `integ-20261002` 23296a4b8), commit **2a300c23c**.
Kernels unchanged (`top-pcache/oxide-kernels -> ../oxide-integ`). Binary (scratch, NOT deployed):
`m4/pcache/out/mistralrs-pcache`, sha256 `4e19be13e9133477f33ea039739695c583dcb991e7bfb6d82e354ea135d7a7d3`
(nvcc-free `cargo build --release -p mistralrs-cli --features oxide`, CARGO_TARGET_DIR `target-pcache-oxide`).
deploy/models.toml, bin/*, titan-mistral untouched; nothing pushed.

## What changed
- `[models.titan] prefix_cache_n = N` (next to `idle_ttl_secs` / `[models.titan.env]`): this model's prefix cache size;
  absent = `[runtime] prefix_cache_n` (default 16); 0 = prefix cache off for this model only. It goes into the
  model's `EngineConfig` in `build_titan_swap` (both the preloaded default and every registered unloaded model);
  unload/reload keeps it through `reboot_state` -> `UnloadedModelState`, so it applies on every (re)load.
  (`TitanModelSettings.prefix_cache_n`, `TitanSwapPolicy::prefix_cache_n(id, global)`.)
- `[models.titan] prefix_cache_max_mib = M` (optional): bounds the KV bytes the sequence-level prefix cache keeps on
  the GPU. It is put in the model's env as `TITAN_PREFIX_CACHE_DEVICE_MIB` (so it also works as a plain env var),
  read once in `Engine::new` (under the model's settings) into `PrefixCacheManagerV2::max_device_bytes`.
  `evict_caches` now drops the oldest on-device entries until the count bound **and** the byte bound hold; bytes are
  each entry's allocated K/V buffers (`all_data`, capacity not length). With no bound set the decisions are the old
  ones (count only), so every existing roster entry is bit-for-bit unchanged. With a bound set it logs one
  `Prefix cache: N sequences, X MiB ... (limit M MiB); evicted k (Y MiB)` line per finished request.
- Files: `mistralrs-cli/src/config/mod.rs` (fields + 3 tests), `mistralrs-cli/src/commands/config.rs`,
  `mistralrs-core/src/titan_swap.rs`, `mistralrs-server-core/src/mistralrs_for_server_builder.rs`,
  `mistralrs-core/src/engine/mod.rs`, `mistralrs-core/src/prefix_cacher.rs` (+ 1 CUDA test).
- Hybrid models (the 35B, qwen3-next) use the host-memory hybrid cache (TITAN_PREFIX_CACHE_ENTRIES/MIB); for them only
  `prefix_cache_n == 0` vs `> 0` matters, unchanged.

## Gates (window 1, `window1-20261002-2105.log`, 27 min)
| gate | result | evidence |
|---|---|---|
| (1) config unit tests (`cargo test --release -p mistralrs-cli --features oxide --bin mistralrs config::tests`) | **11/11 pass**, incl. `titan_prefix_cache_settings_are_per_model`, `titan_prefix_cache_n_absent_is_the_global_default` (absent -> [runtime] / CLI default 16; unknown model -> global), `titan_prefix_cache_n_rejects_bad_values` | out/test-cli.log |
| prefix_cacher unit tests incl. new CUDA `device_byte_limit_evicts_oldest_until_it_fits` | **21/21 pass** | out/test-core.log |
| (2) from-config swap 35B -> gemma4-12b (`prefix_cache_n = 0`) -> 35B, roster.toml | swaps 5.4 s / 28.9 s; gemma4 **6/6 ~4k prompts (3532-4683 tok), 0 OOM**, max VRAM 10218 MiB; top-20 logprobs **4/4 identical** to integ's `--prefix-cache-n 0` G3 run (out/g12-g3.json) | out/g12-pc0.json, out/roster.server.log |
| 35B shared-prefix TTFT (pc.py workload, 8 x ~6.4k-token shared prefix) before the swap | first 2.64 s, 7 repeats median **0.36 s** (6144 cached each), decode 138.8 tok/s | out/q35-before.pc.json |
| same, after swapping through gemma4 | first 2.27 s, repeats median **0.87 s**, last three 0.46 / 0.54 / 0.48 s, **6144 cached on every repeat**, decode 136.5 tok/s (cache off is 2.93 s, m4/integ) | out/q35-after.pc.json |
| per-model setting reached the engine | `Prefix caching enabled` logged for qwen3.6-35b (initial load and reload after gemma4), gemma4-12b-def/-cap, redcell-def/-cap; **not** for gemma4-12b and redcell-26b-pc0 | out/roster.server.log |
| before-gate: gemma4-12b with the global 16 (`gemma4-12b-def`) | 5 ok then **request 6 OOM** (VRAM 9994 -> 10410 -> 10986 -> 11978 -> 13450 -> 14890 MiB) | out/g12-def.json |
| (a) byte bound: gemma4 16 + `prefix_cache_max_mib = 1024` | **7/7 ok, 0 OOM**, kept KV <= 920 MiB, VRAM plateaus at 11818 MiB | out/g12-cap.json, `Prefix cache:` lines |
| REDCELL ~4k, default 16 | **4/7, 3 OOM** (VRAM up to 15916) | out/red-def.json |
| REDCELL ~4k, `prefix_cache_n = 0` | **6/6 ok**, VRAM 14380-15468 MiB | out/red-pc0.json |
| REDCELL ~4k, 16 + 1024 MiB cap | **4/7, 3 OOM**: fails with only 380-590 MiB kept | out/red-cap.json |
| VRAM after each swap (nvidia-smi used) | 35B 13722; gemma4 7274; 35B 13752; gemma4-def/cap 7274; redcell 12234-12236 MiB. Swap log: 15507 MiB free after every unload | window log `TOUCH` lines |
| (3) REGRESSION CORE: 35B MTP=2 vs m6/out/h-off.json | **40/40** | m3/out/pcache-m2.json |
| 35B MTP-off (MTP-dir Q4_K_XL, TITAN_MTP=0) vs m6/out/mtpfile-off.json | **40/40** | m3/out/pcache-mf0.json |
| IQ2_M new vs old (bin/mistralrs-titan-integ, rerun this window), 8x300 | **8/8** (also 8/8 vs integ's recorded i-new and b2merge's i-old) | out/i-new.json, out/i-old.json |
| Bonsai-27B Q1_0 new vs old, 8x300 | **8/8** (8/8 vs recorded q-new, q-old-b2merge) | out/q-new.json, out/q-old.json |
| Bonsai-2 B2-off vs m4/bonsai2/out/o-off.json | **8/8** | out/b-off.json |

The 1024 MiB cap and the gemma4-def runs fail before this change in the sense required: gemma4 with the global 16 OOMs
on the 6th ~4k prompt (out/g12-def.json), and per-model 0 / the byte cap make the same sequence pass.

## Root cause, with evidence
- **gemma4-12b**: non-hybrid, unpaged, so the sequence-level prefix cache keeps each finished sequence's whole KV on
  the GPU, up to 16 entries, with no byte limit. One ~4k gemma4 entry holds **376-400 MiB** of K/V buffers (allocated
  capacity: global layers grow in 512-token steps, sliding layers hold their window), and a 1-token "Hello" holds
  168 MiB. VRAM climbs ~0.4-1.5 GB per request until the 6th OOMs. The byte bound holds it flat (920 MiB kept, VRAM
  11.8 GB).
- **The gemma4 cache brings nothing here**: re-sending the same ~3.5k prompt took 2.39 s with the cache on, as did the
  cold one (out/g12-cap.json rows 5/6), consistent with a rolled-over sliding-window cache that cannot be rewound to
  a prefix (`skips_rolled_over_rotating_candidate_that_cannot_rewind`). So `prefix_cache_n = 0` costs gemma4 nothing
  on long prompts.
- **Numerics with the cache on**: gemma4 outputs with the cache on differ from cache-off (g12-def/-cap vs g12-pc0: 0/6
  and 3/6 identical top-20 logprobs; e.g. prompt 0 top-1 logprob -0.968 vs -0.928, same top-1 token). Likely a short
  shared-prefix (BOS) hit from the earlier "Hello" entry, so prefill resumes from an offset on a different code
  path. This predates the change (same with the global 16), and is one more reason for 0 on gemma4.
- **REDCELL**: the model peaks at ~15.5 GB on a ~4k prompt with nothing cached, so any kept KV (380 MiB is enough)
  pushes it over 16 GB. A byte cap does not help unless it is near 0, so `prefix_cache_n = 0` is the setting.
- **(b) pool fragmentation: not fixed**. After each OOM, `recover_from_oom` trims the pools and free VRAM goes from
  17-19 MiB to 3.2-8.0 GB (log `pools trimmed (17 -> 3345 MiB free)`). That memory was the failed step's
  activations plus pool reserve, and these logs cannot separate the two. A real fix (bounded pool release threshold,
  or retrying the step after trim) touches the allocator / scheduler error path for every model. It is neither cheap
  nor clearly safe, so it was left out. With the per-model 0 / byte bound, nothing reached the OOM path in the passing
  runs.
- **35B repeat TTFT after the swap**: every repeat hit 6144 cached tokens (cache on). Repeats 1-4 took 0.87-1.21 s and
  then 0.46-0.54 s, against 0.29-0.99 s before the swap. The prefix cache behaves the same, so this is probably the
  freshly reloaded tiered-expert state warming up (unverified). It is still ~3-6x better than cache-off (2.93 s).

## Draft roster diff (NOT applied)
`models-toml-draft.diff`: deploy/models.toml (main worktree, current uncommitted state) + m4/integ/models-toml-draft.diff
(bonsai2-27b, spark-x2.5, gemma4-12b, redcell-26b), with `prefix_cache_n = 0` under `[models.titan]` for **gemma4-12b
and redcell-26b** (REDCELL benefits: 3/7 OOM -> 6/6) and their comments updated. Global `[runtime]` untouched (the 35B
keeps 16). Parses with tomllib. Apply with `patch -p1` in ~/titan-engine.

## Files
`roster.toml` (test roster: 35B verbatim from deploy + gemma4-12b with 0 + -def/-cap/-pc0 variants), `pcache.py`
(long / pc / touch / same client; every request names its model), `lib.sh`, `window1.sh`, `b2client.py` (copy, writes
here), `out/` (results, server logs, test logs; references copied in: o-off, i-integ, q-integ, *-old-b2merge).

## Not done
- (b) fragmentation, see above.
- `prefix_cache_max_mib` does not size itself from free VRAM; it is a fixed number per model.
- Windows: 1 of 3 used (27 min). titan-spark was restarted by the EXIT trap (another agent's window took the lock
  right after).
