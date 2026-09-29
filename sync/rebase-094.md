# titan-094: mistral.rs fork rebased onto upstream v0.9.4

Date: 2026-09-28. Branch `titan-094` in worktree `~/titan-engine/mr-094`, created from upstream/master 2370966
(v0.9.4). ktrace was merged into it (not rebased), first at 240cde7 and then again at e3fb03b (the prefix-cache
fix deployed on :1234 in the meantime). ktrace itself and the feature branches are untouched.

Commits on titan-094:

| commit | what |
|---|---|
| 1cda5b96c | Merge ktrace (240cde7): the textual conflict resolutions only. It does not build on its own. |
| 337e393ba | Merge ktrace (e3fb03b): prefix cache keeps the two longest resume points, then chunk-boundary points. Clean merge. |
| ec6dac67e | Adaptation: fork-local GGUF loader, fork-local GDN, oxide ABI layer, candle pin. |
| 2060828a1 | Compile fixes from window 1 (dtype matches, extern "C" adapters, refutable patterns), plus oxide fallbacks for indexed row copy, token-major qk rope and the batched sampling plan. |
| f875b6b7d | Titan GGUF architectures ignore a projector the CLI auto-selects from the model directory. |
| db0d27755 | ...and take text auto-device-map params when the CLI built multimodal ones. |

In the titan-engine repo: b563edc (candle API backports) and this file.

## Candle: kept ours (a9667ca + the fork), with API backports

v0.9.4 pins candle 66a8cf1, not e65eb1de; e65eb1de is an earlier CI-only commit on the same path. Rebasing our candle
onto it would have pulled in "Optimize CUDA kernels (#3600)": new and changed entry points in binary/unary/reduce/cast/
quantized.cu and moe_align. Those would invalidate the cuda-oxide candle PTX set (reference/candle) that the
nvcc-free build serves. Instead, the workspace stays on a9667ca + `../candle`. mistral.rs v0.9.4 needed only a handful
of candle APIs, and these were added to our candle (b563edc, all additive, ktrace unaffected):

- `BarrierPool::execute_chunked` (+ a worker nesting guard);
- `init_global_threadpool` (it keeps rayon's default thread count, not 66a8cf1's physical-core count);
- `pub set_thread_affinity`;
- `QTensor`/`QMatMul::embedding`: a dequantize + index_select fallback. 66a8cf1 has `get_rows` kernels, but the
  titan models embed on their own;
- `QTensor::gemv_fused_shared_lhs` and `indexed_gemv`, both returning `None`. Callers fall back, exactly as upstream
  does on x86 without AVX-512.

`Cargo.toml` pins candle-core/nn/flash-attn-v3/metal-kernels back to a9667ca. `cargo fetch` then downgraded candle's
own tokenizers dependency 0.23 to 0.22, which is the same as ktrace.

## Kept ours (fork-local, registered alongside upstream)

- **GGUF models**: `pipeline/gguf_titan.rs` (`TitanGGUFPipeline`) is the fork's GGUF path, ported to v0.9.4's
  Pipeline and speculative traits. `GGUFLoader::load_model_from_path` sends these architectures there:
  - `qwen35moe`, `qwen35`, `qwen3next` → `quantized_qwen35_moe`;
  - `gpt-oss` → `quantized_gpt_oss`;
  - `qwen3moe` → `quantized_qwen3_moe`, when `TITAN_TIERED=1` only (upstream otherwise).

  Everything else goes to upstream's native GGUF loader (1fdb451). `TITAN_GGUF_ROUTE=upstream` turns the routing off.
  The titan path runs without PagedAttention (it warns and disables it), as the service does. ktrace could run
  qwen3moe and gpt-oss paged; that possibility is dropped. `FromGGUF` is back as a titan trait in utils/model_config.rs.
- **GDN**: `titan_gdn/` is ktrace's `gdn/` + `cuda/gdn.rs`: the GDN the 35B/80B/Bonsai gates were run on, with the
  MTP-verify rows and the chunked-prefill conv carry. Its launchers use the 84b53bf ABI (`titan_gdn/ffi.rs`):
  - oxide build: the plain names, i.e. the titan-oxide-ffi twins;
  - nvcc build: `titan_legacy_*`, compiled from `cuda/titan_gdn_legacy.cu`, which is the 84b53bf gdn.cu in a
    namespace with its C names renamed.

  Upstream's `crate::gdn` and `cuda/gdn.rs` are untouched and serve upstream's own models.
- **Hybrid cache**: upstream's HybridCache (pooled slots, checkpoint lanes, V-major option) with two titan additions:
  - `titan_snapshot_physical` / `titan_restore_slot`: the 84b53bf semantics without the new sequence-ownership
    checks. The model snapshots mid-prompt, where it knows pool rows but not sequence ids.
  - The qwen35 model declares an `Opaque` key-major state, so v0.9.4 never picks the V-major layout for it.
- **Tokenizer**: `gguf/titan_tokenizer.rs` is ktrace's GGUF tokenizer conversion (o200k split, rendered harmony
  tokens), used by the titan models. Upstream's rewrite (per-`pre` profiles) is left for upstream's models, because a
  different tokenization would change the gated outputs.
- MTP (TITAN_MTP), piecewise CUDA graphs, tiered experts + CPU twin + P1 prefetch, chunked prefill, the hybrid
  prefix cache, the Q1_0/IQ4_NL/MXFP4/NVFP4 MMQ ports: carried over as they were.
- Harmony EOS set (chat_template.rs): ours. Upstream's `HARMONY_ALTERNATE_EOS` made `<|message|>`/`<|start|>`/
  `<|channel|>` EOS for harmony templates, which ends a gpt-oss reply after its analysis message.

## Taken from upstream

Everything that did not conflict with the above came from upstream:

- the scheduler/concurrency work (8010b6a, b0882bd);
- the hybrid auto-device-map fixes (d1ffbe7);
- the kernel-object cache (d348a88);
- logprobs in natural log (8423e2e);
- the tool-call delimiter fix (b43d344);
- the Anthropic Messages / Claude Code server (`/v1/messages` checked below);
- the security fixes and SECURITY.md;
- 1fdb451 for all non-titan architectures;
- upstream's process-stable fast_mmvq/mmq workspaces, which supersede our da11fc3;
- `usage.prompt_tokens_details.cached_tokens` (d71ab1f). It is wired to the hybrid prefix cache: a resume calls
  `prefill_v2_normal(.., offset)`, so `prefix_cache_len` is the resume point. It was checked live (10752 cached of
  10883).

**Upstream's GDN in-place pooled-state kernels (7d8fa17) were not adopted for the titan models.** They conflict
with our kernels: same C names, new ABI (state index tables, V-major layout, in-place pool). There is no cuda-oxide
twin, so the nvcc-free build cannot run them. Left for later: porting them is the way to inherit upstream's GDN.

**Logprobs are now natural log.** m4/toplogp.py and the cmp.py scripts compare gaps to the top-1 within one run and
work unchanged. Absolute values in `*.top.json` written by pre-094 mistral.rs builds are log10, so compare those
across builds only after multiplying the old values by ln 10 (2.303). llama.cpp's values were always natural log.
Greedy output is unaffected: argmax does not depend on the scale.

## Conflict resolutions (30 files)

- **pipeline/gguf.rs**: upstream's version, plus the titan route (`mod titan` = gguf_titan.rs) and the
  projector/auto-map handling for titan architectures.
- **gdn/{layer,norm,projection}.rs, cuda/gdn.rs, gguf/gguf_tokenizer.rs**: upstream's. Ours moved to fork-local
  copies (above).
- **quantized_qwen3_moe.rs** (modify/delete): kept. The other quantized_* models ktrace did not change (phi2, phi3,
  qwen, qwen3, starcoder2) stay deleted, and upstream's native loader handles those architectures.
- **models/mod.rs, utils/model_config.rs**: titan models only; FromGGUF re-added.
- **engine/add_request.rs**: unpaged hybrid pipelines use the titan hybrid prefix cache (`search_hybrid`), skip
  upstream's sequence-level lookup, then go through upstream's slot allocation. A hit counts once
  (`record_prefix_cache_hit`).
- **engine/mod.rs**: hybrid restore before the prompt step, with upstream's `logger` argument.
  `save_hybrid_prefix` stays after the completion and prompt steps.
- **pipeline/sampling.rs**: `cache_finished_sequence` skips hybrid sequences when unpaged (46053a3). The engine keeps
  them at saved resume points instead. Upstream's unpaged snapshot would resume partial matches.
- **prefix_cacher.rs, kv_cache/{mod,hybrid_cache}.rs, sequence.rs**: union of upstream's paged recurrent prefix state
  and the titan hybrid entries. `reset_slot` and `reset()` are upstream's, which zero in place (stable addresses for
  captured graphs, as ktrace's in-place reset did).
- **attention/{mod,backends/mod}.rs**: union of the imports; ktrace's `maybe_synchronize_for` and grouped GQA kept.
- **chat_template.rs**: harmony EOS as above.
- **fast_mmq.rs**: upstream's shared_lhs/DenseMmqRun structure, plus the titan Q1_0 and IQ4_NL/MXFP4/NVFP4 MMQ ports,
  on upstream's `workspace_ensure`. `plain()` dispatches Q1_0.
- **fast_mmvq.rs**: upstream's process-stable workspace.
- **afq/ops.rs, blockwise_fp8/ops.rs, utils/ops.rs, core ops.rs**: upstream's. The ktrace side only added inventory
  trace points (`ktrace()`) around code upstream rewrote.
- **quant gguf/{cuda,mod}.rs, lib.rs, paged-attn lib.rs**: unions.
- **Cargo.lock**: upstream's, re-resolved by cargo.

## nvcc-free build: the oxide ABI layer

titan-oxide-ffi exports the cuda-oxide twins of the 84b53bf launchers. v0.9.4 declares 175 launchers that do not
exist there and changes the C ABI of 23 existing ones. Linking them silently would corrupt memory, so:

- **`titan-nvcc-only` crate** (oxide builds only): one stub per unported launcher, listed in
  `titan-nvcc-only/nvcc_only.txt`. A stub prints the launcher name and aborts. With `TITAN_NVCC_ONLY=warn` it logs
  each name once and continues, a discovery mode for listing everything a workload reaches in one run. Upstream
  declarations of changed-ABI launchers link to `titan_nvcc_only_<name>` under `feature = "oxide"`. These are the
  8 GDN launchers, `top1_large_f32_packed`, the 3 flashinfer paged launchers and the 4 marlin launchers.
- **Adapters** (`titan_oxide.rs` in core and quant): the v0.9.4 signature on the 84b53bf kernel, where both compute
  the same thing for the arguments they get:
  - `qk_rms_norm_rope(_positions)`, heads-first output;
  - `fused_glu_*`, packed rows (`fused_glu` copies strided inputs, as ktrace did);
  - `launch_moe_weighted_reduce_flat(_bf16)`;
  - the greedy top-1 (`ops::cuda_top1_logits_f32_cached` is ktrace's single-row version).

  These keep ktrace's numerics. Upstream changed the RMS reduction order in attention_prep.cu, so on an nvcc build
  the titan models' prefill numerics differ from ktrace.
- **v0.9.4 fast paths turned off in oxide builds**, because their launchers are nvcc-only. Each falls back to what
  ktrace did:
  - batched CUDA sampling plans, i.e. the device top-1/top-k/categorical batch (`cuda_batch_sampling_plan` → None);
  - greedy device verification of MTP drafts and sparse rejection. The drafts are verified row by row through the
    sampler, which is ktrace's path;
  - the `indexed_row_copy` recurrent-state scatter (per-slot copies);
  - token-major qk-rope output.

  Found with `TITAN_NVCC_ONLY=warn` on the 35B service config: after these changes, gates 1–5 reach no stub.
- Build flags: core/quant/paged-attn build.rs return before nvcc in oxide mode (as in ktrace). None of v0.9.4's new
  nvcc-only cfgs (`has_flashinfer_gdn_sm90_kernel`, `has_gdn_fp8_producer`, `has_cutlass_fp8_sm90_kernels`,
  `has_deepgemm_fp8_sm90_provider`, `has_nvfp4_cutlass_sm121_kernels`) are set there, and `cutile` stays off.

### nvcc-only kernels (to port to cuda-oxide later)

Changed ABI (84b53bf twin exists, new variant does not):
- GDN: `causal_conv1d_full`, `causal_conv1d_update`, `gated_delta_rule_recurrence`,
  `warp_gated_delta_rule_recurrence`, `chunked_gated_delta_rule_recurrence`, `gdn_prepare_recurrence`,
  `gdn_decode_recurrence`, `gdn_rmsnorm_gated`.
- Other: `top1_large_f32_packed`, `qk_rms_norm_rope(_positions)` (token-major output), `fused_glu_*` (strided rows),
  `launch_moe_weighted_reduce_flat(_bf16)` (status return; adapted), `flashinfer_decode`,
  `gather_kv_cache_flashinfer`, `reshape_and_cache_flashinfer`, `marlin_{gptq,awq}_4bit_{f16,bf16}`.

New in v0.9.4 (175; the full list is titan-nvcc-only/nvcc_only.txt):
- mistralrs-core cuda (47):
  - GDN pooled/V-major/speculative state: `vmajor_*_gated_delta_rule_recurrence`, `gdn_{pending_transition,
    speculative_*,deferred_*,packed/padded,extract_ragged_conv_state}`, `gdn_rmsnorm_gated_quantized_bf16`;
  - sampling: `top1_large_{bf16,f16}_packed(_batched)`, `top1_large_f32_packed_batched`,
    `topk_large_*_packed_batched`, `topk_large_ranked_*`, `sample_ranked_topk`, `categorical_large_f32_packed_batched`,
    `sparse_rejection_{topk,categorical}_f32`;
  - DFlash: `dflash_{greedy,sample}_select`, and `dflash_context_keys_*` / `dflash_pack_taps_*` (6);
  - `indexed_row_copy_*`, `pack_completion_input_u32`, `pad_decode_input_u32`, `rope_sincos_positions`,
    `add_rms_norm_*`, `dynamic_conv_*`, `cuda_graph_copy_2d_bytes`, `copy_blocks_u8`.
- mistralrs-quant:
  - AFQ embedding (45);
  - GGUF MoE LoRA gate/up-pair and down gemv (24), `launch_mmq_quantize_glu_q8_1_*` (3),
    `launch_moe_weighted_reduce_flat_{f16,bf16}_input`;
  - packed affine / marlin affine (6);
  - blockwise FP8 cutlass/deepgemm/mma/fused rms (17);
  - routed/dynamic LoRA (15);
  - NVFP4 cutlass (6);
  - `fused_split_glu_*`, `fused_{split_,}glu_quantize_bf16` (5).
- paged-attn: `copy_blocks_u8`.

## Gates (final window, nvcc-free binary unless noted)

Final binary: titan-094 db0d27755, `--features oxide` (target-094-oxide), window 3 unless noted. Harness:
sync/w094 (gate1.sh, gate23.sh, gate4.sh, tools.sh, live094.py, collect094.sh). `collect094.sh` is m3/collect.sh
recording `reasoning_content + content`. v0.9.4's reasoning parser moves the thinking text to
`reasoning_content`, and ktrace returned it in both fields. The concatenation equals the reference strings.
`TITAN_NVCC_ONLY=warn` was set on the new binary and reported no hits.

| gate | result |
|---|---|
| (1) 35B MTP off, 40×256 vs m3/out/q35-prof.json | **40/40** (83.5 tok/s) |
| (1) 35B MTP=2, 40×256 vs m6/out/h-off.json | **40/40** (103.8 tok/s) |
| (1) same harness, ktrace binary (bin/mistralrs-titan-pc = e3fb03b) | 40/40 and 40/40 (thinking text doubled by ktrace's duplicate `content`) |
| (2) prefix cache, 3-turn live check (~11k tokens; tool step, then new user message) | turn 2 prompt 0.27 s (wall 1.81 s incl. 96 decoded tokens), turn 3 prompt **0.68 s** (wall 1.97 s); cold turn 1 32.9 s. ktrace on the same script: 0.39 s / 1.00 s |
| (2) usage.cached_tokens | 10752 of 10883 prompt tokens on a repeat |
| (3) 3× 13k back-to-back + 27.7k | 13017 tok ×3: 205, then 9736 and 11003 tok/s (prefix hits at 12800); 27708 tok in 65.5 s (resume at 10752). No OOM, no panic |
| (4) 80B 8-prompt vs m4/next/out/w2-8-final.json | **8/8** |
| (4) gpt-oss-20b stock and tiered 50% vs m5/oss (fafce74) | **8/8** stock, **8/8** t50, 8/8 stock vs t50 |
| (5) decode tok/s, same window, same harness vs ktrace | MTP=2 103.8 vs 97.5 (+6.5%); MTP off 83.5 vs 81.8 (+2.1%). No regression |
| (5) chat + tool calls (opencode-shaped `read` tool, temp 0 and 0.6, tool result round trip) | PASS: valid `read` call with `filePath`, answer uses the tool result |
| (5) Anthropic `/v1/messages` | PASS (thinking + text blocks) |
| (6) `--features oxide` links | yes, 3m30s incremental (TITAN_OXIDE_DIR, CUDA_HOME=nocuda-bin, no nvcc on PATH) |
| (6) `--features cuda` links | yes: 29m56s full nvcc build (window 2), 3m33s incremental on the final commit; `mistralrs --version` = 0.9.4 |

Greedy output is byte-identical, so no upstream change had to be re-baselined. 8423e2e (natural-log logprobs)
changes only reported values. The upstream changes that would move greedy numerics are kept away from the titan
models in the oxide build:
- attention_prep.cu's new RMS reduction order (qk-rope adapter on the 84b53bf kernel);
- the GDN rewrite (titan_gdn);
- the tokenizer rewrite (titan_tokenizer).

Window 1 also passed gates 1, 2 and 3 on an earlier binary (w1g1b: 40/40 + 40/40, 102.2 / 84.8 tok/s).
Window 2 passed the 80B gate, the tools gate and the nvcc build.

## Downtime

| window | time | service down |
|---|---|---|
| 1 | 12:03:16–12:59:46 | 56m30s: 4 build iterations, smoke tests (TITAN_NVCC_ONLY=warn), gate 1, gates 2+3. The window ran to its deadline, because the stop request was not queued. |
| 2 | 13:30:44–14:06:33 | 35m49s: oxide rebuild, gate 4, tools, the nvcc build (29m56s) |
| 3 | 14:37:03–15:01:47 | 24m44s: final oxide build, gates 4, 1 (+ktrace A/B), 2+3, tools, incremental nvcc build |

Total: 1h57m in 3 windows (RULES-agents.md: ≤3 windows per task, ≥30 min between them).

## Conflict forecast for the feature branches (rebase onto titan-094)

All four branch off ktrace at da11fc3.
- **cpu-vnni** (2 commits, `titan_cpu.rs` only): clean. titan-094 does not touch titan_cpu.rs.
- **lfu-async** (1 commit: titan_tiered.rs, a new titan_upload.rs, `gguf/mod.rs` +2): one trivial conflict in the
  `gguf/mod.rs` module list (where the titan modules sit next to upstream's `packed_affine`/`weight_source`).
  titan_tiered.rs is unchanged in titan-094.
- **miss-skip** (1 commit: a new titan_miss_skip.rs, qwen35 +36, titan_tiered +15, models/mod.rs +1): the models/mod.rs
  line conflicts with the rewritten module list (trivial). The qwen35 hunks should auto-merge; titan-094 changed that
  file in 5 small places (hybrid cache API, titan_gdn import).
- **ngram-spec** (1 commit): real work.
  - Its `gdn/layer.rs` and `gdn/mod.rs` changes (`replay`, `forward_decode_rows_replayable`) must be re-applied by
    hand to `titan_gdn/layer.rs` and `titan_gdn/mod.rs`. git will not follow the copy, and upstream's gdn/ is a
    different implementation.
  - Its `pipeline/gguf.rs` hunks belong in `pipeline/gguf_titan.rs` now.
  - Its 3-line `speculative/driver.rs` change must be redone against v0.9.4's rewritten driver (speculative_plan /
    speculative_commit replace `speculative_proposal_len`).
  - `speculative/mod.rs` (+1, `pub mod ngram`): trivial.

## Deploy

**Not deployed, by RULES-agents.md** ("Do NOT deploy the service binary; report and the main session merges/deploys").
The task text asked for a deploy when all gates pass. All gates passed, so the deploy is staged for the main session:
- `bin/mistralrs-titan-094` = target-094-oxide/release/mistralrs at db0d27755 (sha256 3dc359ec…736c);
- `bin/mistralrs-titan-pc` stays in place for rollback;
- the units are unchanged. To deploy, change only the binary name in `deploy/titan-mistral.service` and
  `~/.config/systemd/user/titan-mistral.service` (every Environment= line stays), then:

```
sed -i 's#/titan-engine/bin/mistralrs-titan-pc#/titan-engine/bin/mistralrs-titan-094#' ~/titan-engine/deploy/titan-mistral.service
cp ~/titan-engine/deploy/titan-mistral.service ~/.config/systemd/user/titan-mistral.service
systemctl --user daemon-reload
# restart via the trap, e.g. flock ~/titan-engine/.gpu.lock bash -c 'trap "systemctl --user start titan-mistral" EXIT; systemctl --user stop titan-mistral'
python3 ~/titan-engine/m4/bigprompt.py 1234 13000; python3 ~/titan-engine/m4/bigprompt.py 1234 13000
```

A visible API difference for clients of :1234: thinking text now arrives only in `reasoning_content` (upstream's
reasoning parser), not duplicated into `content` as ktrace did. opencode and other OpenAI clients read it from there.
