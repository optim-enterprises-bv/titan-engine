# m7 plan: dynamic memory tiers without a fluctuating decode step (2026-09-29)

## The rule

The VRAM / RAM / disk split can change **per request**. It does not change **per step**. Placement is decided
while a request is admitted and its prompt is prefilled. From the first decode step to the end of the request,
the resident expert set, the KV reservation and the staging ring stay fixed. A step never waits on a copy it
did not issue for itself.

Why: at MTP=2 a decode step is about 7.5 ms of GPU work inside an 18.9 ms window (133 tok/s). Pinned H2D runs at
42-45 GB/s, so moving 100 MiB of experts takes about 2.3 ms, a large share of a whole step. A step that waits on
a disk read (100-400 µs, with no upper bound under load) is worse. The 54% of the nsys window that looks idle is
sync latency in the pipeline, not free time. Work placed there makes the next step late. For agent clients, p99
matters more than p50: a config at 133 tok/s p50 and 40 tok/s p99 does worse in real use than one at a flat 110,
because slow turns set off timeouts and retries. The service logs already show a client retrying a slow turn
three times.

## What the code does today

- `TITAN_TIERED_GPU_FRACTION=auto` (`mistralrs-quant/src/gguf/titan_tiered.rs`, `plan_auto`) sets the resident
  share **once, before any weight loads**. It is stored in a `OnceLock`, and every layer gets the same share.
- Its KV reserve (`mistralrs-core/src/pipeline/gguf_titan.rs`, `set_auto_extra_reserve`) is sized for
  `max_batch_size x max_seq_len`. The service passes `--max-seq-len 65536`, so the full 64k KV room is kept free
  for the whole life of the process. A 500-token chat pays for it in expert slots it could otherwise hold.
- The PFS staging ring (`titan_pfs.rs`) is sized per prompt from the free memory above a headroom
  (`HEADROOM` + per-task + per-context KiB). The first non-streamed forward frees it, so decode sees exactly
  the static placement. If there is no room it falls back to CPU-twin prefill and does not fail. This is the
  property to keep.
- There are already some dynamic policies, all of which **run on the step path**:
  - `TITAN_TIERED_POLICY=lru|lfu` admits a miss at decode;
  - `TITAN_TIERED_FINGERPRINT=M` re-slots experts at the first decode step after a prefill;
  - `TITAN_TIERED_ASYNC=1` makes the `lfu` uploads asynchronous.

  The service runs `static`. Scope (b) replaces these policies rather than tuning them.

## Status

| scope | code | gate |
|---|---|---|
| (a) admission | done: `mistralrs-core/src/titan_admit.rs`, called from `engine/add_request.rs`; `TITAN_ADMIT=1` refuses, `TITAN_ADMIT=log` only logs | not run: needs the GPU (`BENCH_ADMIT=1 bench/run.sh ...` adds the probe) |
| (b) per-request partition | not started: the arena spike needs the GPU | - |
| (c) fault accounting | done: `mistralrs-core/src/titan_faults.rs`, `TITAN_FAULT_LOG=1` | not run |
| (c) `mlock` / lookahead budget | not started | - |

The working-set terms of (a) are the PFS ring's measured ones, not a fit. Run the service once with
`TITAN_ADMIT=log TITAN_PREFILL_MEMLOG=1` over the bench's prompts and compare each logged estimate with the pool
peak before turning refusal on.

## Scopes

Each scope sits behind its own env flag, off by default, and has its own gate. To revert a scope, unset its
flag. If the flag already shipped as the default, revert its merge commit. The scopes are ordered by risk. (b)
does not start until (a) has passed its gate.

### (a) KV-first admission and refusal over budget

**What:** before prefill, compute what the request needs on the device:
- the KV bucket, `ceil((prompt_tokens + max_tokens) / 8192) x 8192` tokens;
- the ring headroom that `titan_pfs::allocate` would ask for;
- the graph reserve.

If the total does not fit in the memory left after the static placement, refuse the request with a clear
error (the context-length error the OpenAI API returns) instead of failing partway through prefill or decode.
The prefix-cache host tier gets the same check against RAM.

**Flag:** `TITAN_ADMIT=1`.

**Changes the output:** no. **Changes performance:** no. Decode is not touched, and admission is arithmetic on
numbers the loader already has.

**Cost (estimate):** 2-3 days.
- One admission function in the scheduler path.
- A byte-accounting helper that shares its formulas with `gguf_titan.rs` and `titan_pfs.rs`, so the numbers
  cannot drift apart.
- Tests at the boundary: exactly fits, one bucket over, and `max_tokens` unset.

**Gate:**
- The standing bench is within its 3% rule on every metric, with identity unchanged.
- A new over-budget probe (prompt + `max_tokens` above the room) returns the error in under 100 ms, and the
  service stays up for the next request.
- The 28k and 40k prompt tiers still run.

### (b) Per-request partition with the decode shape frozen

**What:** choose the resident count per **request class** instead of per load, and trade KV room for expert
slots.

- Classes follow the KV buckets, but coarser: short (≤ 8k), agent (≤ 32k) and long (≤ 64k). The exact edges
  are set from the bench.
- A short class returns the unused KV room to expert slots. A long class gives slots back.
- While the GPU prefills, a copy stream uploads the promoted experts, and demotions just drop slots. Prefill
  runs at about 2,650 tok/s, so a 13k prompt takes about 5 s, which is ample time for the copies.
- The new slot map is published at the prefill/decode boundary behind a CUDA event. If the copies are not done
  by the first decode step, that step **keeps the old map** and does not wait. A step that needs an expert
  outside the map uses the CPU twin, whose cost is fixed and known.
- Hysteresis:
  - resize only when the class changes;
  - a request of the same class never moves an expert;
  - there is a minimum residency of N requests before an expert can be demoted.
- A decode floor (for example 110 tok/s on the 8-prompt set) is a hard constraint in the admission function. A
  partition whose predicted miss rate would break the floor is not chosen. The request then runs in the
  largest class that meets the floor, and prompt latency pays for it: users can absorb 20-30% there, but not
  in decode.
- Every admission logs one line: class, resident count, bytes moved, copy time, and the request's decode tok/s.
  A resize can then be attributed to a change in tok/s.

**Main technical risk:** KV and expert slots are separate allocations today, and decode segments are captured
as CUDA graphs (`TITAN_CUDA_GRAPHS=1`). Resizing with `cudaMalloc`/`free` would move pointers and force a
recapture. The plan is one device arena of fixed-size pages, shared by KV buckets and expert slots, allocated
once at load. A resize then only reassigns pages, and the pointers that graphs capture (the slot map, KV
bases) keep their addresses. A 2-day spike on this comes first. If the arena cannot keep captured graphs valid,
(b) falls back to recapturing once per class change (a one-off cost at admission, never on a step) or stops.

**Flag:** `TITAN_TIER_ADMIT=1`, which requires `TITAN_ADMIT=1`.

**Changes the output:** no. Placement never changes the output; this is the tiered path's existing guarantee,
and the byte-identity check enforces it.

**Cost (estimate):** 2-3 weeks.
- The arena spike: 2 days.
- The arena, with KV buckets moved onto it: 1 week.
- The admission partitioner, copy-stream promotion and event-gated map publication: 1 week.
- The bench tier: 2 days.

**Gate:** a new **mixed-class** bench tier: a fixed sequence (short, 13k agent, short, 40k, short, 13k agent,
short), repeated 3 times, cold. It is recorded in `bench/history.jsonl` with reps and spread like the other
metrics.
- **p50 decode tok/s:** within 3% of static (the bench's standing rule).
- **p99 per-step decode latency:** no worse than static on any class. It also must not show the throughput
  stalls that happen while copies are in flight.
- **Short-class decode tok/s:** higher than static. If it is not, (b) has not earned its complexity and does
  not ship.
- **Prompt tok/s:** within 30% of static on the long class.
- **Byte identity:** identical to static.
- **Hygiene:** the contamination flags are clean.

### (c) Disk as prefetch-only

**What:** this applies to models larger than RAM (`TITAN_TIERED_MMAP=1`: the 80B and 120B). Disk reads come
only from the lookahead thread (`TITAN_TIERED_LOOKAHEAD`) and never start from the step path. Three changes:
- The hottest host set, within a RAM budget, is `mlock`ed, so page-cache pressure cannot evict it.
- The lookahead is given a byte budget per step, so it cannot flood the NVMe queue that a miss depends on.
- Major faults per decode step are counted and logged. This makes a step that paid for a disk read visible
  instead of hiding it in the average.

A miss whose page is still on disk can still fault; nothing avoids that. The goal is to make it rare and to
measure it.

**Flag:** `TITAN_DISK_PREFETCH_ONLY=1`.

**Changes the output:** no.

**Cost (estimate):** 1 week.
- The fault accounting: 1 day. It is useful by itself and should ship first, even with the flag off.
- The `mlock` budget and the lookahead budget: 3 days.
- An 80B bench run: 1 day.

**Gate:** on Qwen3-Next-80B, over 3 runs:
- p50 decode no worse than today's 41.5 tok/s, within spread;
- p99 step latency lower;
- the fraction of steps with a major fault lower;
- the 35B bench unchanged (the flag is inert when `TITAN_TIERED_MMAP` is off).

## Out of scope

- Continuous (per-step) migration of any kind, including a tuned `lfu`. It is the thing this plan exists to
  avoid.
- Batching of concurrent users. The admission function is written so it could later reason about more than
  one request, but this plan assumes one at a time, as the engine does today.
