# cuda-oxide migration off b9847e9 to the cuda-rust tip (2026-10-12)

The kernel manifests live in the **nested** `oxide-kernels/` git repository
(untracked from this outer `titan-engine` repo, per RULES/skill notes), so the
migration commit is there and this patch is its mirror.

## Commits

| repo | branch | commit | note |
|---|---|---|---|
| `oxide-kernels/` (nested) | `cuda-rust-tip` | start `c64373de80e69c6aba448c7530160b76405eb831` | `master` at the time |
| `oxide-kernels/` (nested) | `cuda-rust-tip` | end `ab330a597316e323d8f3572d263e8b9e99b8dec4` | 41 files, migration |
| `titan-engine/` (outer) | `cuda-rust-tip-migration` | start `2c5d2f5fb52fa1f42d90a9fcfda35431e389e9c4` | |

`oxide-kernels` has no remote configured and `titan-work` is not reachable from
this box (https needs credentials); only `titan-engine` `origin` (ssh) is. This
patch is therefore the published form of the change.

## What changed

- 80 `cuda-device`/`cuda-host` pins: `rev = "b9847e9"` ->
  `rev = "d141118c05f705fcf932a8e9077dc714477edc4d"` (40 manifests).
- 41 `cuda-core = "0.3.1"` requirements -> the git dep at the same tip rev.
  A `^0.3.1` crates.io requirement cannot be satisfied by a 0.4.0 git patch, and
  `kdiff` pins 0.3.1 for every crate, so the requirement itself had to move.
  That leaves exactly one `cuda-core` 0.4.0 in the graph.

## Toolchain / backend

- `nightly-2026-10-11` is required (tip `cuda-host` uses the `reflection`
  intrinsic; on `nightly-2026-08-28` it fails `error[E0433]: cannot find
  'reflection' in 'intrinsics'`).
- `cuda-oxide-fast/src` checked out at `d141118c` (the quadratic `detect.rs`
  fix is upstream at the tip, so patch 0001 was dropped).
- `librustc_codegen_cuda.so` rebuilt (1m36s) and `cargo-oxide` rebuilt (44s).

## Result

- All 39 kernel-crate modules codegen at sm_120; API drift = zero.
- Engine `target-x86-oxide/release/mistralrs` rebuilt (rc=0, 270s), links only
  `libcuda` (noblas), 0 nvcc mentions; generates `TITAN_OK`.
- littlebit gate: 9/9, worst relative 5.878e-7 (unchanged).

Full per-crate and per-gate detail: see the parent session's report.
