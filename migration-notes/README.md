# cuda-oxide migration off b9847e9 to the cuda-rust tip (2026-10-12)

The kernel manifests live in the **nested** `oxide-kernels/` git repository
(untracked from this outer `titan-engine` repo, per RULES/skill notes), so the
migration commit is there and this patch is its mirror.

## Commits

| repo | branch | commit | note |
|---|---|---|---|
| `oxide-kernels/` (nested) | `cuda-rust-tip` | start `c64373de80e69c6aba448c7530160b76405eb831` | `master` at the time |
| `oxide-kernels/` (nested) | `cuda-rust-tip` | migration `ab330a597316e323d8f3572d263e8b9e99b8dec4` | 41 files, migration |
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
- Every build recorded here used `nightly-2026-10-11`, invoked **by path**
  (`~/.rustup/toolchains/nightly-2026-10-11-x86_64-unknown-linux-gnu/bin/cargo`,
  `RUSTUP_TOOLCHAIN=nightly-2026-10-11`) — not the 1.99.0 stable default and not
  `nightly-2026-08-28`. The env var is what wins over each crate's
  `rust-toolchain.toml` (`nightly-2026-08-28`), which is *not* sufficient at the tip.
  Confirmed: `cargo 1.101.0-nightly (29c5daa1a 2026-10-07)`, `rustc 1.101.0-nightly (6f5da406b 2026-10-10)`.

## Result

- All 39 kernel-crate modules codegen at sm_120; API drift = zero.
- Engine `target-x86-oxide/release/mistralrs` rebuilt (rc=0, 269s, 386 539 360 B),
  links only `libcuda` (noblas), 0 nvcc mentions.
- littlebit gate: 9/9, worst relative 5.878e-7 (unchanged; re-run at the tip).

## Post-migration analysis and hardening (2026-10-12, second pass)

### 1. Three crates emit NVVM IR, not PTX — not a regression

`candle-reduce`, `candle-unary` and `gemm` build rc=0 and produce `<stem>.ll` +
`<stem>.options` + `<stem>.target` but **no `<stem>.ptx`**; `cargo oxide inspect`
then fails with `could not read generated PTX <dir>/<stem>.ptx: No such file or directory`.

Root cause is upstream and unchanged by this migration. A module with strict CUDA-libdevice
(`__nv_*`) calls is emitted as **NVVM IR** whenever libdevice cannot be linked:
`should_emit_nvvm_ir = uses_strict_libdevice && !can_ir_link_libdevice`. Neither
`libdevice.10.bc` nor a matching-LLVM-major `llvm-link` exists on this immutable host
(`CUDA_HOME` deliberately points at the `nocuda-bin` stub), so `can_ir_link_libdevice == false`
and the build prints, verbatim (with `CUDA_OXIDE_VERBOSE=1`):

```
=== Detected CUDA libdevice (`__nv_*`) calls; auto-emitting NVVM IR (skip llc) ===
=== Exporting to LLVM IR (NVVM IR mode) ===
=== Skipping llc (libdevice present); consumer owns libNVVM/nvJitLink build ===
```

`select_pre_lowering_backend` and `should_emit_nvvm_ir` are byte-identical at `b9847e9` and at
`d141118c` (verified by diffing the two functions between the revs), so the pin move did not
introduce this. The three crates are exactly the ones whose modules carry `__nv_*` calls
(reduce `__nv_exp`/`__nv_expf`; unary 28 symbols; gemm 8) and exactly the ones with
`.options`/`.target` sidecars — the NVVM-IR branch is the only writer of those files, and no
PTX-emitting crate has them (a one-line discriminator: `ls */<stem>.target`).

Resolution: the committed `.ptx` are the pinned artifact. Its `__nv_*` calls are already
lowered to native instructions (no `.extern .func __nv_*` survives), so it is self-contained
and loads without a libdevice JIT step. `cargo oxide build` deletes it and cannot recreate it
here, so any rebuild must restore it:

```sh
git checkout -- candle-reduce/candle_reduce.ptx candle-unary/candle_unary.ptx gemm/gemm.ptx
```

To make these crates emit PTX again, provide both `libdevice.10.bc` and a matching
`llvm-link` (a real CUDA toolkit, or `CUDA_OXIDE_LIBDEVICE` + `CUDA_OXIDE_LLVM_LINK`).
Recorded in `oxide-kernels/PTX-PROVENANCE.md`.

### 2. Six crates were never buildable on an nvcc-free box — not a regression

`mistralrs-core-cuda`, `mistralrs-paged-attn-a/-b`, `mistralrs-quant-a/-b/-c` fail in
`build.rs` / at link because the nvcc-built C **reference** archives they consume are missing:
`reference/mistralrs-core/libmistralrscuda.a`,
`reference/mistralrs-paged-attn*/libmistralrspagedattention.a`,
`reference/mistralrs-quant/libmistralrsquant.a`. Those paths are `.gitignore`d
(`reference/*/lib*.a`), **no `.a` was ever added in `oxide-kernels`' history**, and none exists
on this box (they are an nvcc artifact; this fork's path is nvcc-free). `ab330a5` touches only
`Cargo.toml` in these crates, so they did not build before the pin move either.

### 3. Hardening against a silently missing/stale kernel

- `oxide-kernels/titan-oxide-ffi/build.rs` now checks **every** `embed!` occurrence (the old
  line-scanner saw only the last one per source line, and `image.rs` packs many per line — the
  hole that let a deleted artifact go unnoticed), requires each artifact to exist and be
  non-empty, warns when one is older than the crate source that generates it, and asserts the
  three pinned libdevice-module `.ptx` files exist and are non-empty (gemm's was covered
  nowhere else).
- `oxide-kernels/check-artifact-freshness.sh` — presence / non-empty / freshness gate over
  every embedded artifact, the 11 candle ports and the gemm copy; `--strict` fails on stale.
- `oxide-kernels/PTX-PROVENANCE.md` — the affected crates, the mechanism, the restore recipe.

`candle-kernels/build.rs` already panics on a missing candle port
(`TITAN_OXIDE_DIR is set but these ports are missing`), which is what caught the deleted
artifacts during the migration; the additions above cover the `titan-oxide-ffi`-embedded set
and the gemm copy, which had no equivalent.

Full per-crate and per-gate detail: see the parent session's report.
