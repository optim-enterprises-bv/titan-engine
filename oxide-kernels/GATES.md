# Kernel gate tally (this box: RTX 5080 Laptop, cc 12.0, driver 615.78.08)

Every `gate*.rs` crate, run against its reference. `PASS` means every compared byte matched.
References live in `reference/` and `*/ref/`, which are gitignored — most of the work of running a
gate is staging them (see "Artifacts" below).

## Passing

| crate | evidence |
|---|---|
| `titan-kernels-oxide` (repo root) | 20,184,168 outputs / 32,774,352 CPU-twin rows bit-identical |
| `mistralrs-paged-attn-a` | 2,538 calls, 5,436,402,887 bytes, 0 failing |
| `mistralrs-quant-a` | 7,004 calls, 572,862,725 bytes, 0 failing |
| `mistralrs-quant-b` | 4,670 calls, 38,465,320 bytes, 0 failing |
| `mistralrs-quant-c` | 20,120 calls, 9,436,413,576 bytes, 0 failing |
| `mistralrs-paged-attn-b` | 8,610 calls, 1,783,951,316 bytes, 883/883 instances |
| `mistralrs-core-cuda` | 5,840 calls, 85,026,856 bytes, 0 failing |
| `mmvq-rows` | 810 launches, 2,362,720 bytes, ten formats |
| `mmq-rows` | 144 cases, 13,271,040 bytes (via `RUN_MMQ_ROWS=1` in `mistralrs-quant-c`) |
| `fmt_mmq` (default) | **1,015 launches, 562,321,320 bytes, 0 failing** -- iq4_nl, iq4_xs, and MXFP4 through the int8 path (mxfp4q8), which runs on cc 12.0 |
| `fmt_mmq --fmt=iq4_nl` | 330 launches, 198,296,952 bytes, 0 failing |
| `fmt_mmq --fmt=iq4_xs` | 230 launches, 208,754,192 bytes, 0 failing |
| `candle-quantized` | 5,350 launches, 14,086,960 bytes, 0 failing |
| `candle-reduce` | 10,627 launches, 34,635,265 bytes, 0 failing |
| `candle-mmvq` | 4,440 calls, 4,167,968 bytes, 0 failing |
| `candle-moe` | 1,415 calls, 22,204,528 bytes, 0 failing |
| `gemm` | 3,289 launches, 16,255,088 outputs, worst error/tolerance 0.9 (accuracy gate) |
| `iq2_xxs` `iq2_xs` `iq2_s` `iq3_xxs` `iq4_nl` | 1,074 launches, 2,909,304 bytes each, 0 failing |
| `nvfp4` | 1,074 launches, 3,413,112 bytes, 0 failing |
| `mxfp4` | 1,074 launches, 2,909,304 bytes, 0 failing |
| `iq3_s` | 1,085 launches, 15,608,544 bytes, 0 failing |
| `iq4_xs` | 1,157 launches, 15,612,608 bytes, 0 failing |
| `q1_0` | 1,098 launches: dequantize x3, mmvq x27, quantize x6, mmq x16 + fixup x16 |
| `ptq1_0` | quantize pt x2, fwht x6 + fused x4, mmvq pt x12, quantize mmq x2, mmq x16 + fixup x16 |
| `flash-decode` | PASS |
| `mmvq-moe` | 3,840 comparisons, 33,782,400 output words (135 MB), 0 differ -- against Titan's own deployed kernels (section C) |
| `candle-mmq` | **10 of 10 types**: 408 launcher calls and 806,248,232 bytes of dst each, 0 failing. q2_k/q5_k/q6_k needed the reference recompiled for sm_120 -- see `candle-mmq/REFERENCE.md` |
| `titan-doorbell` | 400 layers, 1599 missed rows (reported, not silent) |

## Not runnable here, with the reason

| crate | status |
|---|---|
| `fmt_mmq --fmt=mxfp4|nvfp4` (block-scaled instances) | **Requires sm_120a hardware.** The MMQ kernels for these two formats are written with block-scaled MMA (`mma.sync.aligned.kind::mxf4nvf4.block_scale`), which ptxas rejects for `.target sm_120` and which takes CUDA 715 at run time on cc 12.0. The reference cannot be rebuilt for plain sm_120 either: llama.cpp gates the same instruction on the Blackwell arch, so every image of it is sm_120a (145/145 in the PrismML bundle). Neither side of the comparison can execute here. Their **dequantizers** are verified: `mxfp4` passes 1,074 launches, and IQ4_NL/IQ4_XS MMQ pass 555 launches in the same crate. |

## Open, not passing

| crate | status |
|---|---|
| `fmt_mmq --fmt=nvfp4q8` (opt-in) | **Runs, 217 bytes of 169,161,784 differ** -- all in `rate=256` cases (the harness injects inf / nan / denormal / 3e38 activations), all <=1 ULP. The CUDA 719 trap it used to die on was `j0 += 8` instead of `j0 += Jc::<J>::JSTEP`, overflowing `sum[64]` into a Rust bounds check whose panic lowers to `trap;`. Arithmetic now matches the reference's nesting; see `fmt_mmq/NVFP4-INT8.md` for what is ruled out (exact `div.rn` made it *worse*: the reference is fast-math). |
| `titan-kernels-oxide`, `indexed_moe` sections | **Needs `m0/ref/indexed_moe.ptx`, `m1/ref/indexed_moe.ptx` and the `data/` tensors**, none of which exist any more. The rest of that gate passes: 20,184,168 outputs / 32,774,352 CPU-twin rows. |

## Artifacts: what has to be staged, and the traps

`reference/` and `*/ref/` are gitignored, and the restore snapshot at
`/home/dingo/restore/home/dingo/titan-engine/oxide-kernels/` is frequently **hollowed** (files that keep
their size but are all zeros) — a hollow cubin either crashes `cuModuleLoadData` or silently yields
zero kernels.

- Stage per-crate references from the snapshot, then check the first 4 bytes: ELF (`7f454c46`) or
  reject. Several hundred MB of what looks like the right files are zeros.
- Mine what is hollow: `titan-kernels-oxide/scripts/mine-mmvq-ref.py` (mul_mat_vec_q),
  `mine-convert-ref.py` pattern (dequantize_block), and a payload scanner for `mul_mat_q` — the
  PrismML sm_120 bundle carries all of them, split across zstd payloads per type, so search every
  payload rather than taking the largest.
- Most gates read their own PTX and their references from the **deployed** tree
  (`$HOME/titan-engine/oxide-kernels/...`), not from this checkout. When a run panics on a missing
  `.ptx` that clearly exists, that is why.
- Per-format crates build the mmvq symbol as `<type, ncols_dst, has_fusion, has_gate, small_k,
  halve_iters>`; the only reference that executes on sm_120 is the PrismML one, which has that
  6-param form. The pinned b10934 cubin has the 5-param name but its kernel writes nothing.
