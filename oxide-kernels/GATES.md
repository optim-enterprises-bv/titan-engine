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
| `fmt_mmq` (default) | 555 launches, 410,694,920 bytes, 0 failing; mxfp4/nvfp4 skipped with the reason printed |
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
| `candle-mmq` | 7 of 10 types (q4_0 q4_1 q5_0 q5_1 q8_0 q3_k q4_k): 408 launcher calls and 806,248,232 bytes of dst each, 0 failing; q2_k/q5_k/q6_k blocked (below) |
| `titan-doorbell` | 400 layers, 1599 missed rows (reported, not silent) |

## Not runnable here, with the reason

| crate | status |
|---|---|
| `candle-mmq` q2_k, q5_k, q6_k | The reference `libmoe.a` is sm_120a in all 15 members and those three instances take CUDA 715; an illegal-instruction fault poisons the context, so the default run splits the work one process per type and reports them blocked. No PTX exists for them (candle compiles to SASS only) and no non-`a` image of the archive is on the box. |
| `fmt_mmq --fmt=mxfp4|nvfp4` | Block-scaled MMA, `.target sm_120a` only. The driver loads the module but the instruction is illegal on cc 12.0. Every reference image of these kernels is sm_120a too (145/145 in the PrismML bundle), so there is no runnable oracle either. Their **dequantizers** are verified: `mxfp4` passes 1,074 launches. |

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
