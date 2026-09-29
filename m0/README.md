# M0 — oxide Q4_K expert GEMV, bit-matched to mistral.rs

Gate: PASSED 2026-09-26. `titan-kernels-oxide`'s `q4k_q8_1_moe_gemv` (cuda-oxide b9847e9,
nightly-2026-08-28, `cargo oxide build --arch sm_120`) is bit-identical to mistral.rs
84b53bf `indexed_moe_forward_q4k_q8_1` (nvcc 13.3, `-O3 --use_fast_math`, sm_120) on
Qwen3-Coder-30B-A3B UD-Q4_K_XL `blk.0.ffn_gate_exps` (K=2048) and `blk.0.ffn_down_exps`
(K=768), all 128 experts, 8 inputs: 2,881,536 outputs, 0 bit mismatches.

- `ref/indexed_moe.ptx` — the reference, compiled with mistral.rs's own nvcc flags.
- `ref/q4k.sass` — its machine code. **The SASS is the spec, not the PTX**: nvcc emits
  `mul.ftz; mul.ftz; sub.ftz` without `.rn`, and ptxas contracts them to
  `FMUL p = sumf_m*dm.y; FFMA r = sumf_d*dm.x - p`. Mirroring the PTX left ~70% of outputs 1 ulp off.
- `oxide/titan_kernels_oxide.ptx` — ours.
- `data/` (not committed) — the two tensors, extracted with gguf-py.

Reproduce: `cd titan-kernels-oxide && cargo oxide build --arch sm_120 && cp titan_kernels_oxide.ptx ../m0/oxide/ && target/release/titan-kernels-oxide`
(needs ~250 MB VRAM; runs beside alive). cuobjdump is not in titan's CUDA install; Triton ships one
(chmod a private copy).
