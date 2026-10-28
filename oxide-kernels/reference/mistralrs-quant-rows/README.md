# mistralrs-quant-rows — the M6 rows launcher reference

`mmvq-rows/src/gate.rs` gates every oxide `mmvq_gguf_<q>_<dst>_rows<b>` against **two** nvcc references:

- **(a)** `mmvq_gguf_rows.cubin` — the M6 WIP launcher (`*_plain_cuda1` with `blockIdx.y` offsets), and
- **(b)** `reference/mistralrs-quant/mmvq_gguf.cubin` — the unmodified batch-1 `*_plain_cuda1`, launched
  once per column.

**(a) is unusable here: the only copy on this box is 20,127,744 bytes of zeros.** It reports its full
size but has no device code, exactly like `flashinfer_decode.cubin` in
`reference/mistralrs-paged-attn-094` (see that README). No `.o` or other build of it exists anywhere
(`find /home/dingo -name 'mmvq_gguf_rows*'` returns only hollow copies).

**(b) is real** (20,123,288 bytes, ELF, carries `mmvq_gguf_q3_k_bf16_plain_cuda1`), so run the gate in
plain-only mode, which is also the stronger statement — the rows kernel must equal *b independent
batch-1 launches*, bit for bit:

```sh
cd oxide-kernels/mmvq-rows
systemd-run --user --scope env CARGO_BUILD_JOBS=2 cargo oxide build --arch sm_120   # needs the GPU off prod
ROWS_PLAIN_ONLY=1 ./target/release/mmvq-rows
```

Measured on this box:

```
f32 outputs: 275250 of 293100 finite
mmvq-rows kernel-level kdiff vs nvcc: 810 launches, 2362720 bytes compared, 0 failing -> PASS (bit-identical)
```

That covers all ten formats the gate declares — `q4_0 q4_1 q5_0 q5_1 q8_0 q2_k q3_k q4_k q5_k q6_k`
— across `bf16/f16/f32` destinations, four shapes and `b = 1..8` rows per launch.

What is lost by skipping (a): only the cross-check against the WIP launcher's `blockIdx.y` offsetting.
The correctness property (rows<b> == b batch-1 launches) is what (b) proves, and it is the one that
matters. If a real `mmvq_gguf_rows.cubin` ever turns up, drop it in this directory and run without
`ROWS_PLAIN_ONLY`.
