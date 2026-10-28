# Rebuilding the candle-mmq reference for cc 12.0

`libmoe.a` (in `../reference/candle-ffi/`, gitignored) is the reference side of the differential gate:
candle's CUDA C++ mmq kernels plus their `launch_mmq_gguf_*` C launchers, compiled with nvcc.

The shipped archive is **sm_120a only** (15/15 members) and its kernels were built with a newer
toolchain than the one on this box — `cuobjdump -sass` on the q2_k member reports `arch = sm_120a`,
`code version [1,8]`, and the local `nvdisasm` rejects the instructions outright ("Illegal
instruction found"). On an RTX 5080 Laptop (compute capability **12.0**, i.e. not `12.0a`) the q2_k,
q5_k and q6_k instances take **CUDA 715 (illegal instruction)** at launch, and because that fault
poisons the context, every later type in the same process fails too.

That is a property of the reference, not of the twin: q4_0, q4_1, q5_0, q5_1, q8_0, q3_k and q4_k run
fine from the same archive.

## What was done

The three instances were recompiled from the in-tree sources for plain `sm_120` and put back into the
archive (the original is kept alongside as `libmoe.sm120a-orig.a`). Only the *reference* is rebuilt
here; titan's own kernels stay cuda-oxide/PTX and never go through nvcc.

Two things had to be handled:

1. **`-Xcompiler -fPIC`**, otherwise the link fails with
   `relocation R_X86_64_32 cannot be used against local symbol; recompile with -fPIC`.
2. **The in-tree source is newer than the archive.** It takes a 15th argument, `type_dst`, that the
   gate's 14-argument launcher signature does not pass. Calling the 15-argument entry point through a
   14-argument pointer shifts `stream` and segfaults immediately. A shim exposes the old signature
   over the new body with `type_dst = GGML_TYPE_F32` (0), which is what the gate compares.

```sh
K=/home/dingo/src/titan-engine/mistral.rs/mistralrs-quant/kernels/mmq_gguf
S=/home/dingo/src/titan-engine/oxide-kernels/reference/candle-ffi
for t in q2_k q5_k q6_k; do cat > shim_$t.cu <<EOF
#define launch_mmq_gguf_$t launch_mmq_gguf_${t}_impl
#include "mmq_instance_$t.cu"
#undef launch_mmq_gguf_$t
extern "C" void launch_mmq_gguf_$t(void *tmp_fixup, const void *x, const void *y_q8_1_mmq, void *dst,
                                   int64_t ncols_x, int64_t nrows_x, int64_t ncols_y,
                                   int64_t stride_row_x, int64_t stride_col_dst, int cc, int nsm,
                                   int64_t smpbo, int warp_size_host, void *stream) {
  launch_mmq_gguf_${t}_impl(tmp_fixup, x, y_q8_1_mmq, dst, ncols_x, nrows_x, ncols_y, stride_row_x,
                            stride_col_dst, cc, nsm, smpbo, warp_size_host, 0, stream);
}
EOF
nvcc -c shim_$t.cu -arch=sm_120 -O3 -std=c++17 -I$K -Xcompiler -fPIC -o ${t}_sm120.o
done
cp -n $S/libmoe.a $S/libmoe.sm120a-orig.a
ar d $S/libmoe.a mmq_instance_q2_k-*.o mmq_instance_q5_k-*.o mmq_instance_q6_k-*.o
for t in q2_k q5_k q6_k; do cp ${t}_sm120.o $S/mmq_instance_${t}-sm120.o; done
ar r $S/libmoe.a $S/mmq_instance_q2_k-sm120.o $S/mmq_instance_q5_k-sm120.o $S/mmq_instance_q6_k-sm120.o
```

Then force the gate to relink (`touch src/main.rs`) — cargo will not notice that the archive changed.

## Result

All ten types pass, 408 launcher calls and 806,248,232 bytes of `dst` bit-identical each:

```
pass: 10 (q4_0 q4_1 q5_0 q5_1 q8_0 q2_k q3_k q4_k q5_k q6_k)
```

If the rebuilt archive is ever lost, the three types go back to `BLOCKED` (the per-type run reports it)
rather than producing false mismatches.
