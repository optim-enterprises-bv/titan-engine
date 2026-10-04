#!/bin/bash
# nvcc reference of the mistral.rs v0.9.4 paged-attention sources (titan model-swap 2c4b140e3 / titan-094 f601844a6,
# identical .cu): the flags of mistralrs-paged-attn/build.rs + cudaforge 0.1.6 (sm_120a, --default-stream per-thread).
# Usage: nvcc_094.sh <mistralrs-paged-attn dir> <out dir> [file stems...]; writes <stem>.o (host launchers + SASS),
# <stem>.cubin (extracted) and <stem>.ptx. Compare the SASS with ~/titan-engine/cuda-build-094 (sass_cmp094.sh).
set -euo pipefail
src=$1; out=$2; shift 2
stems=${*:-flashinfer_decode copy_blocks_kernel}
mkdir -p $out
hash=$(cd $src && python3 - <<'PY'
# cuda_header_hash("src/cuda", &["src/cuda/fa3"]) of build.rs (FNV-1a over sorted *.cuh/*.h/*.hpp paths + bytes)
import os
h=0xcbf29ce484222325
def upd(b):
    global h
    for x in b:
        h ^= x; h = (h * 0x100000001b3) & 0xffffffffffffffff
def visit(p):
    if p == "src/cuda/fa3": return
    if os.path.isdir(p):
        for e in sorted(os.listdir(p)): visit(os.path.join(p, e))
        return
    if os.path.splitext(p)[1] not in (".cuh", ".h", ".hpp"): return
    upd(p.encode()); upd(open(p, "rb").read())
visit("src/cuda")
print("%016x" % h)
PY
)
F="-std=c++17 -O3 -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math --compiler-options -fPIC -DMISTRALRS_CUDA_HEADER_HASH=$hash -DENABLE_FP8 -gencode=arch=compute_120a,code=sm_120a --default-stream per-thread -allow-unsupported-compiler -ccbin /usr/bin/g++-15 -Xcompiler -fPIC"
for s in $stems; do
  ( cd $src && /usr/local/cuda/bin/nvcc $F -c -o $out/$s.o src/cuda/$s.cu && /usr/local/cuda/bin/nvcc $F --ptx -o $out/$s.ptx src/cuda/$s.cu )
  ( cd $out && $HOME/titan-engine/oxide-kernels/tools/cuobjdump -xelf all $s.o >/dev/null && mv $s.1.sm_120a.cubin $s.cubin )
  echo "$s: $(ls -la $out/$s.o | awk '{print $5}') bytes .o, $(grep -c '^.entry\|^\.visible .entry\|\.entry ' $out/$s.ptx) entries"
done
