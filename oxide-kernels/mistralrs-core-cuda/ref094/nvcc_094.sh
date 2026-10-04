#!/bin/bash
# nvcc 13.3 rebuild of mistral.rs v0.9.4 mistralrs-core/src/cuda/{input_packing,graph}.cu with mistralrs-core/build.rs's flags
# (cudaforge: sm_120a, --default-stream per-thread); compares the SASS with the cudaforge objects the gate links
# (reference/mistralrs-core-094, from ~/titan-engine/cuda-build-094/core). usage: nvcc_094.sh <mistralrs-core dir> <out dir>
set -euo pipefail
src=$1; out=$2; mkdir -p $out; T=$HOME/titan-engine/oxide-kernels/tools/cuobjdump; R=$HOME/titan-engine/oxide-kernels/reference/mistralrs-core-094
F="-std=c++17 -O3 -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math --compiler-options -fPIC -gencode=arch=compute_120a,code=sm_120a --default-stream per-thread -allow-unsupported-compiler -ccbin /usr/bin/g++-15"
for s in input_packing graph; do
  (cd $src && /usr/local/cuda/bin/nvcc $F --cubin -o $out/$s.cubin src/cuda/$s.cu && /usr/local/cuda/bin/nvcc $F --ptx -o $out/$s.ptx src/cuda/$s.cu)
  ref=$(ls $R/$s-*.o); (cd $out && $T -xelf all $ref > /dev/null 2>&1; mv $(basename ${ref%.o}).1.sm_120a.cubin $s.ref.cubin 2>/dev/null || true)
  if [ -e $out/$s.ref.cubin ]; then
    diff -q <($T -sass $out/$s.cubin | grep -v "code for\|Fatbin\|^$" | sed "s/_GLOBAL__N__[0-9a-f]*_/_GLOBAL__N__H_/") <($T -sass $out/$s.ref.cubin | grep -v "code for\|Fatbin\|^$" | sed "s/_GLOBAL__N__[0-9a-f]*_/_GLOBAL__N__H_/") > /dev/null && echo "$s: SASS identical to the cudaforge object" || echo "$s: SASS DIFFERS"
  else echo "$s: no device code in the reference object"; fi
done
