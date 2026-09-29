#!/bin/bash
cd $HOME/titan-engine/oxide-kernels/iq4_nl
export CUDA_OXIDE_BACKEND=$HOME/titan-engine/cuda-oxide-fast/librustc_codegen_cuda.so
systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 env CARGO_BUILD_JOBS=2 cargo oxide build --arch sm_120 > build.log 2>&1
rc=$?
grep -E "^error|^warning: unused|not unrolled" -A12 build.log | head -80
echo "build rc=$rc"
