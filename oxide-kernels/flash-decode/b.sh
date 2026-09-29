#!/bin/bash
# Build under the memory cap (inside a GPU window only: RULES-agents.md); prints errors and elapsed time.
cd $HOME/titan-engine/oxide-kernels/flash-decode
start=$(date +%s)
export CUDA_OXIDE_BACKEND=$HOME/titan-engine/cuda-oxide-fast/librustc_codegen_cuda.so
systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 env CARGO_BUILD_JOBS=2 cargo oxide build --arch sm_120 > build.log 2>&1
rc=$?
grep -E "^error" -A12 build.log | head -150
grep -E "not unrolled" build.log | sort | uniq -c | head
echo "build rc=$rc in $(( $(date +%s) - start ))s"
