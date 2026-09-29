#!/bin/bash
# Build under the memory cap (inside a GPU window only); prints errors and elapsed time.
cd $HOME/titan-engine/oxide-kernels/titan-doorbell
start=$(date +%s)
export CUDA_OXIDE_BACKEND=$HOME/titan-engine/cuda-oxide-fast/librustc_codegen_cuda.so
timeout 900 systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 env CARGO_BUILD_JOBS=2 cargo oxide build --arch sm_120 > build.log 2>&1
rc=$?
grep -E "^error" -A14 build.log | head -150
ls -la titan_doorbell.ptx 2>/dev/null && grep -E "^\.visible \.entry|^\.entry" titan_doorbell.ptx
echo "build rc=$rc in $(( $(date +%s) - start ))s"
exit $rc
