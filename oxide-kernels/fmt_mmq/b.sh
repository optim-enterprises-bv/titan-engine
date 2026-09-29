#!/bin/bash
# sm_120a: the MXFP4 / NVFP4 MMQ uses mma.sync ... kind::mxf4[nvf4].block_scale (and cvt ... ue8m0x2 / e2m1x2).
cd $HOME/titan-engine/oxide-kernels/fmt_mmq
export CUDA_OXIDE_BACKEND=$HOME/titan-engine/cuda-oxide-fast/librustc_codegen_cuda.so
flock $HOME/titan-engine/.build.lock systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 env CARGO_BUILD_JOBS=2 cargo oxide build --arch sm_120a > build.log 2>&1
rc=$?
grep -E "^error|^warning: unused|not unrolled" -A12 build.log | head -80
echo "build rc=$rc"
