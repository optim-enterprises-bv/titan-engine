#!/bin/bash
# Build under the memory cap (inside a GPU window only); prints errors and elapsed time.
cd $HOME/titan-engine/oxide-kernels/mmvq-moe
start=$(date +%s)
export CUDA_OXIDE_BACKEND=$HOME/titan-engine/cuda-oxide-fast/librustc_codegen_cuda.so
timeout 900 systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 env CARGO_BUILD_JOBS=2 cargo oxide build --arch sm_120 > build.log 2>&1
rc=$?
grep -E "^error" -A14 build.log | head -150
grep -E "not unrolled" build.log | sort | uniq -c | head
grep -E "^warning: unused" -A4 build.log | head -20
ls -la mmvq_moe.ptx 2>/dev/null
[ -f mmvq_moe.ptx ] && echo "entries: $(grep -c '^.visible .entry\|^\.entry' mmvq_moe.ptx)" && grep -c "ld.local\|st.local" mmvq_moe.ptx
echo "build rc=$rc in $(( $(date +%s) - start ))s"
exit $rc
