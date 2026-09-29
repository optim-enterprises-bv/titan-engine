#!/bin/bash
cd $HOME/titan-engine/oxide-kernels/candle-mmq
start=$(date +%s)
systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 env CARGO_BUILD_JOBS=2 cargo oxide build --arch sm_120 > build.log 2>&1
rc=$?
grep -E "^(error|warning: unused)" -A12 build.log | head -100
echo "build rc=$rc in $(( $(date +%s) - start ))s"
