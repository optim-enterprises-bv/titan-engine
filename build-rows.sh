#!/bin/bash
# Build the mmvq-rows cuda-oxide kernels for sm_120 and report the PTX artifact.
set -u
export PATH="/usr/local/cuda/bin:$PATH"
cd /home/dingo/src/titan-engine/oxide-kernels/mmvq-rows || exit 1
start=$(date +%s)
systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 \
  env CARGO_BUILD_JOBS=2 cargo oxide build --arch sm_120 > build.log 2>&1
rc=$?
grep -E "^error" -A10 build.log | head -40
echo "build rc=$rc in $(( $(date +%s) - start ))s"
ls -la --time-style=+%H:%M *.ptx 2>/dev/null | awk '{print $5, $6, $7}'
for p in *.ptx; do
  [ -f "$p" ] || continue
  echo "--- $p: $(grep -c '^.visible .entry\|^\.entry' "$p") entries, $(stat -c %s "$p") bytes"
done
echo "--- embedded PTX currently in mistral.rs ---"
ls -la --time-style=+%H:%M /home/dingo/src/titan-engine/mistral.rs/mistralrs-quant/src/gguf/mmvq_rows_oxide.ptx | awk '{print $5, $6}'
