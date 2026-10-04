#!/bin/bash
# Copy the gated kernels and the planner into the candle tree that embeds them (default: the noblas worktree).
set -e
here=$(dirname "$(readlink -f "$0")")
dst=${1:-$HOME/titan-engine/top-noblas/candle}/candle-core/src/cuda_backend
cp "$here/gemm.ptx" "$dst/gemm_oxide.ptx"
{ echo "// Copied from titan-engine/oxide-kernels/gemm/src/plan.rs by its export.sh: edit there, not here."; cat "$here/src/plan.rs"; } > "$dst/oxide_gemm_plan.rs"
/usr/local/cuda/bin/ptxas -arch=sm_120 -O3 "$dst/gemm_oxide.ptx" -o /dev/null
echo "$dst/gemm_oxide.ptx: $(grep -c '^.visible .entry' "$dst/gemm_oxide.ptx") entries, $(stat -c %s "$dst/gemm_oxide.ptx") bytes, ptxas ok; plan $(md5sum < "$here/src/plan.rs" | cut -c1-8)"
