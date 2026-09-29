#!/bin/bash
# Copy the gated rows kernels into the mistral.rs tree that embeds them (default: the M6 worktree).
set -e
here=$(dirname "$(readlink -f "$0")")
dst=${1:-$HOME/titan-engine/mr-m6}/mistralrs-quant/src/gguf/mmvq_rows_oxide.ptx
cp "$here/mmvq_rows.ptx" "$dst"
/usr/local/cuda/bin/ptxas -arch=sm_120a -O3 "$dst" -o /dev/null
echo "$dst: $(grep -c '^.visible .entry\|^\.entry' "$dst") entries, $(stat -c %s "$dst") bytes, ptxas ok"
