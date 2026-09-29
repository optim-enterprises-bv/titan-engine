#!/bin/bash
# Copy the gated kernels into the mistral.rs tree that embeds them (default: the idle-cut worktree).
set -e
here=$(dirname "$(readlink -f "$0")")
dst=${1:-$HOME/titan-engine/mr-idle}/mistralrs-quant/src/gguf/titan_doorbell_oxide.ptx
cp "$here/titan_doorbell.ptx" "$dst"
/usr/local/cuda/bin/ptxas -arch=sm_120a -O3 "$dst" -o /dev/null
echo "$dst: $(grep -c '^.visible .entry\|^\.entry' "$dst") entries, $(stat -c %s "$dst") bytes, ptxas ok"
