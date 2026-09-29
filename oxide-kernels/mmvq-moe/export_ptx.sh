#!/bin/bash
# Copy the gated kernels into the mistral.rs tree that embeds them (default: the mmvq-moe worktree).
set -e
here=$(dirname "$(readlink -f "$0")")
dst=${1:-$HOME/titan-engine/mr-mmvqmoe}/mistralrs-quant/src/gguf/mmvq_moe_oxide.ptx
cp "$here/mmvq_moe.ptx" "$dst"
/usr/local/cuda/bin/ptxas -arch=sm_120a -O3 "$dst" -o /dev/null
echo "$dst: $(grep -c '^.visible .entry\|^\.entry' "$dst") entries, $(stat -c %s "$dst") bytes, ptxas ok"
