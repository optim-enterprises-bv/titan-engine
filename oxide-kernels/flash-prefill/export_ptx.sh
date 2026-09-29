#!/bin/bash
# Copy the gated kernels into the mistral.rs tree that embeds them (default: the flash-prefill worktree).
set -e
here=$(dirname "$(readlink -f "$0")")
dst=${1:-$HOME/titan-engine/mr-pf2}/mistralrs-core/src/attention/flash_prefill_oxide.ptx
cp "$here/flash_prefill.ptx" "$dst"
/usr/local/cuda/bin/ptxas -arch=sm_120a -O3 "$dst" -o /dev/null
echo "$dst: $(grep -c '^.visible .entry\|^\.entry' "$dst") entries, $(stat -c %s "$dst") bytes, ptxas ok"
