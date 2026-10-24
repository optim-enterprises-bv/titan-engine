#!/bin/bash
# Copy the mmq rows kernels into the mistral.rs tree that embeds them.
set -e
here=$(dirname "$(readlink -f "$0")")
dst=${1:-$here/../../mistral.rs}/mistralrs-quant/src/gguf/mmq_rows_oxide.ptx
cp "$here/mmq_rows.ptx" "$dst"
/usr/local/cuda/bin/ptxas -arch=sm_120a -O3 "$dst" -o /dev/null
echo "$dst: $(grep -c '^.visible .entry\|^\.entry' "$dst") entries, $(stat -c %s "$dst") bytes, ptxas ok"
