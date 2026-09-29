#!/bin/bash
# run_mistral.sh <port> <gguf dir> <gguf file> [extra env...]: mistral.rs (target-q1 build) server under a memory cap.
port=$1; dir=$2; file=$3; shift 3
systemd-run --user --unit=q1-mistral --collect -q -p MemoryMax=10G -p MemorySwapMax=0 \
  --setenv=LD_LIBRARY_PATH=$HOME/titan-engine/lib:/usr/local/cuda/lib64 "$@" \
  $HOME/titan-engine/target-q1/release/mistralrs --seed 0 serve -p $port --no-ui --paged-attn off --max-seq-len 4096 --format gguf -m "$dir" -f "$file"
for i in $(seq 1 300); do curl -sf -m 2 -o /dev/null localhost:$port/v1/models && exit 0; systemctl --user -q is-active q1-mistral || exit 1; sleep 2; done; exit 1
