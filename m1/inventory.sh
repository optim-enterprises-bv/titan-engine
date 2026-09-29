#!/bin/bash
# List the CUDA kernels a stock mistral.rs run launches (first 3000 launches: load, prefill, decode).
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=$HOME/titan-engine/lib:/usr/local/cuda/lib64
B=$HOME/titan-engine/mistral.rs/target/release/mistralrs
timeout 1200 /usr/local/cuda/bin/ncu --target-processes all -c 3000 --metrics gpu__time_duration.sum --csv \
  --log-file ncu-stock.csv env TITAN_TIERED=${TITAN_TIERED:-0} $B --seed 0 run --paged-attn off --max-seq-len 512 \
  --format gguf -m data -f qwen3coder30b-first24.gguf -i "Write a Rust function that adds two numbers." > ncu-stock.out 2>&1
echo "ncu exit=$?"
