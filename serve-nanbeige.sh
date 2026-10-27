#!/bin/bash
# Serve Nanbeige4.3-33B-UNCENSORED Q4_K_M (dense Llama arch, 52 layers, GQA 8 KV heads) on the
# 16 GB RTX 5080 Laptop. The Q4_K_M file is ~20 GB > VRAM, so this leans on titan's device map to
# split layers GPU/CPU. NO MTP head and NO GDN recurrent slots here (both are Qwen3.5-specific) —
# this is a pure-attention Llama, so every layer carries a real KV cache.
#   ./serve-nanbeige.sh [max-seq-len, default 8192] [extra mistralrs args...]
#
# Knobs to tune after reading the TITAN_DEVMAP_LOG split line:
#   TITAN_KV_HOST=1  -> attention KV lives in host RAM (F8E4M3), decode attn runs on CPU. Frees VRAM
#                       so more weight layers fit the card. Default on, since weights are the bottleneck.
#   DEVLAYERS="0:N"  -> force N of 52 layers onto GPU 0, rest to CPU (passed as -n). Omit for auto map.
set -eu
CTX="${1:-8192}"
shift || true
BIN="${TITAN_BIN:-/usr/local/bin/mistralrs}"
MODELS=/home/dingo/src/titan-engine/models-nanbeige
FILE="${TITAN_FILE:-Nanbeige4.3-33B-UNCENSORED-Q4_K_M.gguf}"

export TITAN_MTP=0
export TITAN_KV_HOST="${TITAN_KV_HOST:-1}"
export TITAN_DEVMAP_LOG="${TITAN_DEVMAP_LOG:-1}"
export TITAN_CUDA_GRAPHS="${TITAN_CUDA_GRAPHS:-1}"

ARGS=(--seed 0 serve -p 1234 --no-ui
  --paged-attn off
  --max-batch-size 1
  --max-seqs 1
  --prefix-cache-n 1
  --max-seq-len "$CTX"
  --format gguf -m "$MODELS" -f "$FILE")

# Manual layer split only if DEVLAYERS is set; otherwise automatic device mapping.
if [ -n "${DEVLAYERS:-}" ]; then
  ARGS+=(-n "$DEVLAYERS")
fi

exec "$BIN" "${ARGS[@]}" "$@"
