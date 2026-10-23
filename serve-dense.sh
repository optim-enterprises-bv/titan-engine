#!/bin/bash
# Serve the optim 27B (qwen35 dense, GDN hybrid, MTP head) on titan-engine.
#   ./serve-dense.sh <max-seq-len> [extra mistralrs args...]
# Dense qwen35 notes from the repo's own m4/orca runs: no experts, so no tiering;
# every layer must be on the GPU (a CPU layer gives 'device mismatch'); the
# 2048-row big prefill chunks OOM on a dense 27B, so they are off (512-row only).
set -eu
CTX="${1:-16384}"
shift || true
BIN=/home/dingo/src/titan-engine/mistral.rs/target/release/mistralrs
# A directory with ONLY the gguf in it: ~/models also holds the HF
# generation_config.json whose top_k is -1, which the mistral.rs parser rejects
# ("invalid value: integer -1, expected usize").
MODELS=/home/dingo/src/titan-engine/models-optim
FILE="${TITAN_FILE:-Qwen3.8-27B-DevSecOps-QLoRA-abliterated-Q3_K_M.gguf}"
export TITAN_MTP="${TITAN_MTP:-0}"
export TITAN_PREFILL_BIG_CHUNK="${TITAN_PREFILL_BIG_CHUNK:-0}"
export TITAN_RECURRENT_SLOTS="${TITAN_RECURRENT_SLOTS:-8}"
export TITAN_ADMIT="${TITAN_ADMIT:-log}"
export TITAN_DEVMAP_LOG="${TITAN_DEVMAP_LOG:-1}"
export TITAN_CUDA_GRAPHS="${TITAN_CUDA_GRAPHS:-1}"
exec "$BIN" --seed 0 serve -p 1234 --no-ui \
  --paged-attn "${TITAN_PAGED:-on}" \
  --max-batch-size "${TITAN_BATCH:-8}" \
  --prefix-cache-n "${TITAN_PREFIX_N:-16}" \
  --max-seq-len "$CTX" --format gguf -m "$MODELS" -f "$FILE" -n 0:64 "$@"
