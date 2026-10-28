#!/bin/bash
# Serve the Q4_K_M 27B (qwen35 dense, GDN hybrid, MTP head) on titan-engine, single user, up to 256k context.
#   ./serve-q4.sh [max-seq-len, default 262144] [extra mistralrs args...]
# The file is the stock Q4_K_M with only output.weight and the 33 Q6_K ffn_down tensors
# requantized to Q5_K (every other tensor byte-identical): 15.5 GB instead of 16.0, so the
# weights fit the 16 GB card with room for KV. token_embd stays on the CPU (titan loads it there).
# --max-batch-size only sizes the device map; --max-seqs caps the sequences the scheduler holds at
# once. Two (an agent and one subagent) decode together with host KV, each keeping its own cache and
# MTP drafts (+30% tokens/s at short context, +20% at 12k), every token as it would be alone. Each
# held sequence pins a ~150 MiB GDN recurrent slot on the GPU, allocated at load: growing the pool
# on a full GPU fails with OOM. Further requests wait in the admission queue.
# TITAN_KV_HOST=1 keeps the 16 attention layers' KV in host RAM as F8E4M3 (32 KiB/token, 8 GiB at
# 256k): decode attention runs on the CPU, prompt chunks stream it to the GPU. TITAN_KV_HOST=0 keeps
# it on the GPU (faster at short context, but only ~6k tokens fit next to the weights).
set -eu
CTX="${1:-262144}"
shift || true
BIN="${TITAN_BIN:-${TITAN_BIN:-/usr/local/bin/mistralrs}}"
MODELS=/home/dingo/src/titan-engine/models-q4fit
FILE="${TITAN_FILE:-Qwen3.8-27B-DevSecOps-QLoRA-abliterated-Q4_K_M-fit.gguf}"
export TITAN_MTP="${TITAN_MTP:-1}"
export TITAN_KV_HOST="${TITAN_KV_HOST:-1}"
export TITAN_PREFILL_BIG_CHUNK="${TITAN_PREFILL_BIG_CHUNK:-0}"
SEQS="${TITAN_SEQS:-2}"
export TITAN_RECURRENT_SLOTS="${TITAN_RECURRENT_SLOTS:-$SEQS}"
# host-RAM prefix cache: a 30k-token agent turn needs ~1.6 GiB (FP8 KV plus GDN resume points), above
# the 1280 MiB default, so each turn fell back to an old shorter entry and re-ran ~10k prompt tokens;
# four entries so the subagents an agent fans out do not evict its own conversation
export TITAN_PREFIX_CACHE_MIB="${TITAN_PREFIX_CACHE_MIB:-6144}"
export TITAN_PREFIX_CACHE_ENTRIES="${TITAN_PREFIX_CACHE_ENTRIES:-4}"
export TITAN_ADMIT="${TITAN_ADMIT:-log}"
export TITAN_DEVMAP_LOG="${TITAN_DEVMAP_LOG:-1}"
export TITAN_CUDA_GRAPHS="${TITAN_CUDA_GRAPHS:-1}"
exec "$BIN" --seed 0 serve -p 1234 --no-ui \
  --paged-attn off \
  --max-batch-size "${TITAN_BATCH:-1}" \
  --max-seqs "$SEQS" \
  --prefix-cache-n "${TITAN_PREFIX_N:-1}" \
  --max-seq-len "$CTX" --format gguf -m "$MODELS" -f "$FILE" -n 0:64 "$@"
