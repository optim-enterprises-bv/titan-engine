#!/bin/bash
# Same model, two quant files, same context and prompt: single-stream decode comparison.
# `--format gguf` wants -m <dir> -f <file>, and Q4_K_M (16.81 GB) does not fit the 16 GB card,
# so its tail layers go to CPU via -n 0:N.
set -u
BIN=/home/dingo/src/titan-engine/mistral.rs/target/release/mistralrs
run() { # label file layers ctx
  local label=$1 file=$2 layers=$3 ctx=$4
  local dir name
  dir=$(dirname "$file"); name=$(basename "$file")
  systemctl --user stop titan-dense.service 2>/dev/null; sleep 2
  sudo -n pkill -f "mistralrs ser[v]e" 2>/dev/null; sleep 5
  export TITAN_MTP=0 TITAN_CUDA_GRAPHS=1 TITAN_RECURRENT_SLOTS=8
  echo "=== $label  (layers on GPU: $layers, ctx $ctx) ==="
  ( cd "$dir" && "$BIN" serve -p 1234 --no-ui --paged-attn off \
      --max-seq-len "$ctx" --format gguf -f "$name" -n "0:$layers" \
      >/tmp/quant-$label.log 2>&1 & )
  for i in $(seq 1 80); do sleep 3; ss -tln | grep -q ':1234' && break; done
  if ! ss -tln | grep -q ':1234'; then
    echo "  FAILED to start:"
    grep -viE "gitstatus|setopt|zshrc|GITSTATUS" /tmp/quant-$label.log | grep -iE "error|oom|memory|requires" | tail -3
    return
  fi
  echo "  VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  cd /tmp && python3 decode-once.py 16 >/dev/null 2>&1; python3 decode-once.py 256; cd /home/dingo
}

run "q4km-n60" /home/dingo/src/titan-engine/models-q4km/Qwen3.8-27B-DevSecOps-QLoRA-abliterated-Q4_K_M.gguf 60 8192
run "q4km-n62" /home/dingo/src/titan-engine/models-q4km/Qwen3.8-27B-DevSecOps-QLoRA-abliterated-Q4_K_M.gguf 62 8192
run "q4km-n64" /home/dingo/src/titan-engine/models-q4km/Qwen3.8-27B-DevSecOps-QLoRA-abliterated-Q4_K_M.gguf 64 8192
