#!/bin/bash
# Upstream mistral.rs 0.9.5 (.local/bin/mistralrs) on both quant files, same ctx/prompt as the
# titan runs, so the two engines are compared on identical inputs.
set -u
BIN=/home/dingo/.local/bin/mistralrs
run() { # label dir file layers ctx
  local label=$1 dir=$2 name=$3 layers=$4 ctx=$5
  systemctl --user stop titan-dense.service 2>/dev/null; sleep 2
  sudo -n pkill -f "mistralrs ser[v]e" 2>/dev/null; sleep 5
  echo "=== upstream $label (layers $layers, ctx $ctx) ==="
  ( cd "$dir" && "$BIN" serve -p 1234 --no-ui --max-model-len "$ctx" --format gguf \
      -f "$name" -n "0:$layers" >/tmp/up-$label.log 2>&1 & )
  for i in $(seq 1 70); do sleep 3; ss -tln | grep -q ':1234' && break; done
  if ! ss -tln | grep -q ':1234'; then
    echo "  DID NOT START. Last lines:"
    grep -viE "gitstatus|setopt|zshrc|GITSTATUS|exec zsh" /tmp/up-$label.log | tail -4
    return
  fi
  echo "  VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  cd /tmp && python3 decode-once.py 16 >/dev/null 2>&1; python3 decode-once.py 256; cd /home/dingo
  sudo -n pkill -f "mistralrs ser[v]e" 2>/dev/null; sleep 4
}

run "q3km-n60" /home/dingo/src/titan-engine/models-optim        optim-Q3_K_M.gguf 60 4096
run "q3km-n64" /home/dingo/src/titan-engine/models-optim        optim-Q3_K_M.gguf 64 4096
run "q4km-n52" /home/dingo/src/titan-engine/models-q4km Qwen3.8-27B-DevSecOps-QLoRA-abliterated-Q4_K_M.gguf 52 4096
run "q4km-n56" /home/dingo/src/titan-engine/models-q4km Qwen3.8-27B-DevSecOps-QLoRA-abliterated-Q4_K_M.gguf 56 4096
