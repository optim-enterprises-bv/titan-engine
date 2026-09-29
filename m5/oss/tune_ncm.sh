#!/bin/bash
# Smallest llama.cpp --n-cpu-moe (most expert layers on the GPU) that starts and generates on GGUF.
# usage: tune_ncm.sh GGUF START  -> prints and writes out/ncm.<basename>
set -u; cd "$(dirname "$0")"
G=$1; START=$2; LL=$HOME/ai/llama.cpp/build/bin/llama-server; LDP=$HOME/titan-engine/lib:/usr/local/cuda/lib64
probe() {
  local n=$1 u=oss-probe-$$-$1 ok=1
  systemd-run --user --unit=$u --collect -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=600 \
    --setenv=LD_LIBRARY_PATH=$LDP -p StandardOutput=truncate:$PWD/out/probe-$n.log -p StandardError=truncate:$PWD/out/probe-$n.log \
    $LL -m $G -ngl 99 --n-cpu-moe $n -c 4096 --port 18493 --temp 0 --top-k 1 -np 1
  for i in $(seq 1 200); do
    curl -sf -m 2 -o /dev/null localhost:18493/health && { ok=0; break; }
    systemctl --user -q is-active $u || break; sleep 2
  done
  if [ $ok = 0 ]; then
    curl -sf -m 300 localhost:18493/completion -d '{"prompt":"The capital of France is","n_predict":4}' -o /dev/null || ok=1
  fi
  echo "ncm $n: $([ $ok = 0 ] && echo fits || echo fails) ($(grep -o 'CUDA0 model buffer size = [0-9.]* MiB' out/probe-$n.log | head -1))"
  systemctl --user stop $u 2>/dev/null; sleep 2
  return $ok
}
best=
if probe $START; then
  best=$START; n=$((START - 1))
  while [ $n -ge 0 ] && probe $n; do best=$n; n=$((n - 1)); done
else
  n=$((START + 1))
  while [ $n -le 64 ]; do probe $n && { best=$n; break; }; n=$((n + 1)); done
fi
echo "best --n-cpu-moe: $best"; echo $best > out/ncm.$(basename $G)
