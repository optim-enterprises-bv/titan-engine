#!/bin/bash
# M4 end-to-end: Qwen3.6-35B-A3B greedy raw completions, llama.cpp (--n-cpu-moe 18) vs mistral.rs tiered.
set -u; cd "$(dirname "$0")"; mkdir -p out
M=$HOME/ai/models; F=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
export LD_LIBRARY_PATH=$HOME/titan-engine/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
serve_wait() { for i in $(seq 1 300); do curl -sf -m 2 -o /dev/null localhost:$1${3:-/v1/models} && return 0; kill -0 $2 2>/dev/null || return 1; sleep 2; done; return 1; }
if [ "${SKIP_LLAMA:-0}" != 1 ]; then
  $HOME/ai/llama.cpp/build/bin/llama-server -m $M/$F -ngl 99 --n-cpu-moe 18 -c 4096 --port 18471 --temp 0 --top-k 1 -np 1 > out/llama.log 2>&1 &
  pid=$!; serve_wait 18471 $pid /health || { echo "llama died"; tail out/llama.log; exit 1; }
  python3 client.py 18471 llama 64; kill $pid; wait $pid 2>/dev/null
fi
for spec in "${@:-tiered TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=0.5}"; do
  set -- $spec; name=$1; shift
  env "$@" $HOME/titan-engine/mistral.rs/target/release/mistralrs --seed 0 serve -p 18472 --no-ui --paged-attn off --max-seq-len 4096 \
      --format gguf -m $M -f $F > out/$name.log 2>&1 &
  pid=$!; serve_wait 18472 $pid || { echo "$name died"; grep -v INFO out/$name.log | tail -20; exit 1; }
  python3 client.py 18472 $name 64; kill $pid; wait $pid 2>/dev/null
  python3 compare.py llama $name
done
