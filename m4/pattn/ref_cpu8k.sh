#!/bin/bash
# llama.cpp CPU reference (m4/g4s/ref_cpu.sh procedure: -ngl 0, GPU hidden, 6 threads, f16 KV, no LoRA) for the ~8k-token
# G3 prompts (gates.py g3 at 28000 chars), only while ~/titan-engine/.gpu.lock is free (YIELD_LOCK resumes later).
# usage: systemd-run --user --collect --wait -q -p MemoryMax=6G -p MemorySwapMax=0 bash ref_cpu8k.sh NAME GGUF
set -u
I=$HOME/titan-engine/top-pattn/m4/pattn; G=$HOME/titan-engine/m4/g4s; name=$1 gguf=$2; PORT=18702
flock -n $HOME/titan-engine/.gpu.lock true || { echo "gpu lock held: not starting"; exit 3; }
CUDA_VISIBLE_DEVICES= $HOME/ai/llama.cpp/build/bin/llama-server -m "$gguf" -ngl 0 -t 6 -c 12288 -np 1 \
  -ctk f16 -ctv f16 --temp 0 --port $PORT --host 127.0.0.1 --no-webui > $I/out/ref-$name.server.log 2>&1 &
pid=$!
trap 'kill -TERM $pid 2>/dev/null; sleep 3; kill -KILL $pid 2>/dev/null' EXIT
for i in $(seq 1 120); do curl -sf -m 2 -o /dev/null localhost:$PORT/health && break; sleep 2; done
cd $G && YIELD_LOCK=1 python3 gates.py g3 $PORT llama $I/out/ref-$name-g3l.json 28000
