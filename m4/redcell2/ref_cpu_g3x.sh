#!/bin/bash
# llama.cpp CPU reference for g3x (long prompts 4..15 at 13500 chars), as m4/g4s/ref_cpu.sh: -ngl 0, 6 threads, only while
# the GPU lock is free (YIELD_LOCK: stops with exit 3, resumes from out/ref-red-g3x.json.partial).
set -u
I=$HOME/titan-engine/m4/redcell2; PORT=18702
flock -n $HOME/titan-engine/.gpu.lock true || { echo "gpu lock held: not starting"; exit 3; }
CUDA_VISIBLE_DEVICES= $HOME/ai/llama.cpp/build/bin/llama-server -m $HOME/ai/models/redcell-26b/REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf \
  -ngl 0 -t 6 -c 8192 -b 512 -ub 512 -np 1 -ctk f16 -ctv f16 --temp 0 --port $PORT --host 127.0.0.1 --no-webui > $I/out/ref-red-g3x.server.log 2>&1 &
pid=$!
trap 'kill -TERM $pid 2>/dev/null; sleep 3; kill -KILL $pid 2>/dev/null' EXIT
for i in $(seq 1 300); do curl -sf -m 2 -o /dev/null localhost:$PORT/health && break; sleep 2; done
YIELD_LOCK=1 python3 $I/gates2.py g3x $PORT llama $I/out/ref-red-g3x.json 13500 4 12
