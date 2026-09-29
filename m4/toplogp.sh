#!/bin/bash
cd "$(dirname "$0")"; M=$HOME/ai/models; F=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
export LD_LIBRARY_PATH=$HOME/titan-engine/lib:/usr/local/cuda/lib64
sw() { for i in $(seq 1 300); do curl -sf -m 2 -o /dev/null localhost:$1$3 && return 0; kill -0 $2 2>/dev/null || return 1; sleep 2; done; return 1; }
[ -f out/llama.top.json ] || { $HOME/ai/llama.cpp/build/bin/llama-server -m $M/$F -ngl 99 --n-cpu-moe 18 -c 4096 --port 18471 -np 1 > out/llama-lp.log 2>&1 & p=$!
sw 18471 $p /health && python3 toplogp.py 18471 llama llama; kill $p; wait $p 2>/dev/null; }
TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=0.55 $HOME/titan-engine/mistral.rs/target/release/mistralrs --seed 0 serve -p 18472 --no-ui --paged-attn off --max-seq-len 4096 --format gguf -m $M -f $F > out/mistral-lp.log 2>&1 & p=$!
sw 18472 $p /v1/models && python3 toplogp.py 18472 mistral mistral; kill $p; wait $p 2>/dev/null
python3 - <<'PY'
import json
a, b = json.load(open("out/llama.top.json")), json.load(open("out/mistral.top.json"))
for i, (x, y) in enumerate(zip(a, b)):
    print(i, "llama:", [(t, round(l, 2)) for t, l in x[:4]])
    print(i, "mistr:", [(t, round(l, 2)) for t, l in y[:4]])
PY
