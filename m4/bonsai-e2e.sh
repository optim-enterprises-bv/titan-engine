#!/bin/bash
# Dense qwen35 (Bonsai-27B Q1_0): llama.cpp vs mistral.rs, greedy 64 tokens + first-token top-4.
# usage: bonsai-e2e.sh [MISTRALRS_BIN]   (SKIP_LLAMA=1 reuses out/bonsai_llama*.json)
set -u; cd "$(dirname "$0")"; mkdir -p out
M=$HOME/ai/models/bonsai/Bonsai-27B-gguf; F=Bonsai-27B-Q1_0.gguf
BIN=${1:-$HOME/titan-engine/target-q35d/release/mistralrs}
export LD_LIBRARY_PATH=$HOME/titan-engine/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
sw() { for i in $(seq 1 300); do curl -sf -m 2 -o /dev/null localhost:$1$3 && return 0; kill -0 $2 2>/dev/null || return 1; sleep 2; done; return 1; }
if [ "${SKIP_LLAMA:-0}" != 1 ]; then
  systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 \
    $HOME/ai/llama.cpp/build/bin/llama-server -m $M/$F -ngl 99 -c 4096 --port 18481 --temp 0 --top-k 1 -np 1 > out/bonsai_llama.log 2>&1 &
  p=$!; sw 18481 $p /health || { echo "llama died"; tail out/bonsai_llama.log; exit 1; }
  python3 client.py 18481 bonsai_llama 64 && python3 toplogp.py 18481 bonsai_llama llama
  kill $p; wait $p 2>/dev/null
fi
systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 \
  $BIN --seed 0 serve -p 18482 --no-ui --paged-attn off --max-seq-len 4096 --format gguf -m $M -f $F > out/bonsai_mistral.log 2>&1 &
p=$!; sw 18482 $p /v1/models || { echo "mistral died"; grep -v INFO out/bonsai_mistral.log | tail -20; exit 1; }
python3 client.py 18482 bonsai_mistral 64 && python3 toplogp.py 18482 bonsai_mistral mistral
kill $p; wait $p 2>/dev/null
python3 compare.py bonsai_llama bonsai_mistral
python3 - <<'PY'
import json, math
vocab = json.load(open("out/bonsai_vocab.json"))
dec = lambda t: vocab[t].replace("Ġ", " ").replace("Ċ", "\n") if isinstance(t, int) else t
a, b = json.load(open("out/bonsai_llama.top.json")), json.load(open("out/bonsai_mistral.top.json"))
agree = 0
for i, (x, y) in enumerate(zip(a, b)):
    y = sorted(y, key=lambda t: -t[1])
    lt = [t for t, _ in x[:4]]; mt = [dec(t) for t, _ in y[:4]]
    agree += lt[0] == mt[0]
    print(i, "llama:", [(t, round(l, 2)) for t, l in x[:4]])
    print(i, "mistr:", [(dec(t), round(l, 3)) for t, l in y[:4]], "top-4 set match:", set(lt) == set(mt))
print(f"first-token argmax agreement {agree}/{len(a)}")
PY
