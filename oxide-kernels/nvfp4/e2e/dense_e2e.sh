#!/bin/bash
# Dense model: llama.cpp vs mistral.rs, greedy 64 tokens on m4/prompts.txt + first-token top-4.
# usage: dense_e2e.sh <gguf dir> <gguf file> <tag> [MISTRALRS_BIN]   (SKIP_LLAMA=1 reuses out/<tag>_llama*.json)
set -u; cd "$(dirname "$0")"; mkdir -p out
M=$1; F=$2; T=$3
BIN=${4:-$HOME/titan-engine/target-fmt/release/mistralrs}
M4=$HOME/titan-engine/m4
export LD_LIBRARY_PATH=$HOME/titan-engine/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
sw() { for i in $(seq 1 300); do curl -sf -m 2 -o /dev/null localhost:$1$3 && return 0; kill -0 $2 2>/dev/null || return 1; sleep 2; done; return 1; }
[ -f out/${T}_vocab.json ] || python3 vocab.py $M/$F out/${T}_vocab.json
cp $M4/prompts.txt . 2>/dev/null
if [ "${SKIP_LLAMA:-0}" != 1 ]; then
  systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 \
    $HOME/ai/llama.cpp/build/bin/llama-server -m $M/$F -ngl 99 -c 4096 --port 18491 --temp 0 --top-k 1 -np 1 > out/${T}_llama.log 2>&1 &
  p=$!; sw 18491 $p /health || { echo "llama died"; tail out/${T}_llama.log; exit 1; }
  python3 $M4/client.py 18491 ${T}_llama 64 && python3 $M4/toplogp.py 18491 ${T}_llama llama
  kill $p; wait $p 2>/dev/null
fi
systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 \
  $BIN --seed 0 serve -p 18492 --no-ui --paged-attn off --max-seq-len 4096 --format gguf -m $M -f $F > out/${T}_mistral.log 2>&1 &
p=$!; sw 18492 $p /v1/models || { echo "mistral died"; grep -v INFO out/${T}_mistral.log | tail -20; exit 1; }
python3 $M4/client.py 18492 ${T}_mistral 64 && python3 $M4/toplogp.py 18492 ${T}_mistral mistral
kill $p; wait $p 2>/dev/null
python3 cmp_strip.py ${T}_llama ${T}_mistral
T=$T python3 - <<'PY'
import json, os
T = os.environ["T"]
vocab = json.load(open(f"out/{T}_vocab.json"))
def dec(t):
    try: t = int(t)
    except (TypeError, ValueError): return t
    return vocab[t].replace("Ġ", " ").replace("Ċ", "\n")
a, b = json.load(open(f"out/{T}_llama.top.json")), json.load(open(f"out/{T}_mistral.top.json"))
agree = sets = 0
for i, (x, y) in enumerate(zip(a, b)):
    y = sorted(y, key=lambda t: -t[1])  # mistral.rs: ids with log10(logit): rank them
    lt = [t for t, _ in x[:4]]; mt = [dec(t) for t, _ in y[:4]]
    agree += lt[0] == mt[0]; sets += set(lt) == set(mt)
    print(i, "llama:", [(t, round(l, 2)) for t, l in x[:4]])
    print(i, "mistr:", [(dec(t), round(l, 3)) for t, l in y[:4]], "top-4 set match:", set(lt) == set(mt))
print(f"first-token argmax agreement {agree}/{len(a)}, top-4 set agreement {sets}/{len(a)}")
PY
