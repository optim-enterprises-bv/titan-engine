#!/bin/bash
# First real MXFP4 model (unsloth Qwen3.6-35B-A3B-MXFP4_MOE): llama.cpp vs mistral.rs (nvcc-free, tiered auto).
E=$HOME/titan-engine; cd $E/m4; mkdir -p out
M=$HOME/ai/models/Qwen3.6-35B-A3B-MTP; F=Qwen3.6-35B-A3B-MXFP4_MOE.gguf; BIN=${BIN:-$E/target-oxide/release/mistralrs}
export LD_LIBRARY_PATH=$E/lib:/usr/local/cuda/lib64
sw() { for i in $(seq 1 300); do curl -sf -m 2 -o /dev/null localhost:$1$3 && return 0; kill -0 $2 2>/dev/null || return 1; sleep 2; done; return 1; }
# llama.cpp: raw greedy + top-4
$HOME/ai/llama.cpp/build/bin/llama-server -m $M/$F -ngl 99 --n-cpu-moe 18 -c 4096 --port 18471 --temp 0 --top-k 1 -np 1 > out/mx-llama.log 2>&1 & p=$!
sw 18471 $p /health || { echo "llama died"; tail out/mx-llama.log; }
python3 client.py 18471 mx-llama 64; python3 toplogp.py 18471 mx-llama llama; kill $p; wait $p 2>/dev/null
# mistral.rs: raw greedy + top-4
TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto $BIN --seed 0 serve -p 18472 --no-ui --paged-attn off --max-seq-len 4096 --format gguf -m $M -f $F > out/mx-mistral.log 2>&1 & p=$!
sw 18472 $p /v1/models || { echo "mistral died"; grep -v " INFO " out/mx-mistral.log | tail; }
python3 client.py 18472 mx-mistral 64; python3 toplogp.py 18472 mx-mistral mistral; kill $p; wait $p 2>/dev/null
grep -o "titan tiered auto.*" out/mx-mistral.log | head -1
python3 - <<'PY'
import json
a,b=json.load(open("out/mx-llama.json")),json.load(open("out/mx-mistral.json"))
same=0
for i,(x,y) in enumerate(zip(a,b)):
    x,y=x.lstrip(),y.lstrip(); k=next((j for j in range(min(len(x),len(y))) if x[j]!=y[j]),min(len(x),len(y)))
    same+=x==y; print(i,"identical" if x==y else f"same for {k}/{len(x)} chars")
print(f"greedy identical (leading ws ignored): {same}/{len(a)}")
v=json.load(open("out/vocab.json"))
la,mb=json.load(open("out/mx-llama.top.json")),json.load(open("out/mx-mistral.top.json"))
tok=lambda t: v[int(t)].replace('Ġ',' ').replace('Ċ','\\n')
arg=sum(x[0][0]==tok(y[0][0]) for x,y in zip(la,mb)); top4=sum(set(t for t,_ in x[:4])==set(tok(t) for t,_ in y[:4]) for x,y in zip(la,mb))
print(f"first-token argmax {arg}/{len(la)}, top-4 set {top4}/{len(la)}")
PY
# speed: 40 chat prompts x 256
cd $E/m3 && BIN=$BIN DIR=$M FILE=$F ./collect.sh mx-q35 $E/m4/prompts-eval.txt 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto 2>&1 | grep -v INFO | grep tok/s
grep -o "decode hit rate.*" out/mx-q35.server.log | tail -1
