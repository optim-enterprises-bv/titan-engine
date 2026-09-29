#!/bin/bash
# e2e.sh <name> <mistral-bin|llama> <iq4_nl|mxfp4|nvfp4> [env...]: dense Qwen3-14B-<fmt> server under a
# memory cap; greedy 64-token raw completions over prompts.txt (out/<name>.json), then prefill2k.py
# (~2k-token prompt, median of 5) and bigprompt.py (~2k-token chat). Run inside a GPU window.
set -u; name=$1 bin=$2 fmt=$3; shift 3
E=$HOME/titan-engine; D=$E/m4/fmtmmq
M=$E/m1/data/dense; F=Qwen3-14B-$fmt.gguf
export LD_LIBRARY_PATH=$E/lib:/usr/local/cuda/lib64
port=18497
mkdir -p $D/out
if [ $bin = llama ]; then
  cmd="$HOME/ai/llama.cpp/build/bin/llama-server -m $M/$F -ngl 99 -c 4096 --port $port --temp 0 --top-k 1 -np 1"; probe=/health
else
  cmd="$bin --seed 0 serve -p $port --no-ui --paged-attn off --max-seq-len 4096 --format gguf -m $M -f $F"; probe=/v1/models
fi
# the previous server's VRAM must be gone
for i in $(seq 1 30); do u=$(timeout 10 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1); [ -n "$u" ] && [ "$u" -lt 1500 ] && break; sleep 2; done
echo "[$name] GPU used before start: ${u:-?} MiB"
t0=$(date +%s)
systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 env "$@" $cmd > $D/out/$name.server.log 2>&1 &
pid=$!
stop() { kill $pid 2>/dev/null; for i in $(seq 1 30); do kill -0 $pid 2>/dev/null || break; sleep 1; done; kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null; }
ok=0; for i in $(seq 1 240); do curl -sf -m 2 -o /dev/null localhost:$port$probe && { ok=1; break; }; kill -0 $pid 2>/dev/null || break; sleep 1; done
[ $ok = 1 ] || { echo "[$name] server failed"; grep -v " INFO " $D/out/$name.server.log | tail -20; stop; exit 1; }
echo "[$name] up in $(( $(date +%s) - t0 ))s"
cd $D
# warm-up (first-use PTX JIT, allocator): not timed; the first prompt is > 8 tokens so it takes the MMQ path
curl -s -m 600 -o /dev/null localhost:$port/v1/completions -H 'Content-Type: application/json' -d '{"model":"default","prompt":"Hello, my name is Hello, my name is Hello, my name is","max_tokens":8,"temperature":0}'
curl -s -m 600 -o /dev/null localhost:$port/v1/completions -H 'Content-Type: application/json' -d '{"model":"default","prompt":"Hi","max_tokens":4,"temperature":0}'
rc=0
timeout 300 python3 client.py $port $name 64 || rc=1
timeout 300 python3 prefill2k.py $port $name 2048 || rc=1
timeout 120 python3 $E/m4/bigprompt.py $port 2048 | sed "s/^/[$name] bigprompt /" || rc=1
stop
grep -m3 -iE "panic|CUDA_ERROR|error:" $D/out/$name.server.log | sed "s/^/[$name] server: /"
exit $rc
