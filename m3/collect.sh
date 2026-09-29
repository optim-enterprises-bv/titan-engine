#!/bin/bash
# M3: routing traces (TITAN_TIERED_TRACE) + M2 throughput of the AVX2 CPU twin.
# usage: collect.sh NAME PROMPTFILE MAX_TOKENS ENV...
set -u
cd "$(dirname "$0")"
BIN=${BIN:-$HOME/titan-engine/mistral.rs/target/release/mistralrs}
export LD_LIBRARY_PATH=$HOME/titan-engine/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
DIR=${DIR:-$HOME/titan-engine/m1/data}
FILE=${FILE:-qwen3coder30b-first24.gguf}
PORT=18462
name=$1 prompts=$2 max_tokens=$3; shift 3
mkdir -p out
rm -f out/$name.trace
env "$@" "$BIN" --seed 0 serve -p $PORT --no-ui --paged-attn off --max-seq-len 4096 \
    --format gguf -m "$DIR" -f "$FILE" > out/$name.server.log 2>&1 &
pid=$!
for i in $(seq 1 240); do
    curl -s -m 2 -o /dev/null localhost:$PORT/v1/models && break
    kill -0 $pid 2>/dev/null || { echo "server died"; tail -20 out/$name.server.log; exit 1; }
    sleep 2
done
python3 - "$name" "$PORT" "$max_tokens" "$prompts" <<'PY'
import json, sys, time, urllib.request
name, port, max_tokens, pf = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
prompts = [l.strip() for l in open(pf) if l.strip()]
res, t0, toks, dec = [], time.time(), 0, []
for p in prompts:
    body = json.dumps({"model": "default", "messages": [{"role": "user", "content": p}],
                       "temperature": 0.0, "max_tokens": max_tokens, "seed": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    res.append(r["choices"][0]["message"]["content"])
    n = r.get("usage", {}).get("completion_tokens", 0)
    toks += n; dec.append(n / (time.time() - t))
dt = time.time() - t0
json.dump(res, open(f"out/{name}.json", "w"), ensure_ascii=False, indent=1)
print(f"{name}: {len(res)} completions, {toks} tokens, {dt:.1f}s, {toks/dt:.1f} tok/s (median per-request {sorted(dec)[len(dec)//2]:.1f})")
PY
rc=$?
grep -m2 'titan tiered experts' out/$name.server.log
kill $pid; wait $pid 2>/dev/null
exit $rc
