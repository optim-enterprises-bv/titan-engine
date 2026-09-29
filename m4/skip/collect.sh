#!/bin/bash
# One 80B server per variant: 40 held-out prompts x 64 (speed + texts for first divergence), then quality.py.
# usage: collect.sh NAME ENV...   (BIN, F80 from the environment)
set -u
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=$HOME/titan-engine/lib:${LD_LIBRARY_PATH:-}
PORT=18473
name=$1; shift
mkdir -p out
env "$@" "$BIN" --seed 0 serve -p $PORT --no-ui --paged-attn off --max-seq-len 4096 \
    --format gguf -m "$(dirname $F80)" -f "$(basename $F80)" > out/$name.server.log 2>&1 &
pid=$!
for i in $(seq 1 240); do
    curl -s -m 2 -o /dev/null localhost:$PORT/v1/models && break
    kill -0 $pid 2>/dev/null || { echo "server died"; tail -20 out/$name.server.log; exit 1; }
    sleep 2
done
python3 - "$name" "$PORT" 64 $HOME/titan-engine/m4/prompts-eval.txt <<'PY'
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
[ "${QUALITY:-1}" = 1 ] && { timeout ${QTIME:-900} python3 quality.py $PORT $name || rc=1; }
grep -o 'titan miss-skip: .*\|titan prune: .*' out/$name.server.log | tail -2
grep -o 'decode hit rate.*' out/$name.server.log | tail -1
grep -o 'titan tiered timing.*' out/$name.server.log | tail -1
kill -TERM $pid; for i in 1 2 3 4 5; do kill -0 $pid 2>/dev/null || break; sleep 1; done; kill -KILL $pid 2>/dev/null
wait $pid 2>/dev/null
exit $rc
