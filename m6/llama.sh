#!/bin/bash
# llama.cpp on the MTP GGUF, same 40 prompts / client as collect.sh. usage: llama.sh NAME [llama-server args...]
E=$HOME/titan-engine; M=$HOME/ai/models/Qwen3.6-35B-A3B-MTP; F=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf; n=$1; shift
PORT=18472
# shared GPU: wait (up to 90 min) until no other process holds VRAM
for i in $(seq 1 540); do [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) -lt 700 ] && break; sleep 10; done
# quiet CPU: wait (up to 30 min) for the 1-minute load average to drop below 2
for i in $(seq 1 180); do awk '{exit !($1 < 2)}' /proc/loadavg && break; sleep 10; done
echo "gpu used before: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"; echo "loadavg before $n: $(cat /proc/loadavg)"
systemd-run --user --unit=mtp-llama-$n --collect -q -p MemoryMax=14G -p MemorySwapMax=0 \
  -p StandardOutput=truncate:$E/m6/out/$n.server.log -p StandardError=truncate:$E/m6/out/$n.server.log \
  $HOME/ai/llama.cpp/build/bin/llama-server -m $M/$F -ngl 99 --n-cpu-moe ${NCPUMOE:-18} -c 4096 --port $PORT -np 1 --jinja "$@"
for i in $(seq 1 300); do curl -sf -m 2 -o /dev/null localhost:$PORT/health && { echo "gpu used loaded: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"; break; }; systemctl --user -q is-active mtp-llama-$n || { echo died; tail -5 $E/m6/out/$n.server.log; exit 1; }; sleep 2; done
python3 - $n $PORT 256 ${PROMPTS:-$E/m4/prompts-eval.txt} $E/m6/out <<'PY'
import json, sys, time, urllib.request
name, port, max_tokens, pf, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
prompts = [l.strip() for l in open(pf) if l.strip()]
res, t0, toks, dec, dn, da = [], time.time(), 0, [], 0, 0
for p in prompts:
    body = json.dumps({"model": "default", "messages": [{"role": "user", "content": p}],
                       "temperature": 0.0, "max_tokens": max_tokens, "seed": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t = time.time(); r = json.load(urllib.request.urlopen(req, timeout=1800))
    m = r["choices"][0]["message"]; res.append((m.get("reasoning_content") or "") + "|" + (m.get("content") or ""))
    n = r.get("usage", {}).get("completion_tokens", 0); tm = r.get("timings", {})
    dn += tm.get("draft_n", 0); da += tm.get("draft_n_accepted", 0)
    toks += n; dec.append(n / (time.time() - t))
dt = time.time() - t0
json.dump(res, open(f"{out}/{name}.json", "w"), ensure_ascii=False, indent=1)
acc = f", draft {dn} accepted {da} ({100*da/dn:.1f}%)" if dn else ""
print(f"{name}: {len(res)} completions, {toks} tokens, {dt:.1f}s, {toks/dt:.1f} tok/s (median per-request {sorted(dec)[len(dec)//2]:.1f}){acc}")
PY
systemctl --user stop mtp-llama-$n
echo "loadavg after $n: $(cat /proc/loadavg)"
