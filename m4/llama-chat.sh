#!/bin/bash
# llama.cpp baseline on the same 40 eval prompts, same client as collect.sh (chat, greedy, 256 tokens)
E=$HOME/titan-engine; M=$HOME/ai/models; F=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
$HOME/ai/llama.cpp/build/bin/llama-server -m $M/$F -ngl 99 --n-cpu-moe 18 -c 4096 --port 18462 -np 1 --jinja > $E/m4/out/llama-chat.server.log 2>&1 & pid=$!
for i in $(seq 1 300); do curl -sf -m 2 -o /dev/null localhost:18462/health && break; sleep 2; done
cd $E/m3 && python3 - llama-chat 18462 256 $E/m4/prompts-eval.txt <<'PY'
import json, sys, time, urllib.request
name, port, max_tokens, pf = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
prompts = [l.strip() for l in open(pf) if l.strip()]
res, t0, toks, dec = [], time.time(), 0, []
for p in prompts:
    body = json.dumps({"model": "default", "messages": [{"role": "user", "content": p}],
                       "temperature": 0.0, "max_tokens": max_tokens, "seed": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t = time.time(); r = json.load(urllib.request.urlopen(req, timeout=1800))
    res.append(r["choices"][0]["message"]["content"] or ""); n = r.get("usage", {}).get("completion_tokens", 0)
    toks += n; dec.append(n / (time.time() - t))
dt = time.time() - t0
print(f"{name}: {len(res)} completions, {toks} tokens, {dt:.1f}s, {toks/dt:.1f} tok/s (median per-request {sorted(dec)[len(dec)//2]:.1f})")
PY
kill $pid; wait $pid 2>/dev/null
