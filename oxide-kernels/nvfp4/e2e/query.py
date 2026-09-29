#!/usr/bin/env python3
"""query.py <base_url> <out.json> [model]: greedy 64-token raw completions (+ first-token top-4) and tok/s."""
import json, sys, time, urllib.request

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Once upon a time, in a small village by the sea,",
    "The three laws of thermodynamics are",
    "1, 1, 2, 3, 5, 8, 13,",
    "Translate to German: The weather is nice today.\nGerman:",
]

def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())

base, out = sys.argv[1], sys.argv[2]
model = sys.argv[3] if len(sys.argv) > 3 else "default"
res = []
for p in PROMPTS:
    body = {"model": model, "prompt": p, "max_tokens": 64, "temperature": 0.0, "top_k": 1, "logprobs": 4, "stream": False}
    t0 = time.time()
    r = post(base + "/v1/completions", body)
    dt = time.time() - t0
    ch = r["choices"][0]
    lp = ch.get("logprobs") or {}
    top = None
    if isinstance(lp, dict):
        if lp.get("top_logprobs"):
            top = lp["top_logprobs"][0]
        elif lp.get("content"):
            top = {t["token"]: t["logprob"] for t in lp["content"][0].get("top_logprobs", [])}
    usage = r.get("usage", {})
    res.append({"prompt": p, "text": ch["text"], "top4": top, "secs": dt, "usage": usage, "timings": r.get("timings")})
    print(json.dumps({"prompt": p[:30], "text": ch["text"][:80], "top4": top, "secs": round(dt, 3), "usage": usage}))
json.dump(res, open(out, "w"), indent=1)
