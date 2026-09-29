#!/usr/bin/env python3
"""First-token top-10 log-probs. usage: toplogp.py PORT NAME llama|mistral"""
import json, sys, urllib.request
port, name, kind = sys.argv[1], sys.argv[2], sys.argv[3]
text = open("prompts.txt").read()
prompts = [p for p in text.split("\n") if p and not p.startswith("    ")]
prompts = [p + '\n    """Return the nth Fibonacci number."""\n' if p.startswith("def fib") else p for p in prompts]
res = []
for p in prompts:
    if kind == "llama":
        body = {"prompt": p, "n_predict": 1, "temperature": 0, "n_probs": 10, "post_sampling_probs": False}
        r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
        top = [(t["token"], t["logprob"]) for t in r["completion_probabilities"][0]["top_logprobs"]]
    else:
        body = {"model": "default", "prompt": p, "max_tokens": 1, "temperature": 0, "logprobs": 10}
        r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
        lp = r["choices"][0]["logprobs"]
        c = (lp or {}).get("content") or []
        top = [(t["token"], t["logprob"]) for t in c[0]["top_logprobs"]] if c else [("?", 0)]
    res.append(top)
json.dump(res, open(f"out/{name}.top.json", "w"), ensure_ascii=False)
print(name, "done")
