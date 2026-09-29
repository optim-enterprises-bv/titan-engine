#!/usr/bin/env python3
"""First-token top-10 as [token text, score]. usage: toplogp.py PORT NAME llama|mistral
llama: score = log-prob (/completion, falling back to /v1/completions).
mistral: greedy requests report log10 of the raw logit, so score = the logit (10 ** value);
compare the two by gaps to the top-1 (a log-prob gap equals a logit gap)."""
import json, sys, urllib.request
port, name, kind = sys.argv[1], sys.argv[2], sys.argv[3]
text = open("prompts.txt").read()
prompts = [p for p in text.split("\n") if p and not p.startswith("    ")]
prompts = [p + '\n    """Return the nth Fibonacci number."""\n' if p.startswith("def fib") else p for p in prompts]
post = lambda path, body: json.load(urllib.request.urlopen(urllib.request.Request(
    f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
res = []
for p in prompts:
    if kind == "llama":
        try:
            r = post("/completion", {"prompt": p, "n_predict": 1, "temperature": 0, "n_probs": 10, "post_sampling_probs": False})
            top = [(t["token"], t["logprob"]) for t in r["completion_probabilities"][0]["top_logprobs"]]
        except Exception:
            r = post("/v1/completions", {"prompt": p, "max_tokens": 1, "temperature": 0, "logprobs": 10})
            lp = r["choices"][0]["logprobs"]
            c = lp.get("content") or []
            top = [(t["token"], t["logprob"]) for t in c[0]["top_logprobs"]]
    else:
        r = post("/v1/completions", {"model": "default", "prompt": p, "max_tokens": 1, "temperature": 0, "logprobs": 10})
        c = (r["choices"][0]["logprobs"] or {}).get("content") or []
        top = [(t.get("bytes") or str(t["token"]), 10 ** t["logprob"]) for t in c[0]["top_logprobs"]]
    res.append(top)
json.dump(res, open(f"out/{name}.top.json", "w"), ensure_ascii=False)
print(name, "top-10 done")
