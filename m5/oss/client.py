#!/usr/bin/env python3
"""Greedy raw completions against an OpenAI-compatible /v1/completions server.
usage: client.py PORT NAME [MAX_TOKENS]  (prompts from prompts.txt, one per blank-line-free block)"""
import json, sys, time, urllib.request
port, name = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 64
text = open("prompts.txt").read()
prompts = [p for p in text.split("\n") if p and not p.startswith("    ")]
# the fibonacci prompt spans two lines
prompts = [p + '\n    """Return the nth Fibonacci number."""\n' if p.startswith("def fib") else p for p in prompts]
out, usage, toks, t0 = [], [], 0, time.time()
for p in prompts:
    body = {"model": "default", "prompt": p, "max_tokens": n, "temperature": 0.0, "top_k": 1, "seed": 0}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    u = r.get("usage", {}); out.append(r["choices"][0]["text"]); toks += u.get("completion_tokens", 0)
    usage.append([u.get("prompt_tokens"), u.get("completion_tokens")])
dt = time.time() - t0
json.dump(out, open(f"out/{name}.json", "w"), ensure_ascii=False, indent=1)
json.dump(usage, open(f"out/{name}.usage.json", "w"))
print(f"{name}: {len(out)} completions, {toks} tokens, {dt:.1f}s, {toks/dt:.1f} tok/s")
