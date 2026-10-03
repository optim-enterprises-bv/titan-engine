#!/usr/bin/env python3
"""one.py PORT PROMPTS.json IDX N -- one greedy completion of prompt IDX (N tokens); prints the text (repr)."""
import json, sys, urllib.request
port, prompts, idx, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
p = json.load(open(prompts))[idx]
body = {"model": "default", "prompt": p, "max_tokens": n, "temperature": 0, "seed": 0}
r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(body).encode(),
                                                            {"Content-Type": "application/json"}), timeout=600))
print("one:", idx, repr(r["choices"][0]["text"]), r.get("usage", {}).get("prompt_tokens"), flush=True)
