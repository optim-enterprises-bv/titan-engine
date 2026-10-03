#!/usr/bin/env python3
"""req1.py PORT PROMPT_FILE: one titan /v1/completions request (max_tokens 1, top-20), prints the top-5."""
import json, sys, urllib.request
p = open(sys.argv[2]).read()
body = {"model": "default", "prompt": p, "max_tokens": 1, "temperature": 0, "seed": 0, "logprobs": 20}
r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{sys.argv[1]}/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
print(r["usage"], [(x["token"], round(x["logprob"], 4)) for x in r["choices"][0]["logprobs"]["content"][0]["top_logprobs"][:5]])
