#!/usr/bin/env python3
"""dec.py PORT MODEL OUT.json [N=2] [MAX=128]: decode speed through a (swap) server: N streamed chat requests, greedy,
MAX tokens; decode tok/s = (chunks - 1) / (last - first chunk time), plus TTFT."""
import json, sys, time, urllib.request
port, model, out = sys.argv[1:4]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 2
mx = int(sys.argv[5]) if len(sys.argv) > 5 else 128
Q = ["Write a detailed explanation of how a hash map works, with an example in Python.",
     "Describe the water cycle step by step for a high-school student."]
res = []
for i in range(n):
    body = {"model": model, "messages": [{"role": "user", "content": Q[i % len(Q)]}], "max_tokens": mx, "temperature": 0,
            "stream": True}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); first = last = None; k = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            for c in d.get("choices") or []:
                dl = c.get("delta") or {}
                if dl.get("content") or dl.get("reasoning_content"):
                    now = time.time(); first = first or now; last = now; k += 1
    dec = (k - 1) / (last - first) if k > 1 and last > first else 0.0
    res.append({"chunks": k, "ttft": (first or time.time()) - t0, "decode_tps": dec})
json.dump(res, open(out, "w"))
d = sorted(r["decode_tps"] for r in res)
print(f"DEC {model}: decode {[round(r['decode_tps'], 1) for r in res]} tok/s (median {d[len(d)//2]:.1f}), chunks {[r['chunks'] for r in res]}, ttft {[round(r['ttft'], 2) for r in res]}", flush=True)
