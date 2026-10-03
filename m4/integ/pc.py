#!/usr/bin/env python3
"""pc.py PORT NAME: repeated-prefix workload for the prefix-cache question. One ~6k-token shared system prefix
(m4/b2merge/bigtest.py's chunk), 8 different user questions, chat completions, greedy 64 tokens, streamed.
Per request: prompt tokens, cached tokens (usage), time to first token, decode tok/s. Writes out/NAME.pc.json."""
import json, sys, time, urllib.request
port, name = sys.argv[1], sys.argv[2]
CHUNK = "The titan engine tiers mixture-of-experts weights between the GPU and the CPU. "
SYS = CHUNK * 400
QS = ["Summarise the system text in one line.", "What does the text say about the GPU?", "Count the sentences, roughly.",
      "Name one word that repeats.", "Is the text about cooking?", "Give the text a title.",
      "What is tiered here?", "Answer in French: what is the text about?"]
res = []
for q in QS:
    body = {"model": "default", "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": q}],
            "max_tokens": 64, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); first = None; last = None; n = 0; usage = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices") or []:
                if (c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning_content"):
                    now = time.time(); first = first or now; last = now; n += 1
    ttft = (first or time.time()) - t0
    dec = (n - 1) / (last - first) if n > 1 and last > first else 0.0
    cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens"))
    res.append({"q": q, "prompt_tokens": usage.get("prompt_tokens"), "cached": cached, "ttft": ttft, "decode_tps": dec, "chunks": n})
    print(f"  {name}: prompt {usage.get('prompt_tokens')} cached {cached} ttft {ttft:.2f}s decode {dec:.1f} tok/s", flush=True)
json.dump(res, open(f"out/{name}.pc.json", "w"), indent=1)
t = sorted(r["ttft"] for r in res[1:]); d = sorted(r["decode_tps"] for r in res)
print(f"PC {name}: first ttft {res[0]['ttft']:.2f}s, repeats (7) ttft median {t[len(t)//2]:.2f}s sum {sum(t):.1f}s, decode median {d[len(d)//2]:.1f} tok/s")
