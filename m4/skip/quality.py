#!/usr/bin/env python3
"""Quality gate for the 80B (Qwen chat template via /v1/chat/completions), greedy:
the fixed 20-question practice set (m5/oss/questions.json) and the 100-question set (q100.json), each question
+ " Answer with just the answer.", scored by answer regex (case-insensitive) on the reply.
usage: quality.py PORT NAME [MAX_TOKENS]  -> out/NAME.quality.json"""
import json, re, sys, time, urllib.request
port, name = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 48
sets = {"practice": json.load(open("~/titan-engine/m5/oss/questions.json")), "q100": json.load(open("q100.json"))}
out, toks, t0, line = {}, 0, time.time(), []
for sname, qs in sets.items():
    res = []
    for q, pat in qs:
        body = {"model": "default", "messages": [{"role": "user", "content": f"{q} Answer with just the answer."}],
                "max_tokens": n, "temperature": 0.0, "seed": 0}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=1800))
        text = r["choices"][0]["message"]["content"] or ""
        toks += r.get("usage", {}).get("completion_tokens", 0)
        res.append({"q": q, "reply": text, "ok": re.search(pat, text, re.I) is not None})
    out[sname] = res
    line.append(f"{sname} {sum(r['ok'] for r in res)}/{len(res)}")
dt = time.time() - t0
json.dump(out, open(f"out/{name}.quality.json", "w"), ensure_ascii=False, indent=1)
print(f"{name}: {', '.join(line)}; {toks} tokens in {dt:.1f}s")
