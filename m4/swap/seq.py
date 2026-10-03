#!/usr/bin/env python3
"""seq.py PORT MODEL OUT.jsonl TAG SIZE... : cold long prompts (unique nonce), prompt tok/s and outcome per size.
SIZE: 13k 23k 28k 36k 64k (approximate prompt tokens; the corpus repeats past 28k)."""
import json, os, sys, time, urllib.request, uuid
C = open(os.path.expanduser("~/titan-engine/bench/prompts/corpus-sys28k.txt")).read()
CH = {"13k": 50800, "23k": 90500, "28k": 110000, "36k": 141500, "64k": 252000}
port, model, out, tag = sys.argv[1:5]
for size in sys.argv[5:]:
    text = (C * 3)[:CH[size]]
    body = {"model": model, "max_tokens": 16, "temperature": 0.0, "seed": 0, "messages": [
        {"role": "system", "content": f"[seq {uuid.uuid4().hex[:12]}] " + text},
        {"role": "user", "content": "Summarise the system text in one line."}]}
    t0 = time.time()
    r = {"tag": tag, "size": size}
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
            json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
        u = d.get("usage") or {}
        r.update(ok=True, prompt_tokens=u.get("prompt_tokens"), prompt_tok_s=u.get("prompt_tokens") / u["total_prompt_time_sec"] if u.get("total_prompt_time_sec") else None)
    except Exception as e:
        r.update(ok=False, error=str(e)[:200])
    r["wall"] = time.time() - t0
    print(time.strftime("%H:%M:%S"), json.dumps(r), flush=True)
    open(out, "a").write(json.dumps(r) + "\n")
