#!/usr/bin/env python3
"""diag.py PORT CHARS N -- one long_prompts() prompt (as gates.py g5 prompt 0); prints status, text or the error body."""
import json, sys, urllib.request, urllib.error
port, chars, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
docs = [d["text"] for d in json.load(open("~/titan-engine/m4/idle/tf/docs.json"))]
p = "\n\n".join(docs * 4)[:chars] + "\n\nIn one sentence, the text above is about"
body = {"model": "default", "prompt": p, "max_tokens": n, "temperature": 0, "seed": 0}
try:
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(body).encode(),
                                                                {"Content-Type": "application/json"}), timeout=900))
    print("diag OK", chars, repr(r["choices"][0]["text"]), r["usage"]["prompt_tokens"], r["usage"].get("avg_prompt_tok_per_sec"), flush=True)
except urllib.error.HTTPError as e:
    print("diag HTTP", e.code, chars, e.read()[:500], flush=True)
