#!/usr/bin/env python3
"""pdump.py PORT CHARS IDX OUT.json: send gates.py's long prompt IDX (of 4, CHARS chars) to titan, 1 token, top-20;
writes the response (prompt_tokens, top) to OUT.json. Used with TITAN_PATTN_DUMP to dump one prompt's prefill."""
import json, sys, urllib.request
E = "~/titan-engine/m4"
port, chars, idx, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
docs = [d["text"] for d in json.load(open(f"{E}/idle/tf/docs.json"))]
blob = "\n\n".join(docs * 4)
step = len(docs[0]) * 5
p = blob[idx * step:idx * step + chars] + "\n\nIn one sentence, the text above is about"
body = {"model": "default", "prompt": p, "max_tokens": 1, "temperature": 0, "seed": 0, "logprobs": 20}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=1800))
c = (r["choices"][0].get("logprobs") or {}).get("content") or []
res = {"prompt_tokens": r.get("usage", {}).get("prompt_tokens"), "top": [[(y["token"], y["logprob"]) for y in (x.get("top_logprobs") or [])] for x in c]}
json.dump(res, open(out, "w"))
print("pdump", out, res["prompt_tokens"], res["top"][0][:3] if res["top"] else None)
