#!/usr/bin/env python3
"""Greedy tokens with top-5 logprobs after a long prompt (gate b).
usage: lp.py PORT SYSFILE CHARS OUT TAG MAXTOK   (CHARS=0: the whole file). Appends a JSON record to OUT."""
import json, sys, time, urllib.request
port, sysf, chars, out, tag, maxtok = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6])
text = open(sysf).read()
if chars > 0:
    text = text[:chars]
body = {"model": "default", "messages": [{"role": "system", "content": text},
        {"role": "user", "content": "Explain the system text above section by section, in as much detail as you can. Do not stop early."}],
        "max_tokens": maxtok, "temperature": 0, "seed": 0, "logprobs": True, "top_logprobs": 5}
t = time.time()
try:
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1500))
except Exception as e:
    print(f"{tag}: FAILED {e}", flush=True); sys.exit(1)
u = r["usage"]; m = r["choices"][0]["message"]
lp = (r["choices"][0].get("logprobs") or {}).get("content") or []
rec = {"tag": tag, "prompt_tokens": u.get("prompt_tokens"), "prompt_time": u.get("total_prompt_time_sec"),
       "decode_tok_s": u.get("avg_compl_tok_per_sec"), "wall": time.time() - t,
       "text": (m.get("reasoning_content") or "") + "|" + (m.get("content") or ""),
       "tokens": [x.get("token") for x in lp], "logprobs": [x.get("logprob") for x in lp],
       "top": [[(y.get("token"), y.get("logprob")) for y in (x.get("top_logprobs") or [])] for x in lp]}
open(out, "a").write(json.dumps(rec) + "\n")
print(f"{tag}: prompt {rec['prompt_tokens']} tok, {len(lp)} tokens, text {rec['text'][:60]!r}", flush=True)
