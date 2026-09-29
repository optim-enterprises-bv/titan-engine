#!/usr/bin/env python3
"""One chat request with a long system text; prints prompt tokens / prompt tok/s and appends a record to OUT.
usage: prompt.py PORT SYSFILE CHARS OUT TAG   (CHARS=0: the whole file; 'big:N' = bigprompt.py's text of ~N tokens)"""
import json, sys, time, urllib.request
port, sysf, chars, out, tag = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
if chars.startswith("big:"):
    text = "The titan engine tiers mixture-of-experts weights between the GPU and the CPU. " * (int(chars[4:]) // 16)
else:
    text = open(sysf).read()
    if int(chars) > 0:
        text = text[: int(chars)]
body = {"model": "default", "messages": [{"role": "system", "content": text},
        {"role": "user", "content": "Summarise the system text in one line."}],
        "max_tokens": 32, "temperature": 0, "seed": 0, "logprobs": True, "top_logprobs": 5}
t = time.time()
try:
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1500))
except Exception as e:
    print(f"{tag}: FAILED {e}", flush=True); sys.exit(1)
wall = time.time() - t
u = r["usage"]; m = r["choices"][0]["message"]
lp = (r["choices"][0].get("logprobs") or {}).get("content") or []
rec = {"tag": tag, "prompt_tokens": u.get("prompt_tokens"), "prompt_time": u.get("total_prompt_time_sec"),
       "prompt_tok_s": u.get("avg_prompt_tok_per_sec"), "decode_tok_s": u.get("avg_compl_tok_per_sec"), "wall": wall,
       "text": (m.get("reasoning_content") or "") + "|" + (m.get("content") or ""),
       "tokens": [x.get("token") for x in lp], "logprobs": [x.get("logprob") for x in lp],
       "top": [[(y.get("token"), y.get("logprob")) for y in (x.get("top_logprobs") or [])] for x in lp]}
with open(out, "a") as f:
    f.write(json.dumps(rec) + "\n")
print(f"{tag}: prompt {rec['prompt_tokens']} tok in {rec['prompt_time']:.2f} s = {rec['prompt_tok_s']:.0f} tok/s, wall {wall:.1f} s, out {rec['text'][:60]!r}", flush=True)
