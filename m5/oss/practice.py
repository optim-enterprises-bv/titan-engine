#!/usr/bin/env python3
"""Practice-task score: 20 short-answer questions (questions.json: [question, answer regex]) as raw harmony prompts
through /v1/completions, greedy, so both engines see the same tokens and no server-side template or reasoning
parser is involved. The answer is the final channel (text after the last `<|channel|>final<|message|>`); a reply
without one scores 0. usage: practice.py PORT NAME [MAX_TOKENS]  -> out/NAME.practice.json"""
import json, re, sys, time, urllib.request
port, name = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 768
SYS = ("<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
       "Knowledge cutoff: 2024-06\nCurrent date: 2026-09-28\n\nReasoning: low\n\n"
       "# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>")
FINAL = re.compile(r"(?:<\|channel\|>|assistant)final(?:<\|message\|>)?")
qs = json.load(open("questions.json"))
res, toks, t0 = [], 0, time.time()
for q, pat in qs:
    prompt = f"{SYS}<|start|>user<|message|>{q} Answer with just the answer.<|end|><|start|>assistant"
    body = {"model": "default", "prompt": prompt, "max_tokens": n, "temperature": 0.0, "top_k": 1, "seed": 0}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    text = r["choices"][0]["text"]
    toks += r.get("usage", {}).get("completion_tokens", 0)
    m = list(FINAL.finditer(text))
    final = text[m[-1].end():].split("<|return|>")[0].split("<|end|>")[0].strip() if m else None
    ok = final is not None and re.search(pat, final, re.I) is not None
    res.append({"q": q, "final": final, "ok": ok, "raw": text})
dt = time.time() - t0
json.dump(res, open(f"out/{name}.practice.json", "w"), ensure_ascii=False, indent=1)
print(f"{name}: practice {sum(r['ok'] for r in res)}/{len(res)} correct, {sum(r['final'] is None for r in res)} without a final "
      f"channel, {toks} tokens in {dt:.1f}s ({toks/dt:.1f} tok/s)")
