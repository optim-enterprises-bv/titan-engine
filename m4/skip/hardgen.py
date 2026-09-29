#!/usr/bin/env python3
"""Generation for the harder gate: 100 HumanEval + 60 GSM8K (fixed subset, seed 20260928), interleaved, chat endpoint,
greedy, max_tokens 512. Resumable: appends one JSON line per item to out/NAME.hard.jsonl and skips items already there.
Stops (cleanly) when the time budget runs out.  usage: hardgen.py PORT NAME BUDGET_SECONDS"""
import gzip, json, os, random, sys, time, urllib.request
port, name, budget = sys.argv[1], sys.argv[2], float(sys.argv[3])
t_end = time.time() + budget
D = os.path.dirname(os.path.abspath(__file__)) + "/data"
he = [json.loads(l) for l in gzip.open(f"{D}/HumanEval.jsonl.gz", "rt")]
gs = [json.loads(l) for l in open(f"{D}/gsm8k-test.jsonl")]
rng = random.Random(20260928)
he_ids, gs_ids = rng.sample(range(len(he)), 100), rng.sample(range(len(gs)), 60)
items = []
for i in range(100):
    items.append(("he", he_ids[i]))
    if i % 5 < 3 and len([x for x in items if x[0] == "gs"]) < 60:
        items.append(("gs", gs_ids[len([x for x in items if x[0] == "gs"])]))
assert len(items) == 160
def prompt(kind, i):
    if kind == "he":
        return ("Complete the following Python function. Reply with the complete function (including the signature and any "
                "imports it needs) in a single ```python code block, and nothing else.\n\n```python\n" + he[i]["prompt"] + "```")
    return (gs[i]["question"] + "\nSolve it step by step, then give the final answer as a plain number on the last line "
            "in the form '#### <number>'.")
path = f"out/{name}.hard.jsonl"
done = set()
if os.path.exists(path):
    done = {(r["kind"], r["idx"]) for r in map(json.loads, open(path))}
toks = dt = n = 0
with open(path, "a") as f:
    for kind, i in items:
        if (kind, i) in done:
            continue
        if time.time() > t_end:
            print(f"{name}: time budget reached"); break
        body = {"model": "default", "messages": [{"role": "user", "content": prompt(kind, i)}],
                "max_tokens": 512, "temperature": 0.0, "seed": 0}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t = time.time()
        r = json.load(urllib.request.urlopen(req, timeout=900))
        el = time.time() - t
        c = r.get("usage", {}).get("completion_tokens", 0)
        toks += c; dt += el; n += 1
        f.write(json.dumps({"kind": kind, "idx": i, "reply": r["choices"][0]["message"]["content"] or "",
                            "tokens": c, "secs": round(el, 2), "finish": r["choices"][0].get("finish_reason")}) + "\n")
        f.flush()
print(f"{name}: {n} new items, {toks} tokens in {dt:.0f}s ({toks / max(dt, 1e-9):.1f} tok/s incl. prefill), "
      f"{len(done) + n}/160 done")
