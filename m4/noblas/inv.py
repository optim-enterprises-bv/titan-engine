#!/usr/bin/env python3
"""inv.py PORT MODEL GLOG OUTJSON: the noblas inventory client. A 1-token warm request (loads the model), then a short
chat prompt (32 greedy tokens) and a ~4k-token raw prompt (16 greedy tokens; m4/g4s/gates.py's G3 construction), with
"M <phase>" marker lines appended to GLOG (the server's TITAN_GEMM_LOG file) before each phase."""
import json, sys, time, urllib.request
port, model, glog, outp = sys.argv[1:5]
E = "~/titan-engine/m4"


def mark(s):
    with open(glog, "a") as f:
        f.write(f"M {s} t={time.time():.3f}\n")


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1200))
    return r, time.time() - t


docs = [d["text"] for d in json.load(open(f"{E}/idle/tf/docs.json"))]
blob = "\n\n".join(docs * 4)
long_p = blob[:13500] + "\n\nIn one sentence, the text above is about"
res = {}
mark("warm")
r, w = post("/v1/completions", {"model": model, "prompt": "Hello", "max_tokens": 1, "temperature": 0})
res["warm"] = {"wall": w}
mark("short")
r, w = post("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "Explain in two sentences why the sky is blue."}],
                                      "max_tokens": 32, "temperature": 0})
res["short"] = {"wall": w, "usage": r.get("usage")}
mark("long")
try:
    r, w = post("/v1/completions", {"model": model, "prompt": long_p, "max_tokens": 16, "temperature": 0})
except Exception as e:  # noqa: BLE001 - e.g. OOM at ~3.3k tokens (gpt-oss-20b, 8k context): retry at half the length
    print(f"INV {model} long failed ({e!r:.120}); retrying with a ~1.7k-token prompt", flush=True)
    mark("long2")
    r, w = post("/v1/completions", {"model": model, "prompt": blob[:6750] + "\n\nIn one sentence, the text above is about", "max_tokens": 16, "temperature": 0})
res["long"] = {"wall": w, "usage": r.get("usage")}
mark("end")
json.dump(res, open(outp, "w"), indent=1)
for k in ("short", "long"):
    u = res[k]["usage"] or {}
    print(f"INV {model} {k}: prompt {u.get('prompt_tokens')} completion {u.get('completion_tokens')} wall {res[k]['wall']:.1f}s "
          f"prefill {u.get('avg_prompt_tok_per_sec')} decode {u.get('avg_compl_tok_per_sec')}", flush=True)
