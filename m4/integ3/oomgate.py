#!/usr/bin/env python3
"""OOM-recovery gate (REDCELL, default prefix cache): the baseline procedure G1 -> G3 -> G2 256, but every request is
sent on its own and a failed one is recorded and the run goes on (gates.py aborts on the first HTTP 500).
usage: oomgate.py PORT NAME G1_PROMPTS.json
Writes out/NAME-g1.json, NAME-g3.json, NAME-g2.json (gates.py row format; a failed request is a row with "error")
and out/NAME-oom.json (per-request status), then one short probe request after everything ("next request succeeds")."""
import json, sys, time, urllib.error, urllib.request

E = "~/titan-engine/m4"
OUT = "~/titan-engine/top-integ3/m4/integ3/out"
port, name, g1p = sys.argv[1], sys.argv[2], sys.argv[3]


def complete(prompt, n, top):
    body = {"model": "default", "prompt": prompt, "max_tokens": n, "temperature": 0, "seed": 0, "logprobs": top}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=1800))
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}", "wall": time.time() - t}
    except Exception as e:  # noqa: BLE001 - recorded (connection refused = the server died)
        return {"error": repr(e)[:300], "wall": time.time() - t}
    c = (r["choices"][0].get("logprobs") or {}).get("content") or []
    u = r.get("usage", {})
    return {"tokens": [x["token"] for x in c],
            "top": [[(y["token"], y["logprob"]) for y in (x.get("top_logprobs") or [])] for x in c],
            "text": r["choices"][0].get("text", ""), "prompt_tokens": u.get("prompt_tokens"),
            "prefill_tps": u.get("avg_prompt_tok_per_sec"), "decode_tps": u.get("avg_compl_tok_per_sec"),
            "wall": time.time() - t}


def long_prompts(chars, count):  # gates.py long_prompts
    docs = [d["text"] for d in json.load(open(f"{E}/idle/tf/docs.json"))]
    blob = "\n\n".join(docs * 4)
    step = len(docs[0]) * 5
    return [blob[i * step:i * step + chars] + "\n\nIn one sentence, the text above is about" for i in range(count)]


status = []
g1 = json.load(open(g1p))
for phase, prompts, n, top in (("g1", g1, 1, 20), ("g3", long_prompts(13500, 4), 1, 20), ("g2", g1, 256, 1)):
    rows = []
    for i, p in enumerate(prompts):
        r = complete(p, n, top)
        if phase == "g2":
            r.pop("top", None)
        rows.append(r)
        status.append({"phase": phase, "i": i, "ok": "error" not in r, "error": r.get("error", ""),
                       "prompt_tokens": r.get("prompt_tokens")})
        if "error" in r:
            print(f"  {name} {phase}[{i}] ERR {r['error'][:200]}", flush=True)
    json.dump(rows, open(f"{OUT}/{name}-{phase}.json", "w"), ensure_ascii=False)
    print(f"OOMGATE {name} {phase}: {sum('error' not in r for r in rows)}/{len(rows)} ok", flush=True)
probe = complete("The capital of France is", 8, 1)
status.append({"phase": "probe", "i": 0, "ok": "error" not in probe, "error": probe.get("error", ""), "text": probe.get("text")})
json.dump(status, open(f"{OUT}/{name}-oom.json", "w"), ensure_ascii=False)
fails = [s for s in status if not s["ok"]]
print(f"OOMGATE {name}: {len(status) - len(fails)}/{len(status)} requests ok, failed: "
      f"{[(s['phase'], s['i'], 'OOM' if 'memory' in s['error'].lower() else s['error'][:60]) for s in fails]}; "
      f"probe after: {'ok ' + repr(probe.get('text')) if 'error' not in probe else 'FAILED ' + probe['error'][:120]}")
