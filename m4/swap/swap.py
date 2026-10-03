#!/usr/bin/env python3
"""Swap-mode test client.

    swap.py cycle PORT OUT.json MODEL...   one question per model in order (each may swap); wall time per step,
                                           answer check, /v1/models status and nvidia-smi after each step
    swap.py queue PORT OUT.json MODEL N    N concurrent requests to MODEL (the first triggers the swap), all must succeed
    swap.py mixed PORT OUT.json A B        concurrent: 2 x A and 2 x B at once (both queue behind the swaps)
    swap.py models PORT                    print /v1/models and /ui/api/list_models
    swap.py oom PORT OUT.json MODEL        13k cold, then ~60k (may fail), then 13k cold: the last must succeed
"""
import json, subprocess, sys, threading, time, urllib.request, uuid

Q = ("What is the capital of France? Answer with the city name only.", "paris")
CORPUS = open(__import__("os").path.expanduser("~/titan-engine/bench/prompts/corpus-sys28k.txt")).read()


def post(port, model, content, max_tokens=256, timeout=1800, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
    body = {"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": 0.0, "seed": 0}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=timeout))
        m = d["choices"][0]["message"]
        return {"ok": True, "wall": time.time() - t0, "content": m.get("content") or "",
                "reasoning": (m.get("reasoning_content") or "")[:200], "finish": d["choices"][0].get("finish_reason"),
                "usage": d.get("usage"), "model": d.get("model")}
    except Exception as e:
        body = getattr(e, "read", lambda: b"")()
        return {"ok": False, "wall": time.time() - t0, "error": f"{e} {body[:300]!r}"}


def get(port, path):
    try:
        return json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=30))
    except Exception as e:
        return {"error": str(e)}


def status(port):
    d = get(port, "/v1/models")
    return {m["id"]: m.get("status") for m in d.get("data", [])} if "data" in d else d


def smi():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:
        return str(e)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ask(port, model):
    r = post(port, model, Q[0])
    r["correct"] = r["ok"] and Q[1] in (r["content"] + " " + r.get("reasoning", "")).lower()
    return r


def cycle(port, out, models):
    res = []
    for m in models:
        r = ask(port, m)
        r.update(step=m, status=status(port), smi=smi())
        res.append(r)
        log(f"{m}: ok={r['ok']} correct={r.get('correct')} wall {r['wall']:.1f}s "
            f"answer {r.get('content', r.get('error'))!r:.80} | smi used,free {r['smi']} | {r['status']}")
    json.dump(res, open(out, "w"), indent=1)
    return all(r.get("correct") for r in res)


def conc(port, jobs):
    out = [None] * len(jobs)

    def run(i, m):
        out[i] = dict(ask(port, m), model_req=m, t_start=time.time())

    ts = [threading.Thread(target=run, args=(i, m)) for i, m in enumerate(jobs)]
    for i, t in enumerate(ts):
        t.start()
        time.sleep(0.05)
    for t in ts:
        t.join()
    return out


def queue(port, out, model, n):
    res = conc(port, [model] * n)
    for r in res:
        log(f"queued {r['model_req']}: ok={r['ok']} correct={r.get('correct')} wall {r['wall']:.1f}s {r.get('content', r.get('error'))!r:.60}")
    json.dump(res, open(out, "w"), indent=1)
    return all(r.get("correct") for r in res)


def mixed(port, out, a, b):
    res = conc(port, [a, b, a, b])
    for r in res:
        log(f"mixed {r['model_req']}: ok={r['ok']} correct={r.get('correct')} wall {r['wall']:.1f}s {r.get('content', r.get('error'))!r:.60}")
    json.dump(res, open(out, "w"), indent=1)
    return all(r.get("correct") for r in res)


def oom(port, out, model):
    res = []
    for key, chars, must in (("13k", 50800, True), ("60k", 236000, False), ("13k", 50800, True), ("short", 0, True)):
        nonce = uuid.uuid4().hex[:12]
        if chars:
            text = (CORPUS * (chars // len(CORPUS) + 1))[:chars]
            r = post(port, model, "Summarise the system text in one line.", max_tokens=16, system=f"[oom {nonce}] {text}")
        else:
            r = ask(port, model)
        u = r.get("usage") or {}
        pt, ptime = u.get("prompt_tokens"), u.get("total_prompt_time_sec")
        r.update(step=key, prompt_tok_s=(pt / ptime if pt and ptime else None), smi=smi())
        res.append(r)
        log(f"oom {key}: ok={r['ok']} wall {r['wall']:.1f}s prompt {pt} tok {r['prompt_tok_s'] or 0:.0f} tok/s "
            f"{(r.get('content') or r.get('error') or '')!r:.120} | smi {r['smi']}")
        if must and not r["ok"]:
            log(f"oom {key}: FAILED (must succeed)")
    json.dump(res, open(out, "w"), indent=1)
    return res[2]["ok"] and res[3]["ok"]


def main():
    cmd, port = sys.argv[1], sys.argv[2]
    if cmd == "models":
        print(json.dumps(get(port, "/v1/models"), indent=1)[:3000])
        print(json.dumps(get(port, "/ui/api/list_models"))[:3000])
        return
    out = sys.argv[3]
    ok = {"cycle": lambda: cycle(port, out, sys.argv[4:]),
          "queue": lambda: queue(port, out, sys.argv[4], int(sys.argv[5])),
          "mixed": lambda: mixed(port, out, sys.argv[4], sys.argv[5]),
          "oom": lambda: oom(port, out, sys.argv[4])}[cmd]()
    log(f"{cmd}: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
