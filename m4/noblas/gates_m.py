#!/usr/bin/env python3
"""G1-G3/G5 client for titan (/v1/completions) and llama-server (/completion), raw pre-rendered prompts.
usage: gates.py MODE PORT KIND OUT [ARGS]
  g1 PORT KIND OUT PROMPTS.json            first token, top-20 logprobs, 40 prompts
  g2 PORT KIND OUT PROMPTS.json N           greedy N tokens per prompt (tokens + text + decode tok/s)
  g3 PORT KIND OUT                          4 long (~4k token) prompts from idle/tf/docs.json, first token top-20
  g5 PORT KIND OUT CHARS N                  two ~CHARS-char prompts, N greedy tokens, prefill/decode tok/s
  g6 PORT KIND OUT G6.json on|off           first token top-20 on LoRA training-style prompts, adapter on/off
KIND is titan or llama. Logprobs are natural log on both."""
import json, os, subprocess, sys, time, urllib.request

E = "~/titan-engine/m4"
mode, port, kind, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]


def post(path, body, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


ADAPTER = None
# YIELD_LOCK=1 (CPU reference runs outside a window): stop before any request while the GPU lock is held by
# another agent, keeping finished results in OUT.partial; a later run resumes from it.
YIELD = os.environ.get("YIELD_LOCK") == "1"
LOCK = "~/titan-engine/.gpu.lock"
PARTIAL = out + ".partial"
done = json.load(open(PARTIAL)) if YIELD and os.path.exists(PARTIAL) else []


def lock_busy():
    return YIELD and subprocess.run(["flock", "-n", LOCK, "true"]).returncode != 0


def complete(prompt, n, top):
    """-> dict(tokens=[str], top=[[(tok, lp)]] per step, text, prompt_tokens, prefill_tps, decode_tps)"""
    t = time.time()
    if kind == "llama":
        body = {"prompt": prompt, "n_predict": n, "temperature": 0, "top_k": 1, "seed": 0,
                "n_probs": top, "post_sampling_probs": False, "cache_prompt": False}
        if ADAPTER is not None:
            body["lora"] = [{"id": 0, "scale": 1.0 if ADAPTER else 0.0}]
        r = post("/completion", body)
        cp = r.get("completion_probabilities") or []
        tm = r.get("timings", {})
        return {"tokens": [c["token"] for c in cp],
                "top": [[(x["token"], x["logprob"]) for x in c["top_logprobs"]] for c in cp],
                "text": r.get("content", ""), "prompt_tokens": tm.get("prompt_n"),
                "prefill_tps": tm.get("prompt_per_second"), "decode_tps": tm.get("predicted_per_second"),
                "wall": time.time() - t}
    body = {"model": os.environ.get("MODEL", "default"), "prompt": prompt, "max_tokens": n, "temperature": 0, "seed": 0, "logprobs": top}
    if ADAPTER:
        body["adapter"] = "spark"
    r = post("/v1/completions", body)
    c = (r["choices"][0].get("logprobs") or {}).get("content") or []
    u = r.get("usage", {})
    return {"tokens": [x["token"] for x in c],
            "top": [[(y["token"], y["logprob"]) for y in (x.get("top_logprobs") or [])] for x in c],
            "text": r["choices"][0].get("text", ""), "prompt_tokens": u.get("prompt_tokens"),
            "prefill_tps": u.get("avg_prompt_tok_per_sec"), "decode_tps": u.get("avg_compl_tok_per_sec"),
            "wall": time.time() - t}


def long_prompts(chars, count):
    docs = [d["text"] for d in json.load(open(f"{E}/idle/tf/docs.json"))]
    blob = "\n\n".join(docs * 4)
    step = len(docs[0]) * 5
    return [blob[i * step:i * step + chars] + "\n\nIn one sentence, the text above is about" for i in range(count)]


res = list(done)


def run_all(prompts, fn):
    for i, p in enumerate(prompts):
        if i < len(res):
            continue
        if lock_busy():
            json.dump(res, open(PARTIAL, "w"), ensure_ascii=False)
            print(f"{mode}: yielding to the GPU lock after {len(res)}/{len(prompts)}", flush=True)
            sys.exit(3)
        res.append(fn(p))


if mode == "g1":
    run_all(json.load(open(sys.argv[5])), lambda p: complete(p, 1, 20))
elif mode == "g2":
    n = int(sys.argv[6])
    run_all(json.load(open(sys.argv[5])), lambda p: {k: v for k, v in complete(p, n, 1).items() if k != "top"})
elif mode == "g3":
    run_all(long_prompts(int(sys.argv[5]) if len(sys.argv) > 5 else 13500, 4), lambda p: complete(p, 1, 20))
elif mode == "g5":
    chars, n = int(sys.argv[5]), int(sys.argv[6])
    run_all(long_prompts(chars, 2), lambda p: {k: v for k, v in complete(p, n, 1).items() if k != "top"})
elif mode == "g6":
    ADAPTER = sys.argv[6] == "on"
    run_all([row["prompt"] for row in json.load(open(sys.argv[5]))], lambda p: complete(p, 1, 20))
elif mode == "tie":
    # tie PORT llama OUT PROMPTS T_G2 L_G2: llama's logprob gap at each first divergence (teacher-forced prefix)
    prompts, tg, lg = (json.load(open(f)) for f in sys.argv[5:8])
    for i, (p, t, l) in enumerate(zip(prompts, tg, lg)):
        k = next((j for j, (x, y) in enumerate(zip(t["tokens"], l["tokens"])) if x != y), None)
        if k is None:
            continue
        r = complete(p + "".join(l["tokens"][:k]), 1, 20)
        top = dict(r["top"][0])
        floor = min(top.values())
        res.append({"prompt": i, "pos": k, "llama_tok": l["tokens"][k], "titan_tok": t["tokens"][k],
                    "gap": top.get(l["tokens"][k], floor) - top.get(t["tokens"][k], floor),
                    "prompt_tokens": r["prompt_tokens"], "prefill_tps": 0, "decode_tps": 0})
    gaps = sorted(x["gap"] for x in res)
    print("tie: llama logprob gap (its token - titan token) at first divergence: median %.3f, max %.3f, >1.0: %d/%d"
          % (gaps[len(gaps) // 2], gaps[-1], sum(g > 1.0 for g in gaps), len(gaps)))
else:
    sys.exit(f"unknown mode {mode}")
json.dump(res, open(out, "w"), ensure_ascii=False)
if os.path.exists(PARTIAL):
    os.remove(PARTIAL)
pt = [r.get("prompt_tokens") for r in res]
dec = sorted(r["decode_tps"] or 0 for r in res)
pre = sorted(r["prefill_tps"] or 0 for r in res)
print(f"{mode} {kind} -> {out}: {len(res)} requests, prompt_tokens {pt[:4]}..., median decode {dec[len(dec)//2]:.1f} tok/s, "
      f"median prefill {pre[len(pre)//2]:.1f} tok/s", flush=True)
