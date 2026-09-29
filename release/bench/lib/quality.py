#!/usr/bin/env python3
"""Quality sanity check for the campaign: the q100 short-answer set of m4/skip/quality.py (same questions, same
" Answer with just the answer." suffix, same case-insensitive answer regexes), greedy.

Difference from m4/skip/quality.py, on purpose: the prompt is sent as raw text to /v1/completions with the model
family's chat format written out here, so both engines tokenize the same bytes and no server-side template or
reasoning parser is involved (as m5/oss/practice.py does for gpt-oss). Thinking models get an empty think block
(the models' own "thinking off" form); gpt-oss gets "Reasoning: low" and is scored on its final channel.

usage: quality.py PORT FAMILY OUT.json [first:last]   (the slice selects questions, for splitting across windows)"""
import json, os, re, sys, time, urllib.request

Q100 = os.path.expanduser("~/titan-engine/m4/skip/q100.json")
HARMONY_SYS = ("<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
               "Knowledge cutoff: 2024-06\nCurrent date: 2026-09-28\n\nReasoning: low\n\n"
               "# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>")
FINAL = re.compile(r"(?:<\|channel\|>|assistant)final(?:<\|message\|>)?")
FAMILIES = {
    # family: (prompt template, max_tokens)
    "qwen-think": ("<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", 48),
    "qwen": ("<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n", 48),
    "harmony": (HARMONY_SYS + "<|start|>user<|message|>{q}<|end|><|start|>assistant", 384),
}


def answer(family, text):
    if family != "harmony":
        return text
    parts = FINAL.split(text)
    return parts[-1] if len(parts) > 1 else None  # no final channel scores 0


Q40_SEED = 20260929


def q40():
    """The fixed 40-question subset for slow models: a seeded sample of q100's indices (same on every engine)."""
    import random
    return sorted(random.Random(Q40_SEED).sample(range(100), 40))


def run(port, family, sl=None, subset=None):
    qs = json.load(open(Q100))
    lo, hi = (0, len(qs)) if not sl else map(int, sl.split(":"))
    idx = subset if subset is not None else list(range(lo, hi))
    tmpl, n = FAMILIES[family]
    res, toks, t0 = [], 0, time.time()
    for i in idx:
        q, pat = qs[i]
        body = {"model": "default", "prompt": tmpl.format(q=f"{q} Answer with just the answer."), "max_tokens": n,
                "temperature": 0.0, "top_k": 1, "seed": 0}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=1800))
        text = r["choices"][0]["text"] or ""
        toks += (r.get("usage") or {}).get("completion_tokens", 0)
        a = answer(family, text)
        res.append({"i": i, "q": q, "reply": text, "ok": a is not None and re.search(pat, a, re.I) is not None,
                    "no_final": a is None})
    return {"set": "q40" if subset is not None else "q100", "indices": idx, "slice": [lo, hi], "family": family, "max_tokens": n, "score": sum(r["ok"] for r in res),
            "n": len(res), "no_final": sum(r["no_final"] for r in res), "completion_tokens": toks,
            "wall_s": time.time() - t0, "items": res}


if __name__ == "__main__":
    out = run(sys.argv[1], sys.argv[2], sys.argv[4] if len(sys.argv) > 4 else None)
    json.dump(out, open(sys.argv[3], "w"), ensure_ascii=False, indent=1)
    print(f"q100 {out['score']}/{out['n']} ({out['completion_tokens']} tokens in {out['wall_s']:.0f}s)")
