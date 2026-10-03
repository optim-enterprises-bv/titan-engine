#!/usr/bin/env python3
"""~N-token prompt (m4/bigprompt.py's text as the system message, b2client's raw rendering), titan only.
    bigtest.py run PORT NAME N    greedy 64-token completion + first-token top-20: out/NAME.big.json
    bigtest.py cmp A B            text identity (first differing char) + first-token top-1 / KL(A || B)"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b2client as c
B = c.B
CHUNK = "The titan engine tiers mixture-of-experts weights between the GPU and the CPU. "


def run(port, name, n):
    pr = c.render("Summarise the system text in one line.", system=CHUNK * (n // 16))
    text, ntok, _, dec, _, _ = c.complete("titan", port, pr, 64, c.GREEDY)
    d = json.load(c.post(port, "/v1/completions", {"model": "default", "prompt": pr, "max_tokens": 1, "temperature": 0, "logprobs": 20}))
    lp = d["choices"][0]["logprobs"] or {}
    cc = lp.get("content") or []
    top = [(t["token"], t["logprob"]) for t in cc[0]["top_logprobs"]] if cc else \
        sorted(((k, v) for k, v in (lp.get("top_logprobs") or [{}])[0].items()), key=lambda x: -x[1])
    ptoks = (d.get("usage") or {}).get("prompt_tokens")
    json.dump({"text": text, "tokens": ntok, "top": top, "prompt_tokens": ptoks, "decode_tok_s": dec},
              open(f"{B}/out/{name}.big.json", "w"), ensure_ascii=False, indent=1)
    print(f"BIG {name}: {ptoks} prompt tokens, {ntok} tokens, {text[:80]!r}")


def cmp(a, b):
    A, Bb = json.load(open(f"{B}/out/{a}.big.json")), json.load(open(f"{B}/out/{b}.big.json"))
    x, y = A["text"], Bb["text"]
    if x == y:
        print(f"GATE big text identity {b} vs {a}: identical ({len(x)} chars, {A['prompt_tokens']} prompt tokens)")
    else:
        k = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
        print(f"GATE big text identity {b} vs {a}: DIFFER at char {k}: {x[max(0,k-40):k+40]!r} vs {y[max(0,k-40):k+40]!r}")
    r = sorted(A["top"], key=lambda t: -t[1]); o = sorted(Bb["top"], key=lambda t: -t[1])
    om = dict((str(t), v) for t, v in o)
    floor = min(v for _, v in o) - math.log(10)
    pr = [math.exp(v) for _, v in r]; zr = sum(pr)
    qo = [math.exp(om.get(str(t), floor)) for t, _ in r]; zo = sum(qo)
    kl = sum((p / zr) * math.log((p / zr) / (q / zo)) for p, q in zip(pr, qo))
    print(f"GATE big first token {b} vs {a}: top-1 {'same' if str(r[0][0]) == str(o[0][0]) else 'DIFFERENT'} ({r[0][0]!r} vs {o[0][0]!r}), "
          f"top-5 set {'same' if set(str(t) for t, _ in r[:5]) == set(str(t) for t, _ in o[:5]) else 'different'}, KL {kl:.2e}, "
          f"max |dlogprob| top-5 {max(abs(v - om.get(str(t), floor)) for t, v in r[:5]):.2e}")


if __name__ == "__main__":
    a = sys.argv[1:]
    run(a[1], a[2], int(a[3])) if a[0] == "run" else cmp(a[1], a[2])
