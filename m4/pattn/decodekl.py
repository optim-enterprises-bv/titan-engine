#!/usr/bin/env python3
"""Decode-time distribution check (what an FP8 KV cache changes: every decode step reads the quantized cache; the first
token of a single-chunk prompt does not, so G1 cannot see it).
    decodekl.py run PORT PROMPTS.json OUT.json [N=16] [MAX=128]   greedy, top-20 logprobs at every generated position
    decodekl.py cmp REF.json OURS.json                             per position up to and including the first divergent
                                                                     token (same context on both sides): top-1 agreement,
                                                                     |dlogprob| of REF's token, KL(ref || ours) on REF's
                                                                     top-20 (renormalised; OURS' missing tokens floored)"""
import json, math, statistics as st, sys, urllib.request


def run(port, pf, out, n=16, mx=128):
    res = []
    for p in json.load(open(pf))[:int(n)]:
        body = {"model": "default", "prompt": p, "max_tokens": int(mx), "temperature": 0, "seed": 0, "logprobs": 20}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=1800))
        c = (r["choices"][0].get("logprobs") or {}).get("content") or []
        res.append({"tokens": [x["token"] for x in c], "top": [[(y["token"], y["logprob"]) for y in x.get("top_logprobs") or []] for x in c]})
    json.dump(res, open(out, "w"))
    print(f"decodekl run -> {out}: {len(res)} prompts, {sum(len(r['tokens']) for r in res)} tokens")


def cmp(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    agree = tot = div = 0
    dl, kls = [], []
    for ra, rb in zip(A, B):
        for i, (ta, tb) in enumerate(zip(ra["top"], rb["top"])):
            if not ta or not tb:
                break
            pa = dict(ta); pb = dict(tb)
            tot += 1
            agree += ta[0][0] == tb[0][0]
            dl.append(abs(pa[ta[0][0]] - pb.get(ta[0][0], min(pb.values()) - 1.0)))
            za = sum(math.exp(v) for v in pa.values())
            floor = min(pb.values()) - 1.0
            zb = sum(math.exp(pb.get(k, floor)) for k in pa)
            kls.append(sum(math.exp(v) / za * ((v - math.log(za)) - (pb.get(k, floor) - math.log(zb))) for k, v in pa.items()))
            if ra["tokens"][i] != rb["tokens"][i]:
                div += 1
                break
    q = sorted(kls)
    print(f"decodekl {b.split('/')[-1]} vs {a.split('/')[-1]}: {tot} positions in {len(A)} prompts, {div} diverged; top-1 agree "
          f"{agree}/{tot}; |dlogprob| median {st.median(dl):.4f} p99 {sorted(dl)[int(0.99 * (len(dl) - 1))]:.4f}; "
          f"KL median {st.median(q):.5f} mean {st.mean(q):.5f} p99 {q[int(0.99 * (len(q) - 1))]:.5f} max {q[-1]:.5f}")


if __name__ == "__main__":
    {"run": run, "cmp": cmp}[sys.argv[1]](*sys.argv[2:])
