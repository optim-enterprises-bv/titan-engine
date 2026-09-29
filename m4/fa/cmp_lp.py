#!/usr/bin/env python3
"""Compare two lp.py record files tag by tag: identical text?, first divergence, top-1 agreement,
mean KL(top-5) and max |dlogp| over the positions up to and including the first divergence (after it the
contexts differ). KL(P||Q) over P's top-5 tokens, both renormalised over that set; a token missing from Q's
top-5 gets Q's smallest top-5 logprob (an upper bound on its probability).
usage: cmp_lp.py A.jsonl B.jsonl [label]"""
import json, math, sys
A = {r["tag"].split(":")[-1]: r for r in map(json.loads, open(sys.argv[1]))}
B = {r["tag"].split(":")[-1]: r for r in map(json.loads, open(sys.argv[2]))}
label = sys.argv[3] if len(sys.argv) > 3 else ""
for t in A:
    if t not in B:
        continue
    a, b = A[t], B[t]
    n = min(len(a["tokens"]), len(b["tokens"]))
    div = next((i for i in range(n) if a["tokens"][i] != b["tokens"][i]), None)
    end = n if div is None else div + 1
    kls, dl, top1 = [], 0.0, 0
    for i in range(end):
        pa = dict(map(tuple, a["top"][i])); pb = dict(map(tuple, b["top"][i]))
        if not pa or not pb:
            continue
        floor = min(pb.values())
        ks = list(pa)
        za = sum(math.exp(pa[k]) for k in ks); zb = sum(math.exp(pb.get(k, floor)) for k in ks)
        kls.append(sum(math.exp(pa[k]) / za * ((pa[k] - math.log(za)) - (pb.get(k, floor) - math.log(zb))) for k in ks))
        top1 += max(pa, key=pa.get) == max(pb, key=pb.get)
        dl = max(dl, abs(a["logprobs"][i] - b["logprobs"][i]))
    print(f"{label} {t}: prompt {a['prompt_tokens']}/{b['prompt_tokens']}, text {'identical' if a['text'] == b['text'] else 'differs'}, "
          f"first divergence {'none' if div is None else div} of {n}, top-1 {top1}/{end}, "
          f"mean KL(top-5) {sum(kls) / max(len(kls), 1):.4f}, max |dlogp| {dl:.3f}")
