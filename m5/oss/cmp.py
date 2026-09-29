#!/usr/bin/env python3
"""cmp.py A B: completions identical after stripping leading whitespace, exact (byte) identity, first-token
argmax agreement and top-4 (out/A.json, out/A.top.json). Scores are compared as gaps to the top-1."""
import json, sys
a, b = sys.argv[1], sys.argv[2]
ca, cb = json.load(open(f"out/{a}.json")), json.load(open(f"out/{b}.json"))
exact = sum(x == y for x, y in zip(ca, cb))
same = sum(x.lstrip() == y.lstrip() for x, y in zip(ca, cb))
for i, (x, y) in enumerate(zip(ca, cb)):
    x, y = x.lstrip(), y.lstrip()
    k = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
    print(f"  {i}: " + ("identical" if x == y else f"diverges at char {k}/{len(x)}: {x[k:k+24]!r} vs {y[k:k+24]!r}"))
print(f"{a} vs {b}: {same}/{len(ca)} identical after lstrip, {exact}/{len(ca)} byte-identical")
try:
    ua, ub = json.load(open(f"out/{a}.usage.json")), json.load(open(f"out/{b}.usage.json"))
    print(f"prompt tokens {a}: {[u[0] for u in ua]}\nprompt tokens {b}: {[u[0] for u in ub]}")
except FileNotFoundError:
    pass
try:
    ta, tb = json.load(open(f"out/{a}.top.json")), json.load(open(f"out/{b}.top.json"))
except FileNotFoundError:
    sys.exit(0)
norm = lambda t: t.strip() if isinstance(t, str) else str(t)
arg = sum(norm(x[0][0]) == norm(y[0][0]) for x, y in zip(ta, tb))
top4 = sum({norm(t) for t, _ in x[:4]} == {norm(t) for t, _ in y[:4]} for x, y in zip(ta, tb))
gap = lambda top: [(t, round(l - top[0][1], 2)) for t, l in top[:4]]
for i, (x, y) in enumerate(zip(ta, tb)):
    print(f"  {i} {a:>12}: {gap(x)}")
    print(f"  {i} {b:>12}: {gap(y)}")
print(f"{a} vs {b}: first-token argmax {arg}/{len(ta)} agree, top-4 set {top4}/{len(ta)} agree")
