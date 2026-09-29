#!/usr/bin/env python3
"""cmp.py A B: identical completions, first-token argmax agreement and top-4 log-probs (out/A.json, out/A.top.json)."""
import json, sys
a, b = sys.argv[1], sys.argv[2]
ca, cb = json.load(open(f"out/{a}.json")), json.load(open(f"out/{b}.json"))
same = sum(x == y for x, y in zip(ca, cb))
for i, (x, y) in enumerate(zip(ca, cb)):
    k = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
    print(f"  {i}: " + ("identical" if x == y else f"diverges at char {k}/{len(x)}: {x[k:k+24]!r} vs {y[k:k+24]!r}"))
print(f"{a} vs {b}: {same}/{len(ca)} identical")
try:
    ta, tb = json.load(open(f"out/{a}.top.json")), json.load(open(f"out/{b}.top.json"))
except FileNotFoundError:
    sys.exit(0)
arg = sum(x[0][0] == y[0][0] for x, y in zip(ta, tb))
gap = lambda top: [(t, round(l - top[0][1], 2)) for t, l in top[:4]]
for i, (x, y) in enumerate(zip(ta, tb)):
    print(f"  {i} {a:>10}: {gap(x)}")
    print(f"  {i} {b:>10}: {gap(y)}")
print(f"{a} vs {b}: first-token argmax {arg}/{len(ta)} agree (top-4 shown as gaps to the top-1 score)")
