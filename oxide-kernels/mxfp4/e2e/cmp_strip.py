#!/usr/bin/env python3
"""cmp_strip.py A B: out/A.json vs out/B.json greedy texts, ignoring leading whitespace (mistral.rs
strips it from completions; llama.cpp keeps it)."""
import json, sys
a, b = json.load(open(f"out/{sys.argv[1]}.json")), json.load(open(f"out/{sys.argv[2]}.json"))
same = 0
for i, (x, y) in enumerate(zip(a, b)):
    x, y = x.lstrip(), y.lstrip()
    k = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
    ok = x == y; same += ok
    print(f"  {i}: {'identical' if ok else f'diverges at char {k}/{len(x)}: {x[k:k+30]!r} vs {y[k:k+30]!r}'}")
print(f"{sys.argv[1]} vs {sys.argv[2]} (leading whitespace ignored): {same}/{len(a)} identical")
