#!/usr/bin/env python3
"""firstdiff.py A.server.log B.server.log | A.server.log --halves -- the g4 probe lines (TITAN_G4_MEMLOG=2) of two runs side by side:
count, first differing probe (with its index = forward pass / layer / op), and how many differ."""
import re, sys
def probes(f):
    return [re.sub(r".*g4 probe ", "", l.rstrip()) for l in open(f, errors="replace") if "g4 probe " in l]
if sys.argv[2] == "--halves":  # one server, the same request twice
    a = probes(sys.argv[1]); h = len(a) // 2; a, b = a[:h], a[h:2 * h]
else:
    a, b = probes(sys.argv[1]), probes(sys.argv[2])
n = min(len(a), len(b))
d = [i for i in range(n) if a[i] != b[i]]
print(f"probes {len(a)} vs {len(b)}; differing {len(d)} of {n}")
if d:
    i = d[0]
    lo = max(0, i - 3)
    for j in range(lo, min(n, i + 4)):
        print(("!! " if a[j] != b[j] else "   ") + f"[{j}] A {a[j]}  |  B {b[j]}")
