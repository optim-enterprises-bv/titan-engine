#!/usr/bin/env python3
"""nanprobe.py REF.server.log X.server.log -- first pass: the first probe that is nan/inf in X, and per-layer
'experts' sums X vs REF (relative difference) for the first forward pass."""
import re, sys
def probes(f):
    out = []
    for l in open(f, errors="replace"):
        m = re.search(r"g4 probe (.*?): sum (\S+), max \|x\| (\S+)", l)
        if m: out.append((m.group(1), float(m.group(2).rstrip(',')), float(m.group(3))))
    return out
r, x = probes(sys.argv[1]), probes(sys.argv[2])
bad = next((i for i, p in enumerate(x) if p[1] != p[1] or abs(p[1]) == float("inf") or p[2] != p[2] or p[2] == float("inf")), None)
print(f"probes ref {len(r)} x {len(x)}; first nan/inf in x: {x[bad] if bad is not None else None} (index {bad})")
for (a, s1, m1), (b, s2, m2) in zip(r, x):
    if a != b: break
    if "experts" in a or " out" in a:
        rel = abs(s1 - s2) / (abs(s1) + 1e-3)
        flag = "!!" if rel > 0.05 or s2 != s2 else "  "
        print(f"{flag} {a:16s} ref {s1:12.4f} {m1:9.3f}  x {s2:12.4f} {m2:9.3f}  rel {rel:.3g}")
    if " out" in a and a.startswith("L29"): break
