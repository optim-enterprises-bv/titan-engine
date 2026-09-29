#!/usr/bin/env python3
"""pcmp.py A B: practice scores of two runs and per-question agreement (out/A.practice.json)."""
import json, sys
a, b = sys.argv[1], sys.argv[2]
ra, rb = json.load(open(f"out/{a}.practice.json")), json.load(open(f"out/{b}.practice.json"))
for i, (x, y) in enumerate(zip(ra, rb)):
    flag = "" if x["ok"] == y["ok"] else "  <- differs"
    print(f"  {i:2} {'+' if x['ok'] else '-'}{'+' if y['ok'] else '-'} {str(x['final'])[:40]!r:44} {str(y['final'])[:40]!r}{flag}")
sa, sb = sum(r["ok"] for r in ra), sum(r["ok"] for r in rb)
same = sum(x["raw"] == y["raw"] for x, y in zip(ra, rb))
print(f"practice: {a} {sa}/{len(ra)}, {b} {sb}/{len(rb)}; {same}/{len(ra)} replies byte-identical")
