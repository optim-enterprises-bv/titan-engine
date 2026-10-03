#!/usr/bin/env python3
"""same.py A.json B.json: exact identity of two gate outputs (gates.py rows: top / tokens / text) or two lists."""
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
if a and isinstance(a[0], dict):
    keys = [k for k in ("top", "tokens", "text") if k in a[0] and k in b[0]]
    for k in keys:
        print(f"SAME {sys.argv[1].split('/')[-1]} vs {sys.argv[2].split('/')[-1]}: {k} identical {sum(x[k] == y[k] for x, y in zip(a, b))}/{len(a)} (len {len(b)})")
else:
    print(f"SAME {sys.argv[1].split('/')[-1]} vs {sys.argv[2].split('/')[-1]}: identical {sum(x == y for x, y in zip(a, b))}/{len(a)} (len {len(b)})")
