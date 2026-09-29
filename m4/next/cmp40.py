#!/usr/bin/env python3
"""cmp40.py REF NAME...: how many completions of each out/NAME.json are byte-identical to out/REF.json
(names with a slash are paths without .json)."""
import json, sys
def load(n):
    return json.load(open(n + ".json" if "/" in n else f"out/{n}.json"))
ref = load(sys.argv[1])
for n in sys.argv[2:]:
    try:
        b = load(n)
    except OSError:
        print(f"{n}: missing"); continue
    same = sum(x == y for x, y in zip(ref, b))
    print(f"{n} vs {sys.argv[1]}: {same}/{len(ref)} identical" + ("" if len(b) == len(ref) else f" (len {len(b)})"))
