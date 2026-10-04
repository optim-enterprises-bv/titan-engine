#!/usr/bin/env python3
"""sp.py FILE...: llama.cpp prints control tokens (<|im_end|>, <|endoftext|>, <|return|>, ...) as "": do the same
to titan's mapped pieces (in place) before cmp.py, as m4/orca's -sp copies do."""
import json, re, sys, os
for fn in sys.argv[1:]:
    if not os.path.exists(fn):
        continue
    rows = json.load(open(fn)); n = 0
    for r in rows:
        for step in r.get("top", []):
            seen = any(p[0] == "" for p in step)
            for k, pair in enumerate(step):
                if isinstance(pair[0], str) and re.fullmatch(r"<\|[^|]*\|>", pair[0]):
                    # only the most probable control token takes "" (a dict lookup keeps the last duplicate)
                    pair[0] = f"<ctl{k}>" if seen else ""; seen = True; n += 1
    json.dump(rows, open(fn, "w"), ensure_ascii=False)
