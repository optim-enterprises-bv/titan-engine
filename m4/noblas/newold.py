#!/usr/bin/env python3
"""newold.py MODEL...: new (noblas) vs old (deployed) titan through the roster swap server: G1 / G3 first-token
top-20 identity, cmp.py metrics (old as the reference), prefill tok/s (G3 / G5 medians)."""
import json, math, statistics, sys, os
O = "~/titan-engine/top-noblas/m4/noblas/out"
def ld(p):
    return json.load(open(p)) if os.path.exists(p) else None
def score(t, l):
    agree, dlp, kls = 0, [], []
    for a, b in zip(t, l):
        ta, lb = a["top"][0], b["top"][0]
        tm = dict((str(k), v) for k, v in ta)
        floor = min(v for _, v in ta)
        agree += str(ta[0][0]) == str(lb[0][0])
        dlp.append(abs(tm.get(str(lb[0][0]), floor) - lb[0][1]))
        kls.append(sum(math.exp(lp) * (lp - tm.get(str(tok), floor)) for tok, lp in lb))
    return agree, statistics.median(dlp), max(dlp), statistics.median(kls), max(kls)
def med(r, k):
    v = sorted(x.get(k) or 0 for x in r)
    return v[len(v) // 2] if v else 0
for m in sys.argv[1:]:
    row = [m]
    for g in ("g1", "g3"):
        n, o = ld(f"{O}/new-{m}-{g}.json"), ld(f"{O}/old-{m}-{g}.json")
        if not n or not o or len(n) != len(o):
            row.append(f"{g}: -")
            continue
        same = sum(a["top"] == b["top"] for a, b in zip(n, o))
        ag, d, dm, k, km = score(n, o)
        row.append(f"{g}: identical {same}/{len(n)} top1 {ag}/{len(n)} dlp med {d:.4f} max {dm:.4f} KL med {k:.5f} max {km:.5f}")
    for g in ("g3", "g5"):
        n, o = ld(f"{O}/new-{m}-{g}.json"), ld(f"{O}/old-{m}-{g}.json")
        if n and o:
            a, b = med(n, "prefill_tps"), med(o, "prefill_tps")
            row.append(f"{g} prefill {a:.0f}/{b:.0f} ({100 * (a / b - 1) if b else 0:+.1f}%)")
    print(" | ".join(row))
