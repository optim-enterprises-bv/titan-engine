#!/usr/bin/env python3
"""Deltas between two bench runs.

    python3 bench/compare.py A B [--all] [--tier e2e,path,kernel] [--no-color]

A and B are run ids (bench/runs/<id>), a unique id prefix, a label (its latest run), or a summary.json path.
Change = (B - A) / A. A metric is a REGRESSION when B is worse than A by more than 3% in the metric's direction
(tok/s higher is better; us, s, ms and ours/llama ratios lower is better); regressions print red/bold.
By default only the headline tiers are shown (e2e, path, kernel ours + ratio); --all adds llama.cpp times,
the path breakdown and the GPU time split."""
import json, os, sys

B = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = [a for a in sys.argv[1:] if a.startswith("--")]
if len(args) != 2:
    print(__doc__)
    sys.exit(2)
runs = [json.loads(l) for l in open(f"{B}/history.jsonl")] if os.path.exists(f"{B}/history.jsonl") else []


def resolve(k):
    if os.path.isfile(k):
        return json.load(open(k))
    for r in runs:
        if r["id"] == k:
            return r
    m = [r for r in runs if r["id"].startswith(k)]
    if len(m) == 1:
        return m[0]
    m = [r for r in runs if r.get("label") == k]
    if m:
        return m[-1]
    sys.exit(f"no run matches {k!r} (ids: {', '.join(r['id'] for r in runs)})")


a, b = resolve(args[0]), resolve(args[1])
color = sys.stdout.isatty() and "--no-color" not in opts
tiers = {"e2e", "path", "kernel"}
if "--all" in opts:
    tiers |= {"kernel_info", "path_info", "split"}
for o in opts:
    if o.startswith("--tier"):
        tiers = set(o.split("=", 1)[1].split(",")) if "=" in o else tiers
RED, BOLD, GREEN, DIM, RST = ("\033[31m", "\033[1m", "\033[32m", "\033[2m", "\033[0m") if color else ("",) * 5

print(f"A = {a['id']} ({'CONTAMINATED' if a.get('contaminated') else 'clean'}, identity {'ok' if a.get('identity_ok') else 'FAIL'})")
print(f"B = {b['id']} ({'CONTAMINATED' if b.get('contaminated') else 'clean'}, identity {'ok' if b.get('identity_ok') else 'FAIL'})")
for r in (a, b):
    for x in r.get("reasons", []):
        print(f"  {r['id']}: {x}")
print()
reg = imp = 0
w = max([len(n) for n in a["metrics"] if n in b["metrics"]] + [10])
print(f"{'metric':{w}s} {'A':>11s} {'B':>11s} {'change':>8s}  spread A/B")
for n, ma in a["metrics"].items():
    mb = b["metrics"].get(n)
    if not mb or ma["tier"] not in tiers or ma["v"] in (None, 0) or mb["v"] is None:
        continue
    ch = (mb["v"] - ma["v"]) / abs(ma["v"])
    worse = ch < -0.03 if ma["better"] == "higher" else ch > 0.03 if ma["better"] == "lower" else False
    better = ch > 0.03 if ma["better"] == "higher" else ch < -0.03 if ma["better"] == "lower" else False
    sp = "/".join("-" if m.get("spread") is None else f"{100 * m['spread']:.1f}%" for m in (ma, mb))
    line = f"{n:{w}s} {ma['v']:11.4g} {mb['v']:11.4g} {100 * ch:+7.1f}%  {sp}"
    if worse:
        reg += 1
        print(f"{RED}{BOLD}{line}  REGRESSION{RST}")
    elif better:
        imp += 1
        print(f"{GREEN}{line}{RST}")
    else:
        print(f"{DIM}{line}{RST}" if color else line)
ia, ib = a.get("identity", {}), b.get("identity", {})
print()
for k in sorted(set(ia) | set(ib)):
    flag = "" if ia.get(k) == ib.get(k) else f"  {RED}{BOLD}CHANGED{RST}"
    print(f"identity {k}: {ia.get(k)} -> {ib.get(k)}{flag}")
print(f"\n{reg} regression(s) > 3%, {imp} improvement(s) > 3%")
if a.get("contaminated") or b.get("contaminated"):
    print("note: at least one run is CONTAMINATED; treat the deltas as indicative only")
