#!/usr/bin/env python3
"""Kernel tier: nsys sqlite of the kbench reps -> kernel.json.

    kanalyze.py KBENCH.sqlite TSV_DIR OUT.json

NVTX ranges are "<side><rep>|<case>|<i>" (side L = llama.cpp, O = ours on the service path, A = our MMQ-MoE port that
the tiered path does not call). GPU time of a run = the kernels fully inside its range (activation quantize included).
Per (side, case): the median over all runs of all reps, each rep's median, and the spread between reps
(max-min)/median. Ratio = ours / llama.cpp."""
import collections, glob, json, re, sqlite3, statistics, sys

db, tsv_dir, out = sys.argv[1:4]
c = sqlite3.connect(db)
strings = dict(c.execute("select id, value from StringIds"))
cols = [r[1] for r in c.execute("pragma table_info(NVTX_EVENTS)")]
q = ("select start, end, text, textId from NVTX_EVENTS where eventType = 59" if "textId" in cols
     else "select start, end, text, null from NVTX_EVENTS where eventType = 59")
ranges = []
for s, e, t, tid in c.execute(q):
    t = t if t is not None else strings.get(tid)
    if t and t.count("|") == 2:
        sr, case, i = t.split("|")
        m = re.match(r"([LOA])(\d*)$", sr)
        if m:
            ranges.append((s, e, m.group(1), m.group(2) or "1", case, int(i)))
ranges.sort()
kern = sorted(c.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL"))

per_run = collections.defaultdict(float)
names = collections.defaultdict(lambda: collections.defaultdict(list))
j = 0
for s, e, side, rep, case, i in ranges:
    while j < len(kern) and kern[j][0] < s:
        j += 1
    k, tot = j, 0
    run_names = collections.defaultdict(float)
    while k < len(kern) and kern[k][0] < e:
        if kern[k][1] <= e:
            tot += kern[k][1] - kern[k][0]
            run_names[strings[kern[k][2]]] += kern[k][1] - kern[k][0]
        k += 1
    per_run[(side, rep, case, i)] = tot / 1e3
    for n, v in run_names.items():
        names[(side, case)][n].append(v / 1e3)

rows = {}
for f in glob.glob(f"{tsv_dir}/*.tsv"):
    for line in open(f):
        if line.startswith("RES\t"):
            _, sr, case, us, extra = line.rstrip("\n").split("\t", 4)
            m = re.match(r"([LOA])(\d*)$", sr)
            rows[(m.group(1), m.group(2) or "1", case)] = (float(us), extra)

by = collections.defaultdict(lambda: collections.defaultdict(list))  # (side, case) -> rep -> [us]
for (side, rep, case, i), us in per_run.items():
    by[(side, case)][rep].append(us)

res = {}
for (side, case), reps in by.items():
    allv = [v for r in reps.values() for v in r]
    med = statistics.median(allv)
    rmeds = {r: statistics.median(v) for r, v in sorted(reps.items())}
    spread = (max(rmeds.values()) - min(rmeds.values())) / med if len(rmeds) > 1 and med else None
    ev = [rows[(side, r, case)][0] for r in rmeds if (side, r, case) in rows]
    extra = next((rows[(side, r, case)][1] for r in rmeds if (side, r, case) in rows), "")
    d = {"us": round(med, 2), "rep_us": [round(v, 2) for v in rmeds.values()], "runs": len(allv),
         "spread": round(spread, 4) if spread is not None else None,
         "event_us": [round(v, 2) for v in ev],
         "top_kernels": {n: round(statistics.median(x), 2) for n, x in
                         sorted(names[(side, case)].items(), key=lambda t: -statistics.median(t[1]))[:3]}}
    eq = re.search(r"eq=(\d+)/(\d+)", extra)
    if eq:
        d["bit_equal_pct"] = round(100 * int(eq.group(1)) / max(1, int(eq.group(2))), 1)
    pm = re.search(r"path=(\S+)", extra)
    if pm:
        d["path"] = pm.group(1)
    res.setdefault(case, {})[{"L": "llama", "O": "ours", "A": "ours_mmq_moe"}[side]] = d
for case, d in res.items():
    if "llama" in d and "ours" in d and d["llama"]["us"]:
        d["ratio"] = round(d["ours"]["us"] / d["llama"]["us"], 3)
json.dump(res, open(out, "w"), indent=1, sort_keys=True)


def order(case):
    m = re.match(r"(.*)/b(\d+)$", case)
    return (m.group(1), int(m.group(2))) if m else (case, 0)


print(f"{'op':44s} {'b':>5s} {'llama us':>9s} {'ours us':>9s} {'ratio':>6s} {'spread':>7s}")
for case in sorted(res, key=order):
    op, b = order(case)
    d = res[case]
    f = lambda x: f"{x['us']:9.1f}" if x else f"{'-':>9s}"
    sp = max([x.get("spread") or 0 for x in d.values() if isinstance(x, dict)] or [0])
    print(f"{op:44s} {b:5d} {f(d.get('llama'))} {f(d.get('ours'))} {d.get('ratio', 0):6.2f} {100 * sp:6.1f}%")
