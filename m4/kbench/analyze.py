#!/usr/bin/env python3
"""kbench analysis: nsys sqlite -> per (side, case) median GPU kernel time per run, merged with the
harness's stdout rows (CUDA-event medians, bit-equality), written as results.json + a markdown table.

    python3 analyze.py [out/kbench.sqlite]
"""
import collections, glob, json, os, re, sqlite3, statistics, sys

K = os.path.dirname(os.path.abspath(__file__))
DB = sys.argv[1] if len(sys.argv) > 1 else f"{K}/out/kbench.sqlite"

c = sqlite3.connect(DB)
strings = dict(c.execute("select id, value from StringIds"))
cols = [r[1] for r in c.execute("pragma table_info(NVTX_EVENTS)")]
q = "select start, end, text, textId from NVTX_EVENTS where eventType = 59" if "textId" in cols else \
    "select start, end, text, null from NVTX_EVENTS where eventType = 59"
ranges = []
for s, e, t, tid in c.execute(q):
    t = t if t is not None else strings.get(tid)
    if t and t.count("|") == 2:
        side, case, i = t.split("|")
        ranges.append((s, e, side, case, int(i)))
ranges.sort()
kern = sorted(c.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL"))

# sweep: kernels fully inside a range
per_run = collections.defaultdict(lambda: collections.defaultdict(float))  # (side,case,i) -> name -> ns
j = 0
for s, e, side, case, i in ranges:
    while j < len(kern) and kern[j][0] < s:
        j += 1
    k = j
    while k < len(kern) and kern[k][0] < e:
        if kern[k][1] <= e:
            per_run[(side, case, i)][strings[kern[k][2]]] += kern[k][1] - kern[k][0]
        k += 1

agg = collections.defaultdict(list)       # (side,case) -> [total_us per run]
names = collections.defaultdict(lambda: collections.defaultdict(list))
for (side, case, i), d in per_run.items():
    agg[(side, case)].append(sum(d.values()) / 1e3)
    for n, v in d.items():
        names[(side, case)][n].append(v / 1e3)

# harness rows
rows = {}
for f in glob.glob(f"{K}/out/*.tsv"):
    for line in open(f):
        if line.startswith("RES\t"):
            _, side, case, us, extra = line.rstrip("\n").split("\t", 4)
            rows[(side, case)] = (float(us), extra)

res = {}
for (side, case), v in agg.items():
    nm = names[(side, case)]
    res.setdefault(case, {})[side] = {
        "nsys_us": statistics.median(v), "runs": len(v),
        "kernels": {n: round(statistics.median(x), 2) for n, x in sorted(nm.items(), key=lambda t: -statistics.median(t[1]))},
        "harness_us": rows.get((side, case), (None, ""))[0],
        "extra": rows.get((side, case), (None, ""))[1],
    }
for (side, case), (us, extra) in rows.items():
    if side not in res.get(case, {}):
        res.setdefault(case, {})[side] = {"nsys_us": None, "runs": 0, "kernels": {}, "harness_us": us, "extra": extra}
json.dump(res, open(f"{K}/out/results.json", "w"), indent=1)


def order(case):
    m = re.match(r"(.*)/b(\d+)$", case)
    return (m.group(1), int(m.group(2))) if m else (case, 0)


def short(n):
    return re.sub(r"<.*", "", n)[:40]


print("| op | batch | llama.cpp us | ours us | ours/llama | ours alt (MMQ-MoE) us | bit-equal | llama kernels | our kernels |")
print("|---|---:|---:|---:|---:|---:|---|---|---|")
for case in sorted(res, key=order):
    op, b = order(case)
    d = res[case]
    L, O, A = d.get("L"), d.get("O"), d.get("A")
    lu = L["nsys_us"] if L else None
    ou = O["nsys_us"] if O else None
    au = A["nsys_us"] if A else None
    ratio = f"{ou / lu:.2f}" if lu and ou else "-"
    eq = re.search(r"eq=(\d+)/(\d+)", O["extra"]) if O else None
    eqs = f"{100 * int(eq.group(1)) / max(1, int(eq.group(2))):.0f}%" if eq else "-"
    lk = ", ".join(f"{short(n)} {v:.1f}" for n, v in list(L["kernels"].items())[:3]) if L else ""
    ok = ", ".join(f"{short(n)} {v:.1f}" for n, v in list(O["kernels"].items())[:3]) if O else ""
    f = lambda x: f"{x:.1f}" if x is not None else "-"
    print(f"| {op} | {b} | {f(lu)} | {f(ou)} | {ratio} | {f(au)} | {eqs} | {lk} | {ok} |")
