#!/usr/bin/env python3
"""Gate verdicts for a window's outputs: check.py PREFIX SERVERLOG -> prints PASS/FAIL lines, exit 0 iff all pass."""
import json, re, sys
pre, log = sys.argv[1], sys.argv[2]
ok = True
def v(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'} {name} {detail}")
def j(n):
    try:
        return json.load(open(f"out/{pre}-{n}.json"))
    except Exception as e:
        return None
fr, af, m0 = j("fresh"), j("after"), j("mtp0")
if fr and af and m0:
    for n, r in (("fresh", fr), ("after", af)):
        for rep in (1, 2):
            v(f"(d) identity MTP=2 vs h-off {n} rep{rep}", r[f"eight_{rep}"]["identity"] == "8/8", r[f"eight_{rep}"]["identity"])
    for rep in (1, 2):
        v(f"(d) identity MTP off vs h-off rep{rep} (MTP=2 == off)", m0[f"eight_{rep}"]["identity"] == "8/8", m0[f"eight_{rep}"]["identity"])
    d = lambda r: sum(r[f"eight_{k}"]["decode_tok_s"] for k in (1, 2)) / 2
    p = lambda r: sum(x["prompt_tok_s"] for x in r["cold_13k"]) / len(r["cold_13k"])
    v("(d) decode after cycle within 3% of fresh", d(af) >= 0.97 * d(fr), f"{d(fr):.1f} -> {d(af):.1f} tok/s")
    v("(d) 13k prompt after cycle within 4% of fresh", p(af) >= 0.96 * p(fr), f"{p(fr):.0f} -> {p(af):.0f} tok/s")
else:
    v("(d) bench outputs present", False)
for n in ("cycle1", "cycle2", "mx"):
    r = j(n)
    v(f"(b) {n} all answered correctly", r and all(x.get("correct") for x in r), [(x["step"], round(x["wall"], 1)) for x in r or []])
q = j("queue20")
v("(e) 4 queued requests during the 20b swap", q and all(x.get("correct") for x in q), [round(x["wall"], 1) for x in q or []])
o = j("oom")
v("OOM: 13k after the failed 60k succeeds", o and o[2]["ok"] and o[3]["ok"], [(x["step"], x["ok"], round(x.get("prompt_tok_s") or 0)) for x in o or []])
t = open(log, errors="replace").read()
plans = re.findall(r"titan tiered auto: (\d+) MiB free, 2543 MiB other weights, 2880 MiB reserve, 18726 MiB tierable experts -> GPU fraction ([0-9.]+)", t)
v("(c) 35B plan identical on every load", len(set(plans)) == 1, plans)
frees = [int(x) for x in re.findall(r"titan state released, pools trimmed\]: (\d+) MiB free", t)]
base = re.findall(r"titan swap: (\d+) MiB VRAM free before any model", t)
v("(c) free VRAM after each unload stable (spread <= 64 MiB)", frees and max(frees) - min(frees) <= 64, f"baseline {base} after unloads {frees}")
rss = [int(x) for x in re.findall(r"titan swap: unloaded .* host RSS (\d+) MiB", t)]
v("(c) host RSS after each unload does not grow (spread <= 256 MiB)", rss and max(rss) - min(rss) <= 256, rss)
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
