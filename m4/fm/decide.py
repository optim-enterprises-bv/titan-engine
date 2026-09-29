#!/usr/bin/env python3
"""Deploy decision for titan-094 fm integration (flash-decode + mmvq-moe, f5458a45c). usage: decide.py OUT.json CANDIDATE_BIN
Criteria (from the task):
  - identity OK: 8/8 x4 in the "fm" bench run, and 40/40 in both m4/fm gates (MTP off vs q35-prof, MTP=2 vs h-off)
  - short-context decode >= mean(pre-fm, pre-fm-2) - 1%, both MTP=2 and MTP off
  - 13k prefill within 2% of the pre-fm mean
  - 27.7k decode improved (candidate > deployed)
"""
import json, os, sys

out_path, cand_bin = sys.argv[1], sys.argv[2]
E = os.path.expanduser("~/titan-engine")
HIST = f"{E}/bench/history.jsonl"
FM_OUT = f"{E}/m4/fm/out"

reasons = []
ok = True


def latest(label):
    if not os.path.exists(HIST):
        return None
    runs = [json.loads(l) for l in open(HIST)]
    m = [r for r in runs if r.get("label") == label]
    return m[-1] if m else None


def mv(metrics, key):
    m = metrics.get(key)
    if isinstance(m, dict):
        return m.get("v")
    return m


pre1 = latest("pre-fm")
fm = latest("fm")
pre2 = latest("pre-fm-2")

for name, r in (("pre-fm", pre1), ("fm", fm), ("pre-fm-2", pre2)):
    if r is None:
        ok = False
        reasons.append(f"no '{name}' bench run found in history.jsonl")

if fm is not None:
    if fm.get("contaminated"):
        reasons.append(f"NOTE: {fm['id']} (fm) is flagged CONTAMINATED ({fm.get('reasons')})")
    if not fm.get("identity_ok", False) or fm.get("identity_missing"):
        ok = False
        reasons.append(f"bench identity NOT ok on fm run: identity_ok={fm.get('identity_ok')} missing={fm.get('identity_missing')}")
    else:
        reasons.append("bench identity OK (8/8 x4) on fm run")

# ---- 40-prompt gates ----
for gname, gfile in (("MTP=2 vs h-off", f"{FM_OUT}/gate-m2.json"), ("MTP off vs q35-prof", f"{FM_OUT}/gate-m0.json")):
    if not os.path.exists(gfile):
        ok = False
        reasons.append(f"gate missing: {gname} ({gfile})")
        continue
    g = json.load(open(gfile))
    passed, total = g.get("pass"), g.get("total")
    good = passed == total and total == 40
    reasons.append(f"gate {gname}: {passed}/{total} -> {'OK' if good else 'FAIL'}")
    if not good:
        ok = False

# ---- short-context decode + 13k prefill ----
if pre1 is not None and fm is not None and pre2 is not None:
    for metric, label in (("e2e.decode_tok_s.mtp2", "decode MTP=2"), ("e2e.decode_tok_s.off", "decode MTP off")):
        v1 = mv(pre1.get("metrics", {}), metric)
        v2 = mv(pre2.get("metrics", {}), metric)
        vf = mv(fm.get("metrics", {}), metric)
        if v1 is None or v2 is None or vf is None:
            ok = False
            reasons.append(f"{label}: missing metric (pre1={v1} fm={vf} pre2={v2})")
            continue
        ref = (v1 + v2) / 2
        delta = (vf - ref) / ref * 100
        within = delta >= -1.0
        reasons.append(f"{label}: fm={vf:.1f} vs mean(pre-fm,pre-fm-2)={ref:.1f} ({delta:+.2f}%) -> {'OK' if within else 'REGRESSION'} (need >= -1%)")
        if not within:
            ok = False

    metric = "e2e.prompt_tok_s.13k"
    v1 = mv(pre1.get("metrics", {}), metric)
    v2 = mv(pre2.get("metrics", {}), metric)
    vf = mv(fm.get("metrics", {}), metric)
    if v1 is None or v2 is None or vf is None:
        ok = False
        reasons.append(f"13k prefill: missing metric (pre1={v1} fm={vf} pre2={v2})")
    else:
        ref = (v1 + v2) / 2
        delta = (vf - ref) / ref * 100
        within = abs(delta) <= 2.0
        reasons.append(f"13k prefill: fm={vf:.1f} vs mean(pre-fm,pre-fm-2)={ref:.1f} ({delta:+.2f}%) -> {'OK' if within else 'REGRESSION'} (need within 2%)")
        if not within:
            ok = False
else:
    ok = False
    reasons.append("13k prefill / short-context decode checks skipped: missing pre-fm/fm/pre-fm-2 bench runs")

# ---- 27.7k long-context decode, candidate vs deployed ----
lc_cand = f"{FM_OUT}/longctx-candidate.jsonl"
lc_dep = f"{FM_OUT}/longctx-deployed.jsonl"


def read_tag(path, tag):
    if not os.path.exists(path):
        return None
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("tag") == tag:
            return r
    return None


c28 = read_tag(lc_cand, "28k-candidate")
d28 = read_tag(lc_dep, "28k-deployed")
c13 = read_tag(lc_cand, "13k-candidate")
d13 = read_tag(lc_dep, "13k-deployed")

if c28 is None or d28 is None:
    ok = False
    reasons.append(f"27.7k long-context decode: missing record (candidate={c28} deployed={d28})")
else:
    cv, dv = c28.get("decode_tok_s"), d28.get("decode_tok_s")
    improved = cv is not None and dv is not None and cv > dv
    delta = (cv - dv) / dv * 100 if (cv and dv) else float("nan")
    reasons.append(f"27.7k decode tok/s: candidate={cv:.1f} vs deployed={dv:.1f} ({delta:+.1f}%) -> {'IMPROVED' if improved else 'NOT IMPROVED'}")
    if not improved:
        ok = False

if c13 is not None and d13 is not None:
    cv, dv = c13.get("decode_tok_s"), d13.get("decode_tok_s")
    if cv and dv:
        reasons.append(f"13k decode tok/s (long-context script): candidate={cv:.1f} vs deployed={dv:.1f} ({(cv-dv)/dv*100:+.1f}%)")

result = {"deploy": ok, "reasons": reasons, "pre_fm": pre1, "fm": fm, "pre_fm_2": pre2,
          "longctx_candidate": {"13k": c13, "28k": c28}, "longctx_deployed": {"13k": d13, "28k": d28},
          "candidate_bin": cand_bin}
json.dump(result, open(out_path, "w"), indent=1)
print(json.dumps({"deploy": ok, "reasons": reasons}, indent=1))
