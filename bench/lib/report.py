#!/usr/bin/env python3
"""Assemble one bench run: bench/runs/<id>/*.json -> one line in bench/history.jsonl, bench/runs/<id>.md,
and a regenerated bench/history.html.

    report.py RUN_DIR"""
import json, os, re, statistics, subprocess, sys

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rd = os.path.abspath(sys.argv[1])
rid = os.path.basename(rd)


def load(n):
    p = f"{rd}/{n}"
    try:
        return json.load(open(p))
    except Exception:  # noqa: BLE001
        return None


meta = load("meta.json") or {}
hb, ha = load("hyg-before.json") or {}, load("hyg-after.json") or {}
kern = load("kernel.json") or {}
svc, off, ident = load("e2e-service.json") or {}, load("e2e-mtpoff.json") or {}, load("e2e-ident.json") or {}
n2, n0 = load("nsys-mtp2.json") or {}, load("nsys-off.json") or {}
adm = (load("e2e-admit.json") or {}).get("admit")

M = {}  # name -> {v, spread, better, unit, tier}


def spread(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return None
    m = statistics.median(vals)
    return (max(vals) - min(vals)) / m if m else None


def put(name, vals, better, unit, tier):
    vals = [v for v in (vals if isinstance(vals, list) else [vals]) if v is not None]
    if not vals:
        return
    M[name] = {"v": round(statistics.median(vals), 4), "reps": [round(v, 4) for v in vals], "spread": spread(vals),
               "better": better, "unit": unit, "tier": tier}


# ---- e2e ----
for tag, d in (("mtp2", svc), ("off", off)):
    if d.get("eight_1"):
        put(f"e2e.decode_tok_s.{tag}", [d[f"eight_{r}"]["decode_tok_s"] for r in (1, 2) if f"eight_{r}" in d], "higher", "tok/s", "e2e")
        put(f"e2e.decode_tok_s_server.{tag}", [d[f"eight_{r}"]["decode_tok_s_server_median"] for r in (1, 2) if f"eight_{r}" in d],
            "higher", "tok/s", "e2e")
        put(f"e2e.ttft_cold_ms.{tag}", d.get("ttft_cold_ms") or [], "lower", "ms", "e2e")
for key in ("4k", "13k"):
    rs = svc.get(f"cold_{key}") or []
    put(f"e2e.prompt_tok_s.{key}", [r["prompt_tok_s"] for r in rs], "higher", "tok/s", "e2e")
    put(f"e2e.ttft_s.{key}_cold", [r["ttft_s"] for r in rs], "lower", "s", "e2e")
rs = svc.get("warm_13k") or []
put("e2e.ttft_s.13k_warm", [r["ttft_s"] for r in rs], "lower", "s", "e2e")
identity = {
    "mtp2_vs_h-off": svc.get("eight_1", {}).get("identity"),
    "mtp2_rep2_vs_h-off": svc.get("eight_2", {}).get("identity"),
    "off_vs_h-off": off.get("eight_1", {}).get("identity"),
    "oldfile_off_vs_q35-prof": ident.get("eight_1", {}).get("identity"),
}
for k, d in (("mtp2", svc.get("eight_1", {})), ("off", off.get("eight_1", {})), ("oldfile", ident.get("eight_1", {}))):
    if "identity_nonstream_recheck" in d:
        identity[f"{k}_nonstream_recheck_of_mismatches"] = d["identity_nonstream_recheck"]
id_ok = all(v and v.split("/")[0] == v.split("/")[1] for k, v in identity.items() if "recheck" not in k and v)
id_missing = [k for k, v in identity.items() if v is None]

# ---- path tier ----
path = {}
for src, d, classes in (("off", n0, ("b1", "b512")), ("mtp2", n2, ("b3", "b512", "mtp_draft"))):
    for c in classes:
        p = (d.get("path") or {}).get(c)
        if not p:
            continue
        path[f"{c}@{src}"] = p
        put(f"path.{c}@{src}.span_us", p["rep_span_us"], "lower", "us", "path")
        put(f"path.{c}@{src}.experts_us", [p["cat_us"].get("experts")], "lower", "us", "path_info")
# llama.cpp's fused expert FFN at the same batch, for scale
for c, lb in (("b1@off", "1"), ("b3@mtp2", "3"), ("b512@off", "512")):
    L = (kern.get(f"q35.ffn_fused.q4_K+q5_K/b{lb}") or {}).get("llama")
    if L and c in path:
        path[c]["llama_fused_ffn_us"] = L["us"]
        path[c]["experts_vs_llama_ffn"] = round(path[c]["cat_us"].get("experts", 0) / L["us"], 2)

# ---- kernel tier ----
kbad = ktot = 0
for case, d in kern.items():
    for side in ("llama", "ours", "ours_mmq_moe"):
        if side in d:
            put(f"kernel.{case}.{side}_us", d[side]["rep_us"], "lower", "us", "kernel" if side == "ours" else "kernel_info")
            ktot += 1
            kbad += (d[side].get("spread") or 0) > 0.05
    if "ratio" in d:
        M[f"kernel.{case}.ratio"] = {"v": d["ratio"], "reps": [], "spread": None, "better": "lower", "unit": "x", "tier": "kernel"}

# ---- split ----
split = {"mtp2": n2.get("split"), "off": n0.get("split")}
for tag, s in split.items():
    for k, v in ((s or {}).get("pct") or {}).items():
        M[f"split.{tag}.{k}_pct"] = {"v": v, "reps": [r.get(k) for r in s.get("rep_pct", [])], "spread": None,
                                    "better": None, "unit": "%", "tier": "split"}

# ---- contamination ----
reasons = []
for when, h in (("before", hb), ("after", ha)):
    if h.get("load") and h["load"][0] > 6:
        reasons.append(f"load {h['load'][0]} > 6 ({when})")
    if h.get("non_titan_over_4g"):
        reasons.append(f"non-titan RSS > 4 GB ({when}): {', '.join(h['non_titan_over_4g'])}")
noisy = [n for n, m in M.items() if m["tier"] in ("e2e", "path") and (m["spread"] or 0) > 0.05]
if noisy:
    reasons.append("spread > 5%: " + ", ".join(f"{n} {100 * M[n]['spread']:.1f}%" for n in noisy))
if ktot and kbad / ktot > 0.10:
    reasons.append(f"kernel tier: {kbad}/{ktot} items with spread > 5%")
contaminated = bool(reasons)


def hsum(h):
    g = h.get("gpu", {})
    return {"load": h.get("load"), "mem_avail_mib": (h.get("mem_mib") or {}).get("available"),
            "gpu_sm_mhz": g.get("clocks.sm"), "gpu_mem_mhz": g.get("clocks.mem"), "gpu_temp_c": g.get("temperature.gpu"),
            "gpu_power_w": g.get("power.draw"), "gpu_mem_used_mib": g.get("memory.used"), "pstate": g.get("pstate"),
            "throttle": g.get("clocks_throttle_reasons.active"),
            "top_rss": [f"{p['comm']}({p['pid']}) {p['rss_mib']} MiB" for p in h.get("top_rss", [])]}


line = {"id": rid, "date": meta.get("date"), "label": meta.get("label"), "binary": meta.get("binary"),
        "binary_sha256": meta.get("binary_sha256"), "src": meta.get("src"), "revs": hb.get("revs"),
        "duration_s": meta.get("duration_s"), "steps_s": meta.get("steps_s"), "contaminated": contaminated,
        "reasons": reasons, "identity": identity, "identity_ok": id_ok and not id_missing, "identity_missing": id_missing,
        "hygiene": {"before": hsum(hb), "after": hsum(ha)}, "split": split, "path": path, "metrics": M,
        "e2e_detail": {"prompt_tokens": {k: [r.get("prompt_tokens") for r in svc.get(f"cold_{k}", [])] for k in ("4k", "13k")},
                       "warm13k_cached_tokens": [r.get("cached_tokens") for r in svc.get("warm_13k", [])],
                       "repeat_identical_mtp2": svc.get("repeat_identical"), "repeat_identical_off": off.get("repeat_identical")}}
if adm:
    line["admit"] = {k: adm.get(k) for k in ("pass", "refused_all", "refuse_ms_max", "after_ok")}
json.dump(line, open(f"{rd}/summary.json", "w"), indent=1)
hist = f"{B}/history.jsonl"
old = [l for l in open(hist)] if os.path.exists(hist) else []
old = [l for l in old if json.loads(l).get("id") != rid]  # re-running report.py replaces the run's line
with open(hist, "w") as f:
    f.writelines(old)
    f.write(json.dumps(line) + "\n")

# ---- markdown ----
f = lambda x, n=1: "-" if x is None else f"{x:.{n}f}"
sp = lambda m: "-" if m.get("spread") is None else f"{100 * m['spread']:.1f}%"
L = [f"# bench run {rid}", "",
     f"**{'CONTAMINATED' if contaminated else 'clean'}**" + (": " + "; ".join(reasons) if reasons else "") + "  ",
     f"**identity: {'OK' if line['identity_ok'] else 'FAIL'}** " + ", ".join(f"{k} {v}" for k, v in identity.items()), "",
     f"- binary: `{meta.get('binary')}` (sha256 {str(meta.get('binary_sha256'))[:16]})",
     f"- source: `{meta.get('src')}`; revs: " + ", ".join(f"{k} {v.get('rev')}{'+dirty' if v.get('dirty_files') else ''}"
                                                         for k, v in (hb.get("revs") or {}).items()),
     f"- duration {f(meta.get('duration_s'), 0)} s; steps: " + ", ".join(f"{k} {v:.0f}s" for k, v in (meta.get("steps_s") or {}).items()),
     "", "## E2E (service config, private port)", "", "| metric | value | reps | spread |", "|---|---:|---|---:|"]
for n, m in M.items():
    if m["tier"] == "e2e":
        L.append(f"| {n[4:]} ({m['unit']}) | {f(m['v'], 2)} | {', '.join(f(x, 2) for x in m['reps'])} | {sp(m)} |")
L += ["", f"Prompt tokens: {line['e2e_detail']['prompt_tokens']}; warm 13k cached tokens {line['e2e_detail']['warm13k_cached_tokens']}; "
      f"8x256 rep2 identical to rep1: MTP=2 {svc.get('repeat_identical')}, off {off.get('repeat_identical')}.", "",
      *([f"Admission probe (TITAN_ADMIT=1): **{'PASS' if adm.get('pass') else 'FAIL'}**, over-context refused "
         f"{adm.get('refused_all')} in at most {f(adm.get('refuse_ms_max'), 0)} ms, next request ok {adm.get('after_ok')}, "
         f"28k prompt {(adm.get('long_28k') or {}).get('prompt_tokens')} tokens.", ""] if adm else []),
      "## Path tier: one layer's MoE forward as the model runs it (nsys, µs, median over layers)", "",
      "| batch@profile | blocks | span | busy | idle in span | routed experts | shared expert | other | rep spans | spread | llama fused FFN | experts / llama FFN |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
for c, p in path.items():
    L.append(f"| {c} | {p['blocks']} | {f(p['span_us'])} | {f(p['busy_us'])} | {f(p['idle_us'])} | {f(p['cat_us'].get('experts'))} | "
             f"{f(p['cat_us'].get('shared_expert'))} | {f(p['cat_us'].get('other'))} | {', '.join(f(x) for x in p['rep_span_us'])} | "
             f"{sp(p)} | "
             f"{f(p.get('llama_fused_ffn_us'))} | {f(p.get('experts_vs_llama_ffn'), 2)} |")
L += ["", "## GPU time split of the decode window (nsys, ~32 tokens after a 2.1k prompt, 2 reps)", "",
      "| profile | window ms | experts | attention/GDN | dense | lm_head | copies | other | idle/gaps | H2D ms |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
tops = []
for tag, s in split.items():
    if s:
        p = s["pct"]
        L.append(f"| {tag} | {f(s['window_ms'])} | {f(p['experts'])}% | {f(p['attention/GDN'])}% | {f(p['dense'])}% | {f(p['lm_head'])}% | "
                 f"{f(p.get('copies'))}% | {f(p['other'])}% | {f(p['idle/gaps'])}% | {f(s['h2d_ms'])} |")
        tops.append(f"- top decode-window kernels ({tag}, ms per rep): " + "; ".join(f"{k} {v}" for k, v in list((s.get('top_kernels_ms_per_rep') or {}).items())[:10]))
L += [""] + tops
L += ["", "## Kernel tier (µs per op, GPU kernel time under nsys, L2 flushed, median of 2 x 50 runs)", "",
      "| op | b | llama.cpp | ours | ours/llama | spread (ours) | ours MMQ-MoE (not on service path) | bit-equal |",
      "|---|---:|---:|---:|---:|---:|---:|---:|"]


def order(case):
    m = re.match(r"(.*)/b(\d+)$", case)
    return (m.group(1), int(m.group(2))) if m else (case, 0)


for case in sorted(kern, key=order):
    d = kern[case]
    op, b = order(case)
    o, l, a = d.get("ours"), d.get("llama"), d.get("ours_mmq_moe")
    r = d.get("ratio")
    rs = "-" if r is None else (f"**{r:.2f}**" if r >= 1.25 else f"{r:.2f}")
    L.append(f"| {op} | {b} | {f(l and l['us'])} | {f(o and o['us'])} | {rs} | {sp(o or {})} | "
             f"{f(a and a['us'])} | {'-' if not o or 'bit_equal_pct' not in o else str(o['bit_equal_pct']) + '%'} |")
L += ["", "## Hygiene", "", "| | load 1/5/15 | MemAvailable MiB | SM MHz | mem MHz | temp C | power W | GPU mem MiB | top RSS |",
      "|---|---|---:|---:|---:|---:|---:|---:|---|"]
for when, h in (("before", line["hygiene"]["before"]), ("after", line["hygiene"]["after"])):
    L.append(f"| {when} | {h['load']} | {h['mem_avail_mib']} | {h['gpu_sm_mhz']} | {h['gpu_mem_mhz']} | {h['gpu_temp_c']} | "
             f"{h['gpu_power_w']} | {h['gpu_mem_used_mib']} | {'; '.join(h['top_rss'])} |")
L += ["", f"Raw files: `bench/runs/{rid}/`. Compare: `python3 bench/compare.py <other-run> {rid}`.", ""]
open(f"{B}/runs/{rid}.md", "w").write("\n".join(L))
subprocess.run([sys.executable, f"{B}/lib/mkhtml.py"], check=False)
print(f"{rid}: {'CONTAMINATED' if contaminated else 'clean'}; identity {'OK' if line['identity_ok'] else 'FAIL'} {identity}")
for r in reasons:
    print("  -", r)
