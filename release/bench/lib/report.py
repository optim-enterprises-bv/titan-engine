#!/usr/bin/env python3
"""Build results.md (tables) and results.json (combined, for the web page and the claims tables) from
results/*.json. usage: report.py [ROOT]   (ROOT: release/bench, or release/bench/dryrun for the rehearsal)

Every number in results.json's "claims" points at the item file and the JSON path it was read from, so a draft's
claims table can cite "claim id -> results/<item>.json : <path>"."""
import glob, json, os, statistics, sys, time

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = json.load(open(os.path.join(B, "campaign.json")))


def jl(p, d=None):
    try:
        return json.load(open(p))
    except Exception:  # noqa: BLE001
        return d


def fmt(v, nd=1):
    if v is None:
        return "-"
    return f"{v:,.{nd}f}" if abs(v) < 1e5 else f"{v:,.0f}"


def cell(summ, key, nd=1, scale=1.0):
    s = (summ or {}).get(key)
    if not s or s.get("median") is None:
        return "-"
    sp = s.get("spread")
    return f"{fmt(s['median'] * scale, nd)}" + (f" ±{100 * sp / 2:.1f}%" if sp is not None else "")


def main(root=B):
    res_dir = os.path.join(root, "results")
    R = {os.path.basename(p)[:-5]: jl(p) for p in sorted(glob.glob(os.path.join(res_dir, "*.json")))}
    R = {k: v for k, v in R.items() if v}
    synthetic = any(v.get("synthetic") for v in R.values())
    pin = jl(os.path.join(root, "state", "pinned.json"), {})
    claims, models_out = {}, []
    md = []
    title = "titan-engine vs llama.cpp: release benchmark campaign"
    if synthetic:
        md += ["# SYNTHETIC DRY RUN - NOT BENCHMARK RESULTS", "",
               "Produced by fake engines (lib/fake_server.py) to test parsing and tables. No number here is a measurement.", ""]
    md += [f"# {title}", "", f"Generated {time.strftime('%Y-%m-%d %H:%M %z')} from `{os.path.relpath(res_dir, B)}/` "
           f"({len(R)} item files).", ""]
    md += ["## Setup", "",
           f"- machine: {pin.get('cpu', '?')}; GPU {pin.get('gpu', '?')}; {pin.get('mem_total_mib', '?')} MiB RAM; kernel {pin.get('kernel', '?')}",
           f"- ours: `{pin.get('ours_binary', '?')}` sha256 {str(pin.get('ours_sha256'))[:16]}; mistral.rs {pin.get('mistral_rs_rev')}, "
           f"oxide-kernels {pin.get('oxide_kernels_rev')}, titan-engine {pin.get('titan_engine_rev')}",
           f"- llama.cpp: `{pin.get('llama_server', '?')}` at {pin.get('llama_rev')}",
           f"- every server: its own transient unit, MemoryMax={CFG['memory_max']}, no swap, page cache of the GGUF dropped "
           f"before start; one server at a time; greedy decoding; single user (one request at a time)",
           f"- every timed measurement run {CFG['reps']}x, plus a third run when the two differ by more than "
           f"{100 * CFG['spread_gate']:.0f}% (the gate then asks that two of the three agree within it); cells show the "
           f"median ±half the (max-min)/median spread of all runs; an item is redone (up to {CFG['max_redo']}x) when "
           f"the gate fails, the "
           f"1-minute load before a fresh server exceeds {CFG['load_gate']}, or other processes use more than "
           f"{CFG['foreign_cpu_gate_cores']} CPU core on average during it", ""]
    head_rows = []
    for m in CFG["models"]:
        mid = m["id"]
        its = {k: v for k, v in R.items() if v.get("model") == mid}
        if not its:
            continue
        rows = {}
        for k, v in its.items():
            if v["kind"] in ("short", "long", "quality"):
                rows.setdefault((v["engine"], v["variant"]), {})[v["kind"] if v["kind"] != "long" else "L" + v["length"]] = (k, v)
        if not rows:
            continue
        mo = {"id": mid, "name": m["name"], "ctx": m["ctx"], "rows": [], "headline": {}}
        md += [f"## {m['name']}", "", f"GGUF `{os.path.basename(m['gguf'])}`, context {m['ctx']}; short decode = "
               f"{m['short'][0]} prompts x {m['short'][1]} tokens; long prompts decode {m['long_decode']} tokens.", ""]
        lens = [k for k in ("4k", "13k", "28k") if any("L" + k in r for r in rows.values())]
        hdr = ["engine", "config", "decode tok/s", "TTFT short ms"]
        for k in lens:
            hdr += [f"prompt tok/s {k}", f"TTFT {k} s", f"decode @{k}"]
        hdr += ["peak VRAM MiB", "peak RAM MiB (cgroup / RSS)", "q100 / q40", "status"]
        md += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        best = {}
        for (eng, var), r in sorted(rows.items()):
            out = {"engine": eng, "variant": var}
            status, vram, cg, rss = [], [], [], []
            sh = r.get("short")
            cells = [eng, var]
            if sh:
                s = sh[1]["result"]["summary"]
                cells += [cell(s, "decode_tok_s"), cell(s, "ttft_short_ms", 0)]
                out["decode_tok_s"] = s["decode_tok_s"]["median"]
                out["ttft_short_ms"] = s["ttft_short_ms"]["median"]
                claims[f"{mid}.{eng}.{var}.decode_tok_s"] = {"value": out["decode_tok_s"], "file": f"results/{sh[0]}.json",
                                                             "path": "result.summary.decode_tok_s.median"}
                claims[f"{mid}.{eng}.{var}.ttft_short_ms"] = {"value": out["ttft_short_ms"], "file": f"results/{sh[0]}.json",
                                                              "path": "result.summary.ttft_short_ms.median"}
            else:
                cells += ["-", "-"]
            for k in lens:
                lr = r.get("L" + k)
                if lr:
                    s = lr[1]["result"]["summary"]
                    cells += [cell(s, "prompt_tok_s", 0), cell(s, "ttft_s", 2), cell(s, "decode_tok_s")]
                    for met in ("prompt_tok_s", "ttft_s", "decode_tok_s"):
                        out[f"{met}@{k}"] = s[met]["median"]
                        claims[f"{mid}.{eng}.{var}.{met}@{k}"] = {"value": s[met]["median"], "file": f"results/{lr[0]}.json",
                                                                  "path": f"result.summary.{met}.median"}
                    pts = [x for x in lr[1]["result"]["raw"].get("prompt_tokens", []) if x]
                    out[f"prompt_tokens@{k}"] = statistics.median(pts) if pts else None
                else:
                    cells += ["-", "-", "-"]
            for kk, (iid, v) in r.items():
                if kk == "quality":
                    continue
                mon = v["result"].get("monitor", {})
                vram.append(mon.get("peak_vram_mib") or 0)
                cg.append(v["result"].get("cgroup_memory_peak_mib") or mon.get("peak_cgroup_mib") or 0)
                rss.append(mon.get("peak_rss_mib") or 0)
                if v.get("status") != "ok":
                    status.append(f"{kk} unstable")
            out["peak_vram_mib"] = max(vram) if vram else None
            out["peak_cgroup_mib"] = max(cg) if cg else None
            out["peak_rss_mib"] = max(rss) if rss else None
            q = [v for k2, v in its.items() if v["kind"] == "quality" and v["engine"] == eng and v["variant"] == var]
            if q:
                score, n = sum(v["result"]["score"] for v in q), sum(v["result"]["n"] for v in q)
                out["q100"] = f"{score}/{n}" + (" (q40)" if q[0]["result"].get("set") == "q40" else "")
                claims[f"{mid}.{eng}.{var}.q100"] = {"value": out["q100"], "file": ", ".join(
                    f"results/{k2}.json" for k2, v in its.items() if v["kind"] == "quality" and v["engine"] == eng and v["variant"] == var),
                    "path": "result.score / result.n (summed over parts)"}
            cells += [fmt(out["peak_vram_mib"], 0), f"{fmt(out['peak_cgroup_mib'], 0)} / {fmt(out['peak_rss_mib'], 0)}",
                      out.get("q100", "-"), ", ".join(status) or "ok"]
            md.append("| " + " | ".join(cells) + " |")
            argv_src = (sh or next(iter(r.values())))[1]["server"]
            out["argv"], out["env"] = argv_src["argv"], argv_src["env"]
            mo["rows"].append(out)
            if out.get("decode_tok_s") and (eng not in best or out["decode_tok_s"] > best[eng]["decode_tok_s"]):
                best[eng] = out
        md.append("")
        mo["headline"] = {e: {"variant": b["variant"]} for e, b in best.items()}
        for e, b in best.items():  # stable aliases for the drafts: <model>.<engine>.head.<metric>
            pre = f"{mid}.{e}.{b['variant']}."
            for k in [k for k in claims if k.startswith(pre)]:
                claims[f"{mid}.{e}.head." + k[len(pre):]] = dict(claims[k], alias_of=k)
        if "ours" in best and "llama" in best:
            o, l = best["ours"], best["llama"]
            rat = {"decode": o["decode_tok_s"] / l["decode_tok_s"]}
            for k in lens:
                for met in ("prompt_tok_s", "decode_tok_s"):
                    a, b = o.get(f"{met}@{k}"), l.get(f"{met}@{k}")
                    if a and b:
                        rat[f"{met}@{k}"] = a / b
            mo["ratios"] = rat
            for k, v in rat.items():
                claims[f"{mid}.ratio.{k}"] = {"value": v, "file": "derived", "path":
                                              f"{mid}.ours.{o['variant']} / {mid}.llama.{l['variant']} ({k})"}
            md += [f"Headline rule (pre-declared): each engine's configuration with the fastest short decode. "
                   f"Ours `{o['variant']}` vs llama.cpp `{l['variant']}`: decode x{rat['decode']:.2f}"
                   + "".join(f", {k.replace('_tok_s', '')} x{v:.2f}" for k, v in rat.items() if k != "decode") + ".", ""]
            head_rows.append((m["name"], o, l, rat))
        md += ["Exact command lines:", ""]
        for out in mo["rows"]:
            env = " ".join(f"{k}={v}" for k, v in sorted(out["env"].items()))
            md += [f"- {out['engine']} `{out['variant']}`:", "", "  ```", f"  {env} \\", "  " + " ".join(out["argv"]), "  ```"]
        md.append("")
        tl = its.get(f"tune.llama.{mid}")
        if tl:
            md += [f"llama.cpp sweep ({tl['rule']}) chose `{json.dumps(tl['chosen'])}`; MTP helps: {tl.get('mtp_helps')}.", "",
                   "| config | fits longest prompt | prompt tok/s (4k) | decode tok/s (8x64) | peak VRAM MiB |", "|---|---|---|---|---|"]
            fits = {json.dumps(p["t"], sort_keys=True): p for p in tl.get("fit_probes", [])}
            for p in sorted(fits.values(), key=lambda p: json.dumps(p["t"], sort_keys=True)):
                md.append(f"| `{json.dumps(p['t'], sort_keys=True)}` | {'yes' if p.get('ok') else 'no'} | - | - | {fmt(p.get('peak_vram_mib'), 0)} |")
            for p in tl.get("measured", []):
                md.append(f"| `{json.dumps(p['t'], sort_keys=True)}` | (measured) | {fmt(p.get('prompt_tok_s'), 0)} | "
                          f"{fmt(p.get('decode_tok_s'))} | {fmt(p.get('peak_vram_mib'), 0)} |")
            md.append("")
        to = its.get(f"tune.ours.{mid}")
        if to:
            md += [f"Ours: candidate configs ({to['rule']}) chose `{to['chosen']}`.", "",
                   "| candidate | ran | prompt tok/s (4k) | decode tok/s (8x64) |", "|---|---|---|---|"]
            for p in to["measured"]:
                md.append(f"| {p['candidate']} | {'yes' if p.get('ok') else 'no: ' + p.get('error', '')[:80]} | "
                          f"{fmt(p.get('prompt_tok_s'), 0)} | {fmt(p.get('decode_tok_s'))} |")
            md.append("")
        models_out.append(mo)
    if head_rows:
        summ = ["## Headline", "", "| model | ours decode tok/s | llama.cpp decode tok/s | decode ratio | "
                "prompt tok/s ratio (13k) | decode ratio @13k |", "|---|---|---|---|---|---|"]
        for name, o, l, rat in head_rows:
            summ.append(f"| {name} | {fmt(o['decode_tok_s'])} ({o['variant']}) | {fmt(l['decode_tok_s'])} ({l['variant']}) | "
                        f"x{rat['decode']:.2f} | {('x%.2f' % rat['prompt_tok_s@13k']) if 'prompt_tok_s@13k' in rat else '-'} | "
                        f"{('x%.2f' % rat['decode_tok_s@13k']) if 'decode_tok_s@13k' in rat else '-'} |")
        i = md.index("## Setup")
        md[i:i] = summ + [""]
    dur = [v.get("duration_s", 0) for v in R.values()]
    if dur:
        claims["campaign.hours"] = {"value": round(sum(dur) / 3600 + 0.05, 1), "file": "results/*.json",
                                    "path": "sum of duration_s over all items"}
    # hygiene
    redone = [(k, v) for k, v in R.items() if v.get("discarded_attempts")]
    unstable = [k for k, v in R.items() if v.get("status") == "unstable"]
    loads = [v["result"]["hygiene"]["before"]["load"][0] for v in R.values() if v.get("kind") in ("short", "long")]
    md += ["## Hygiene", "",
           f"- items redone: {len(redone)}; still unstable after the redo limit: {len(unstable)} {unstable if unstable else ''}",
           f"- 1-minute load at item start: min {fmt(min(loads)) if loads else '-'}, max {fmt(max(loads)) if loads else '-'}", ""]
    for k, v in redone:
        md.append(f"  - {k}: discarded attempts: " + "; ".join(", ".join(a["contaminated"]) for a in v["discarded_attempts"]))
    fails = jl(os.path.join(root, "state", "failures.json"), {})
    if fails:
        md += ["", "Failed items (not in the tables):", ""] + [f"- {k}: {v[-1]['error'][:300]}" for k, v in fails.items()]
    md += ["", "## Definitions", "",
           "- decode tok/s: sum(completion tokens - 1) / sum(last chunk - first chunk), client side, streamed; prefill excluded.",
           "- TTFT short: send to end of a max_tokens=1 reply to one of the 8 short prompts with a fresh nonce (no prefix "
           "cache hit), median of 8.",
           "- prompt tok/s at k: prompt tokens (server-reported) / client TTFT for a cold prompt of about k tokens "
           "(a fixed system text with a fresh nonce at its start, max_tokens 1); TTFT k is that TTFT.",
           "- decode @k: the same fixed ~k-token prompt in every repetition (so the greedy reply, the expert routing "
           "and the draft acceptance are identical across repetitions); its prefill may be prefix-cached, which is "
           "outside the decode rate. Prompt tok/s and TTFT come from separate cold prompts with a fresh nonce.",
           "- peak VRAM: nvidia-smi memory.used sampled every 0.5 s (the card runs nothing else); peak RAM: the server "
           "unit's cgroup memory.peak (includes page cache of mmapped weights) / the process's peak RSS.",
           "- q100: 100 short-answer questions (m4/skip/q100.json), greedy, raw prompt in the model's chat format, regex scoring. "
           "q40: for models under ~10 tok/s (gpt-oss-120b; the 80B if llama.cpp decodes it under 10 tok/s), a fixed "
           "40-question subset of q100 (seeded sample, seed 20260929, lib/quality.py q40()), the same on both engines.",
           ""]
    open(os.path.join(root, "results.md"), "w").write("\n".join(md))
    json.dump({"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "synthetic": synthetic, "provenance":
               {k: v for k, v in pin.items() if k != "unit"}, "models": models_out, "claims": claims},
              open(os.path.join(root, "results.json"), "w"), indent=1)
    print(f"wrote {os.path.join(root, 'results.md')} and results.json ({len(claims)} claims)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else B)
