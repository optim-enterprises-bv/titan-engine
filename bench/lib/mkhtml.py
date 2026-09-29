#!/usr/bin/env python3
"""bench/history.jsonl -> bench/history.html: a self-contained static page (inline SVG + CSS, no scripts, no CDN).
One chart per metric family (one unit per chart), runs in order on x. Hollow markers = CONTAMINATED run.
Hover a marker or bar for its value (SVG <title>). A table at the bottom lists every run."""
import html, json, os

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
runs = [json.loads(l) for l in open(f"{B}/history.jsonl")] if os.path.exists(f"{B}/history.jsonl") else []

W, H, PL, PR, PT, PB = 520, 210, 52, 96, 14, 34
SERIES = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"]


def mv(r, name):
    m = r.get("metrics", {}).get(name)
    return m["v"] if m else None


def nice_max(v):
    if v <= 0:
        return 1
    import math
    e = 10 ** math.floor(math.log10(v))
    for k in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if k * e >= v:
            return k * e
    return 10 * e


def line_chart(title, unit, series):
    """series: [(label, metric_name)]"""
    vals = [mv(r, m) for r in runs for _, m in series]
    vals = [v for v in vals if v is not None]
    if not vals:
        return f'<figure class="card"><figcaption>{html.escape(title)}</figcaption><p class="empty">no data yet</p></figure>'
    ymax = nice_max(max(vals) * 1.05)
    n = len(runs)
    x = lambda i: PL + (W - PL - PR) * (0.5 if n == 1 else i / (n - 1))
    y = lambda v: PT + (H - PT - PB) * (1 - v / ymax)
    g = []
    for k in range(5):
        v = ymax * k / 4
        g.append(f'<line class="grid" x1="{PL}" x2="{W - PR}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                 f'<text class="tick" x="{PL - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:g}</text>')
    for i, r in enumerate(runs):
        if n <= 12 or i % max(1, n // 12) == 0 or i == n - 1:
            g.append(f'<text class="tick" x="{x(i):.1f}" y="{H - PB + 16}" text-anchor="middle">{html.escape(str(r.get("label"))[:14])}</text>')
    ends = []
    for si, (lab, m) in enumerate(series):
        col = SERIES[si]
        pts = [(i, mv(r, m), r) for i, r in enumerate(runs)]
        pts = [p for p in pts if p[1] is not None]
        if not pts:
            continue
        if len(pts) > 1:
            d = " ".join(f"{'M' if j == 0 else 'L'}{x(i):.1f},{y(v):.1f}" for j, (i, v, _) in enumerate(pts))
            g.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2"/>')
        for i, v, r in pts:
            hollow = r.get("contaminated")
            g.append(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="4.5" fill="{"var(--surface)" if hollow else col}" stroke="{col}" '
                     f'stroke-width="2"><title>{html.escape(lab)} {v:g} {unit} - {html.escape(r["id"])}'
                     f'{" (CONTAMINATED)" if hollow else ""}</title></circle>')
        i, v, _ = pts[-1]
        ends.append([y(v) + 4, x(i) + 8, col, lab, v])
    ends.sort()
    for k in range(1, len(ends)):  # keep end labels >= 12 px apart
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 12)
    for yy, xx, col, lab, v in ends:
        g.append(f'<text class="dl" x="{xx:.1f}" y="{yy:.1f}"><tspan fill="{col}">&#9679;</tspan> {html.escape(lab)} {v:.4g}</text>')
    legend = " ".join(f'<span class="key"><i style="background:{SERIES[k]}"></i>{html.escape(l)}</span>' for k, (l, _) in enumerate(series))
    return (f'<figure class="card"><figcaption>{html.escape(title)} <span class="unit">({unit})</span></figcaption>'
            f'<div class="legend">{legend}</div><svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">{"".join(g)}</svg></figure>')


CATS = [("experts", "var(--s1)"), ("attention/GDN", "var(--s2)"), ("dense", "var(--s3)"), ("lm_head", "var(--s4)"),
        ("copies", "var(--s6)"), ("other", "var(--s5)"), ("idle/gaps", "var(--idle)")]


def split_chart(tag):
    rows = [(r, ((r.get("split") or {}).get(tag) or {}).get("pct")) for r in runs]
    rows = [(r, p) for r, p in rows if p]
    if not rows:
        return f'<figure class="card"><figcaption>Decode GPU time split, {tag}</figcaption><p class="empty">no data yet</p></figure>'
    W2 = 1040
    bh, gap = 18, 8
    h = PT + len(rows) * (bh + gap) + 10
    lw = 120
    g = []
    for k, (r, p) in enumerate(rows):
        yy = PT + k * (bh + gap)
        g.append(f'<text class="tick" x="{lw - 6}" y="{yy + 13}" text-anchor="end">{html.escape(str(r.get("label"))[:16])}</text>')
        tot = sum(p.get(c, 0) for c, _ in CATS) or 100
        xx = lw
        for c, col in CATS:
            w = (W2 - lw - 10) * p.get(c, 0) / tot
            if w > 0.5:
                g.append(f'<rect x="{xx:.1f}" y="{yy}" width="{max(w - 2, 0.5):.1f}" height="{bh}" rx="2" fill="{col}">'
                         f'<title>{c}: {p.get(c, 0):.1f}% - {html.escape(r["id"])}</title></rect>')
                if w > 34:
                    g.append(f'<text class="inbar" x="{xx + 4:.1f}" y="{yy + 13}">{p.get(c, 0):.0f}%</text>')
            xx += w
    legend = " ".join(f'<span class="key"><i style="background:{c}"></i>{n}</span>' for n, c in CATS)
    return (f'<figure class="card wide"><figcaption>Decode GPU time split, {tag} <span class="unit">(% of decode window)</span></figcaption>'
            f'<div class="legend">{legend}</div><svg viewBox="0 0 {W2} {h}" role="img">{"".join(g)}</svg></figure>')


def table():
    hd = ("run", "status", "identity", "decode MTP=2", "decode off", "13k prompt tok/s", "13k warm TTFT s", "b1 MoE layer us", "b3 MoE layer us")
    tr = []
    for r in reversed(runs):
        f = lambda n, d=1: "-" if mv(r, n) is None else f"{mv(r, n):.{d}f}"
        tr.append("<tr>" + "".join(f"<td>{c}</td>" for c in (
            html.escape(r["id"]), "CONTAMINATED" if r.get("contaminated") else "clean", "ok" if r.get("identity_ok") else "FAIL",
            f("e2e.decode_tok_s.mtp2"), f("e2e.decode_tok_s.off"), f("e2e.prompt_tok_s.13k", 0), f("e2e.ttft_s.13k_warm", 3),
            f("path.b1@off.span_us"), f("path.b3@mtp2.span_us"))) + "</tr>")
    return ("<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in hd) + "</tr></thead><tbody>" + "".join(tr) + "</tbody></table>")


charts = [
    line_chart("Decode, 8 prompts x 256 greedy", "tok/s", [("MTP=2", "e2e.decode_tok_s.mtp2"), ("MTP off", "e2e.decode_tok_s.off")]),
    line_chart("Cold prompt throughput", "tok/s", [("~4k", "e2e.prompt_tok_s.4k"), ("~13k", "e2e.prompt_tok_s.13k")]),
    line_chart("Time to first token, short prompts (cold)", "ms", [("MTP=2", "e2e.ttft_cold_ms.mtp2"), ("MTP off", "e2e.ttft_cold_ms.off")]),
    line_chart("13k prompt re-sent (prefix cache warm) TTFT", "s", [("warm 13k", "e2e.ttft_s.13k_warm")]),
    line_chart("MoE layer forward, decode (path tier)", "us", [("b1 (MTP off)", "path.b1@off.span_us"), ("b3 verify", "path.b3@mtp2.span_us"),
                                                               ("MTP draft", "path.mtp_draft@mtp2.span_us")]),
    line_chart("MoE layer forward, prefill chunk (path tier)", "us", [("b512", "path.b512@off.span_us")]),
    line_chart("Expert kernels vs llama.cpp", "ours / llama.cpp", [
        ("gate/up b1", "kernel.q35.exp_gate_up.q4_K.k2048n512/b1.ratio"), ("gate/up b3", "kernel.q35.exp_gate_up.q4_K.k2048n512/b3.ratio"),
        ("down Q5_K b1", "kernel.q35.exp_down.q5_K.k512n2048/b1.ratio"), ("down Q5_K b3", "kernel.q35.exp_down.q5_K.k512n2048/b3.ratio")]),
    line_chart("Expert kernels at prefill vs llama.cpp", "ours / llama.cpp", [
        ("gate/up b512", "kernel.q35.exp_gate_up.q4_K.k2048n512/b512.ratio"), ("down Q5_K b512", "kernel.q35.exp_down.q5_K.k512n2048/b512.ratio")]),
    split_chart("mtp2"),
]
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>titan-engine bench history</title><style>
:root{{--surface:#fcfcfb;--bg:#f4f4f2;--ink:#0b0b0b;--ink2:#52514e;--grid:#e3e2dd;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#4a3aa7;--idle:#b9b8b0}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--surface:#1a1a19;--bg:#111110;--ink:#fff;--ink2:#c3c2b7;--grid:#34342f;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#9085e9;--idle:#5d5c55}}}}
:root[data-theme="dark"]{{--surface:#1a1a19;--bg:#111110;--ink:#fff;--ink2:#c3c2b7;--grid:#34342f;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#9085e9;--idle:#5d5c55}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}}
main{{max-width:1120px;margin:0 auto;padding:16px}} h1{{font-size:20px;margin:4px 0}} p.sub{{color:var(--ink2);margin:0 0 14px}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:14px}}
.card{{background:var(--surface);border-radius:8px;padding:12px;margin:0}} .card.wide{{grid-column:1/-1}}
figcaption{{font-weight:600}} .unit{{color:var(--ink2);font-weight:400}} svg{{width:100%;height:auto;display:block}}
.grid{{stroke:var(--grid);stroke-width:1}} .tick{{fill:var(--ink2);font-size:11px}} .dl{{fill:var(--ink);font-size:11px}}
.inbar{{fill:#fff;font-size:10px}} .legend{{color:var(--ink2);font-size:12px;margin:4px 0}} .key{{margin-right:12px;white-space:nowrap}}
.key i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}}
.empty{{color:var(--ink2)}} .tw{{overflow-x:auto;margin-top:16px}} table{{border-collapse:collapse;background:var(--surface);font-size:12px;width:100%}}
th,td{{padding:5px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}}
</style></head><body><main><h1>titan-engine bench history</h1>
<p class="sub">{len(runs)} run(s) from bench/history.jsonl. Hollow markers are CONTAMINATED runs. Hover a point for its value. Reports: bench/runs/&lt;id&gt;.md</p>
<div class="grid2">{''.join(charts)}</div><div class="tw">{table()}</div></main></body></html>"""
open(f"{B}/history.html", "w").write(page)
