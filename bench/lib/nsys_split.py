#!/usr/bin/env python3
"""Path tier + GPU time split from an e2e nsys capture (client.py plan `nsys`: two reps of one ~2.1k-token cold prompt
+ 32 decode tokens, service config).

    nsys_split.py CAPTURE.sqlite OUT.json

MoE block = one layer's MoE forward as the model runs it: from the post-attention rmsnorm before the router top-k
kernel to the next rmsnorm (the next layer's attention norm, or the final norm). Its wall span includes the router,
the tiered expert kernels, the CPU-miss sync/upload gaps, the combine and the shared expert. Batch B of a block comes
from the first tiered gate/up gemv grid (512 rows x 8 experts x B blocks) or, without one, 4 x the top-k grid.
Blocks with stock indexed_moe kernels are the MTP draft head.

Decode window (per rep) = end of the rep's last prefill block -> last kernel before a >100 ms gap. Per-category share
of that window: experts, attention/GDN, dense (Q8_0 projections incl. shared expert), lm_head, other, idle."""
import bisect, collections, json, re, sqlite3, statistics, sys

db, outp = sys.argv[1], sys.argv[2]
c = sqlite3.connect(db)
S = dict(c.execute("select id, value from StringIds"))
try:
    K = [(s, e, S[n], gx, gy, by) for s, e, n, gx, gy, by in
         c.execute("select start, end, shortName, gridX, gridY, blockY from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
except sqlite3.OperationalError:
    K = []
try:
    MC = list(c.execute("select start, end, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY order by start"))
except sqlite3.OperationalError:
    MC = []
if not K:
    json.dump({"error": "no kernels in capture"}, open(outp, "w"))
    sys.exit(1)
starts = [k[0] for k in K]

import os
PF_MIN = int(os.environ.get("NSYS_SPLIT_PF_MIN", "64"))  # batch at or above which a block is prefill
GATEUP = re.compile(r"^(q4k|q5k|q6k|q8_0|iq4_nl|mxfp4|nvfp4)_q8_1_moe_gemv$")
# mmvq-moe (llama.cpp-shaped tiered expert kernels): b1 grid (rows, 8 x B) x (32, 4); MoE grid (rows / 2, 8) x (32, B)
GATEUP_B1 = re.compile(r"^(q4k|q5k|q6k|q1_0|iq4_nl|mxfp4|nvfp4)_moe_b1(_sk)?$")
GATEUP_MOE = re.compile(r"^(q4k|q5k|q6k|q1_0|iq4_nl|mxfp4|nvfp4)_moe_mmvq$")
LM_ROWS = 248320


def is_norm(n):
    return n.startswith("rmsnorm")


def base_cat(n, gx):
    if n.startswith(("ucopy", "copy2d")):
        return "copies"  # strided/contiguous copies: in decode mostly the attention KV-cache append (cat) per layer
    if "_moe" in n or n.startswith("indexed_moe") or n in ("quantize_q8_1", "moe_router_topk_kernel", "fast_sum_f32",
                                                              "scatter_rows") or n.startswith("fused_glu"):
        return "experts"
    if n.startswith("mmvq_gguf_q8_0") or n.startswith("mmvq_gguf_q"):
        if "quantize" not in n and (gx * 2 == LM_ROWS or gx == LM_ROWS):
            return "lm_head"
        return "dense"
    if n.startswith("mmq_") or n.startswith("quantize_mmq") or n.startswith("mmvq_gguf_quantize"):
        return "dense"
    if (n.startswith("gdn_") or "conv1d" in n or "gated_delta" in n or n.startswith("save_conv") or "rope" in n
            or n.startswith("softmax") or "causal_mask" in n or "flash" in n or "attn" in n):
        return "attention/GDN"
    return None  # cuBLAS / elementwise: decided by position (inside an MoE block or not)


CUBLAS = re.compile(r"^(kernel|Kernel2|dot_kernel|reduce_1Block_kernel|gemv|gemm|nvjet|splitKreduce|cutlass|sgemm|ampere|sm\d)")

# ---- MoE blocks ----
blocks = []
last_norm = -1
i = 0
while i < len(K):
    n = K[i][2]
    if is_norm(n):
        last_norm = i
    if n == "moe_router_topk_kernel":
        st = last_norm if last_norm >= 0 and (not blocks or last_norm > blocks[-1]["i1"]) else i
        j = i + 1
        while j < len(K) and not is_norm(K[j][2]) and K[j][2] != "moe_router_topk_kernel":
            j += 1
        j_end = min(j, len(K) - 1)
        span_end = K[j][0] if j < len(K) else K[-1][1]
        ks = K[st:j]
        B, indexed = None, False
        for k in ks:
            if GATEUP.match(k[2]) and B is None:
                B = max(1, k[3] // (512 * 8))
            elif GATEUP_B1.match(k[2]) and B is None:
                B = max(1, k[4] // 8)
            elif GATEUP_MOE.match(k[2]) and B is None:
                B = max(1, k[5])
            if k[2].startswith("indexed_moe"):
                indexed = True
        if B is None:
            B = 4 * K[i][3]
        busy = collections.defaultdict(float)
        union, cur_s, cur_e = 0, None, None
        for k in ks:
            cat = base_cat(k[2], k[3])
            if cat is None:
                cat = "experts" if CUBLAS.match(k[2]) or k[2] in ("usigmoid_f32",) else "other"
            if cat == "dense":
                cat = "shared_expert"
            busy[cat] += (k[1] - k[0]) / 1e3
            if cur_e is None or k[0] > cur_e:
                if cur_e is not None:
                    union += cur_e - cur_s
                cur_s, cur_e = k[0], k[1]
            else:
                cur_e = max(cur_e, k[1])
        if cur_e is not None:
            union += cur_e - cur_s
        span = (span_end - K[st][0]) / 1e3
        blocks.append({"i0": st, "i1": j_end, "t0": K[st][0], "t1": span_end, "B": B, "mtp_draft": indexed,
                       "span_us": span, "busy_us": union / 1e3, "idle_us": span - union / 1e3, "cat_us": dict(busy)})
        i = j
        continue
    i += 1

# ---- reps: a prefill block after a non-prefill block starts a new rep ----
rep, prev_pf = 0, False
for b in blocks:
    pf = b["B"] >= PF_MIN
    if pf and not prev_pf:
        rep += 1
    b["rep"], b["prefill"] = max(rep, 1), pf
    prev_pf = pf


def cls(b):
    if b["mtp_draft"]:
        return "mtp_draft"
    return f"b{b['B']}"


path = {}
groups = collections.defaultdict(lambda: collections.defaultdict(list))
for b in blocks:
    groups[cls(b)][b["rep"]].append(b)
for k, reps in groups.items():
    allb = [b for r in reps.values() for b in r]
    med = lambda key, bs: statistics.median(x[key] for x in bs)
    rep_span = {r: med("span_us", bs) for r, bs in sorted(reps.items())}
    m = med("span_us", allb)
    cats = sorted({c for b in allb for c in b["cat_us"]})
    path[k] = {"blocks": len(allb), "span_us": round(m, 1), "busy_us": round(med("busy_us", allb), 1),
               "idle_us": round(med("idle_us", allb), 1), "rep_span_us": [round(v, 1) for v in rep_span.values()],
               "spread": round((max(rep_span.values()) - min(rep_span.values())) / m, 4) if len(rep_span) > 1 and m else None,
               "cat_us": {c: round(statistics.median(b["cat_us"].get(c, 0) for b in allb), 1) for c in cats}}

# ---- decode windows and category split ----
kend = [k[1] for k in K]
split_reps = []
topcat = collections.Counter()  # (category, kernel) -> ns over all decode windows
for r in sorted({b["rep"] for b in blocks}):
    pfs = [b for b in blocks if b["rep"] == r and b["prefill"]]
    if not pfs:
        continue
    w0 = pfs[-1]["t1"]
    idx = bisect.bisect_left(starts, w0)
    last = idx
    while last + 1 < len(K) and K[last + 1][0] - K[last][1] < 100e6:
        last += 1
    w1 = K[last][1]
    if w1 <= w0:
        continue
    cat = collections.defaultdict(float)
    blk = [b for b in blocks if b["rep"] == r and not b["prefill"]]
    bi = 0
    union, cs, ce = 0, None, None
    for k in K[idx:last + 1]:
        while bi < len(blk) and blk[bi]["t1"] <= k[0]:
            bi += 1
        inside = bi < len(blk) and blk[bi]["t0"] <= k[0] < blk[bi]["t1"]
        ct = base_cat(k[2], k[3])
        if ct is None:
            ct = "experts" if inside and (CUBLAS.match(k[2]) or k[2] == "usigmoid_f32") else (
                "attention/GDN" if CUBLAS.match(k[2]) else "other")
        cat[ct] += k[1] - k[0]
        topcat[(ct, k[2])] += k[1] - k[0]
        if ce is None or k[0] > ce:
            if ce is not None:
                union += ce - cs
            cs, ce = k[0], k[1]
        else:
            ce = max(ce, k[1])
    if ce is not None:
        union += ce - cs
    win = w1 - w0
    cat["idle/gaps"] = win - union
    h2d = sum(m[1] - m[0] for m in MC if w0 <= m[0] < w1 and m[2] == 1)
    d2h = [m for m in MC if w0 <= m[0] < w1 and m[2] == 2]
    steps = sum(1 for m in d2h if m[3] >= 200000)  # logits copies
    split_reps.append({"rep": r, "window_ms": win / 1e6, "pct": {k: 100 * v / win for k, v in cat.items()},
                       "ms": {k: v / 1e6 for k, v in cat.items()}, "h2d_ms": h2d / 1e6, "logit_copies": steps,
                       "moe_blocks": len(blk)})

split = {}
if split_reps:
    keys = ["experts", "attention/GDN", "dense", "lm_head", "copies", "other", "idle/gaps"]
    split = {"window_ms": round(statistics.median(s["window_ms"] for s in split_reps), 2),
             "pct": {k: round(statistics.median(s["pct"].get(k, 0) for s in split_reps), 2) for k in keys},
             "rep_pct": [{k: round(s["pct"].get(k, 0), 2) for k in keys} for s in split_reps],
             "h2d_ms": round(statistics.median(s["h2d_ms"] for s in split_reps), 2),
             "logit_copies": [s["logit_copies"] for s in split_reps],
             "reps": len(split_reps),
             "top_kernels_ms_per_rep": {f"{ct}: {n}": round(v / 1e6 / len(split_reps), 2) for (ct, n), v in topcat.most_common(20)}}
top = collections.Counter()
for k in K:
    top[k[2]] += k[1] - k[0]
tot = sum(top.values())
res = {"path": path, "split": split, "kernels_total_ms": tot / 1e6,
       "top_kernels_pct": {n: round(100 * v / tot, 2) for n, v in top.most_common(15)},
       "capture_s": (K[-1][1] - K[0][0]) / 1e9}
json.dump(res, open(outp, "w"), indent=1)
print(f"{db}: {len(K)} kernels, {len(blocks)} MoE blocks, capture {res['capture_s']:.1f}s")
for k in sorted(path, key=lambda x: (x != "mtp_draft", int(x[1:]) if x[1:].isdigit() else 0)):
    p = path[k]
    print(f"  path {k:9s} n={p['blocks']:5d} span {p['span_us']:8.1f} us (busy {p['busy_us']:.1f}, idle {p['idle_us']:.1f}) "
          f"reps {p['rep_span_us']}  {p['cat_us']}")
if split:
    print(f"  decode window {split['window_ms']} ms/rep: " + ", ".join(f"{k} {v:.1f}%" for k, v in split["pct"].items()))
