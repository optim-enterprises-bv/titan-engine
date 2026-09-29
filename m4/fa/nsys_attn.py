#!/usr/bin/env python3
"""Attention share of decode GPU time from an nsys capture (prof.py: one prefix-warm long-context request).

    nsys_attn.py CAPTURE.sqlite [TOKENS]

Window: after the last prefill-looking kernel (mmq / quantize_mmq) if any, else the whole capture.
Attention span: the kernels strictly between a RoPE kernel (`*rope*`, end of the q/k/v prep) and the next
`usigmoid*` (the output gate) - the attention itself: casts, scaling, cuBLAS GEMMs, mask, softmax, copies,
or the flash-decode kernels. Prints the share of GPU busy time, per-span kernel mix (how many GEMMs, i.e.
how many times K and V are read per layer and step) and the top kernels of the window."""
import collections, json, re, sqlite3, sys

db = sys.argv[1]
tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 0
c = sqlite3.connect(db)
S = dict(c.execute("select id, value from StringIds"))
K = [(s, e, S[n], gx, gy, gz) for s, e, n, gx, gy, gz in
     c.execute("select start, end, shortName, gridX, gridY, gridZ from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
if not K:
    print("no kernels"); sys.exit(1)
pre = [i for i, k in enumerate(K) if k[2].startswith("mmq_") or "quantize_mmq" in k[2]]
i0 = pre[-1] + 1 if pre else 0
W = K[i0:]
t0, t1 = W[0][0], max(k[1] for k in W)

def busy(ks):
    tot, cur_s, cur_e = 0, None, None
    for s, e, *_ in sorted(ks):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot

GEMM = re.compile(r"(gemv|gemm|nvjet|cutlass|^Kernel|^kernel|sm\d\d_|splitK|dot_kernel|reduce_1Block)", re.I)
def cat(n):
    if n.startswith("flash_decode"):
        return "flash"
    if GEMM.search(n):
        return "gemm"
    if n.startswith("softmax"):
        return "softmax"
    if n.startswith(("ucopy", "copy2d")):
        return "copy"
    if n.startswith("cast"):
        return "cast"
    return "other"

spans = []
i = 0
while i < len(W):
    if "rope" in W[i][2]:
        j = i + 1
        while j < len(W) and not W[j][2].startswith("usigmoid") and "rope" not in W[j][2]:
            j += 1
        if j < len(W) and W[j][2].startswith("usigmoid"):
            spans.append(W[i + 1:j])
        i = j
    else:
        i += 1

win_busy = busy(W)
attn_k = [k for sp in spans for k in sp]
attn_t = sum(k[1] - k[0] for k in attn_k)
print(f"capture {db}: {len(K)} kernels, window from kernel {i0} ({len(W)} kernels), {(t1 - t0) / 1e6:.1f} ms wall, "
      f"GPU busy {win_busy / 1e6:.1f} ms ({100 * win_busy / (t1 - t0):.1f}%)")
print(f"attention spans: {len(spans)}, attention kernel time {attn_t / 1e6:.1f} ms = {100 * attn_t / win_busy:.1f}% of GPU busy, "
      f"{100 * attn_t / (t1 - t0):.1f}% of wall")
if tokens:
    print(f"per token ({tokens}): wall {(t1 - t0) / 1e6 / tokens:.2f} ms, busy {win_busy / 1e6 / tokens:.2f} ms, "
          f"attention {attn_t / 1e6 / tokens:.2f} ms")
mix = collections.Counter()
for sp in spans:
    cc = collections.Counter(cat(k[2]) for k in sp)
    mix[(cc["gemm"], cc["softmax"], cc["flash"], len(sp))] += 1
print("span kinds (gemm launches, softmax launches, flash launches, kernels): count, mean span kernel time")
for key, n in sorted(mix.items(), key=lambda x: -x[1]):
    ts = [sum(k[1] - k[0] for k in sp) for sp in spans
          if (sum(cat(k[2]) == "gemm" for k in sp), sum(cat(k[2]) == "softmax" for k in sp),
              sum(cat(k[2]) == "flash" for k in sp), len(sp)) == key]
    print(f"  {key}: {n} spans, {sum(ts) / len(ts) / 1e3:.1f} us each")
bycat = collections.Counter()
for k in attn_k:
    bycat[cat(k[2])] += k[1] - k[0]
print("attention time by kind: " + ", ".join(f"{a} {b / 1e6:.1f} ms" for a, b in bycat.most_common()))
byname = collections.defaultdict(lambda: [0, 0])
for k in attn_k:
    byname[k[2]][0] += 1
    byname[k[2]][1] += k[1] - k[0]
print("attention kernels:")
for n, (cnt, t) in sorted(byname.items(), key=lambda x: -x[1][1])[:12]:
    print(f"  {t / 1e6:8.2f} ms {cnt:6d}x {t / cnt / 1e3:8.1f} us  {n[:90]}")
allk = collections.defaultdict(lambda: [0, 0])
for k in W:
    allk[k[2]][0] += 1
    allk[k[2]][1] += k[1] - k[0]
print("top kernels of the window:")
for n, (cnt, t) in sorted(allk.items(), key=lambda x: -x[1][1])[:15]:
    print(f"  {t / 1e6:8.2f} ms {100 * t / win_busy:5.1f}% {cnt:6d}x  {n[:90]}")
json.dump({"db": db, "wall_ms": (t1 - t0) / 1e6, "busy_ms": win_busy / 1e6, "attn_ms": attn_t / 1e6, "spans": len(spans),
           "tokens": tokens, "attn_by_kind_ms": {a: b / 1e6 for a, b in bycat.items()}},
          open(db.replace(".sqlite", ".attn.json"), "w"), indent=1)
