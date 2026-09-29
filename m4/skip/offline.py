#!/usr/bin/env python3
"""Offline studies for the 80B routing.
  offline.py skip WTRACE                 miss-skip variants from a weight trace (decode lines only)
  offline.py prune WTRACE IDTRACE PROFILE  REAP / frequency expert ranking, prune lists and RAM estimate
WTRACE lines: `layer batch topk id:w:norm:res ...` (TITAN_TIERED_WTRACE); IDTRACE: `layer batch topk id ...`."""
import collections, sys
L, E, K = 48, 512, 10
# bytes of one expert summed over gate+up+down, all 48 layers (gate/up Q4_K [512x2048] in every layer; down Q6_K in
# 24 layers and Q4_K in 24: server log "titan tiered experts" lines)
Q4K_EXPERT = 512 * 2048 // 256 * 144
Q6K_EXPERT = 2048 * 512 // 256 * 210
EXPERT_ALL_LAYERS = 48 * 2 * Q4K_EXPERT + 24 * Q6K_EXPERT + 24 * Q4K_EXPERT
GPU_SLOTS = 143

def wlines(path, decode_only=True):
    for line in open(path):
        f = line.split()
        if len(f) < 4 or (decode_only and f[1] != "1"):
            continue
        layer, batch, topk = int(f[0]), int(f[1]), int(f[2])
        tasks = [t.split(":") for t in f[3:]]
        for r in range(batch):
            yield layer, [(int(i), float(w), float(n), r_ == "1") for i, w, n, r_ in tasks[r * topk:(r + 1) * topk]]

def skip(path):
    variants = [("thr 0.02", 0.02, 99, True), ("thr 0.05", 0.05, 99, True), ("thr 0.08", 0.08, 99, True),
                ("thr 0.05, max 1/token", 0.05, 1, True), ("thr 0.08, max 2/token", 0.08, 2, True),
                ("all misses but top-1", 2.0, 99, True), ("thr 0.10", 0.10, 99, True)]
    st = {v[0]: [0, 0.0, 0.0, 0] for v in variants}  # skipped, mass, max mass, rows with a skip
    n = misses = 0; miss_mass = 0.0; wdist = collections.Counter(); top1_miss = 0
    for layer, row in wlines(path):
        n += 1
        top1 = max(range(len(row)), key=lambda j: row[j][1])
        misses += sum(not r for *_, r in row); miss_mass += sum(w for _, w, _, r in row if not r)
        top1_miss += not row[top1][3]
        for _, w, _, r in row:
            if not r:
                wdist[min(int(w / 0.01), 20)] += 1
        for name, thr, mx, keep in variants:
            c = sorted([j for j in range(len(row)) if not row[j][3] and row[j][1] < thr and not (keep and j == top1)], key=lambda j: row[j][1])[:mx]
            if c:
                m = sum(row[j][1] for j in c)
                s = st[name]; s[0] += len(c); s[1] += m; s[2] = max(s[2], m); s[3] += 1
    print(f"{n} decode layer-tokens, {misses} misses ({misses / n:.2f} per layer-token of {K}), non-resident weight mass {miss_mass / n:.3f} per layer-token; top-1 is a miss in {100 * top1_miss / n:.1f}%")
    print("miss weight histogram (0.01 bins, last = >=0.20): " + " ".join(f"{wdist[b] / misses * 100:.1f}" for b in range(21)))
    print(f"{'variant':26} {'misses skipped':>15} {'CPU tasks left':>15} {'mass lost/l-tok':>16} {'max/l-tok':>10} {'l-toks w/ skip':>15}")
    for name, *_ in variants:
        s = st[name]
        print(f"{name:26} {100 * s[0] / misses:14.1f}% {100 * (1 - s[0] / misses):14.1f}% {s[1] / n:16.4f} {s[2]:10.3f} {100 * s[3] / n:14.1f}%")

def prune(wpath, idpath, profile):
    freq = collections.Counter()
    for line in open(idpath):
        f = line.split()
        if len(f) >= 4 and f[1] == "1":
            for e in f[3:]:
                freq[(int(f[0]), int(e))] += 1
    reap_sum, reap_n, wsum = collections.Counter(), collections.Counter(), collections.Counter()
    for layer, row in wlines(wpath):
        for i, w, nrm, _ in row:
            reap_sum[(layer, i)] += w * nrm; reap_n[(layer, i)] += 1; wsum[(layer, i)] += w
    prof = collections.Counter()
    for line in open(profile):
        l, e, c = map(int, line.split()); prof[(l, e)] += c
    resident = {l: set(sorted(range(E), key=lambda e: (-prof[(l, e)], e))[:GPU_SLOTS]) for l in range(L)}
    scores = {
        "freq": lambda l, e: freq[(l, e)],
        # REAP: mean over the tokens routed to the expert of gate weight x ||expert output||
        "reap": lambda l, e: reap_sum[(l, e)] / reap_n[(l, e)] if reap_n[(l, e)] else 0.0,
        # REAP x frequency: the expert's total contribution (sum of w x norm)
        "reap-sum": lambda l, e: reap_sum[(l, e)],
    }
    tot_w = sum(wsum.values()); tot_n = sum(reap_n.values()); tot_f = sum(freq.values())
    print(f"calibration: id trace {tot_f} decode routings, weight trace {tot_n} routings; expert bytes (3 proj x 48 layers) {EXPERT_ALL_LAYERS / 2**20:.1f} MiB per expert index, all experts {E * EXPERT_ALL_LAYERS / 2**30:.1f} GiB")
    lists = {}
    for name, sc in scores.items():
        for pct in (25, 40):
            k = E * pct // 100
            pr = {l: sorted(range(E), key=lambda e: (sc(l, e), e))[:k] for l in range(L)}
            lists[(name, pct)] = pr
            lost_f = sum(freq[(l, e)] for l in range(L) for e in pr[l]) / tot_f
            lost_n = sum(reap_n[(l, e)] for l in range(L) for e in pr[l]) / tot_n
            lost_w = sum(wsum[(l, e)] for l in range(L) for e in pr[l]) / tot_w
            lost_r = sum(reap_sum[(l, e)] for l in range(L) for e in pr[l]) / sum(reap_sum.values())
            gpu_pruned = sum(len(resident[l] & set(pr[l])) for l in range(L)) / L
            host_now = E - GPU_SLOTS
            host_pr = E - k - GPU_SLOTS  # with a prune-aware profile the GPU keeps 143 kept experts
            print(f"{name:8} {pct}%: pruned routings lost {100 * lost_f:.2f}% (id trace) / {100 * lost_n:.2f}% (w trace), weight mass {100 * lost_w:.2f}%, "
                  f"REAP mass {100 * lost_r:.2f}%; {gpu_pruned:.1f} of 143 GPU slots/layer hold a pruned expert (current profile); "
                  f"host experts {host_now} -> {host_pr}/layer = {host_now * EXPERT_ALL_LAYERS / 2**30:.1f} -> {host_pr * EXPERT_ALL_LAYERS / 2**30:.1f} GiB")
            with open(f"prune-{name}-{pct}.txt", "w") as f:
                for l in range(L):
                    f.write(f"{l} " + " ".join(map(str, sorted(pr[l]))) + "\n")
            with open(f"profile-{name}-{pct}.txt", "w") as f:  # placement profile without the pruned experts
                for (l, e), c in sorted(prof.items()):
                    if e not in set(pr[l]):
                        f.write(f"{l} {e} {c}\n")
    for pct in (25, 40):
        a, b = lists[("freq", pct)], lists[("reap", pct)]
        print(f"overlap freq/reap {pct}%: {100 * sum(len(set(a[l]) & set(b[l])) for l in range(L)) / (L * E * pct // 100):.1f}%")

if __name__ == "__main__":
    {"skip": skip, "prune": prune}[sys.argv[1]](*sys.argv[2:])
