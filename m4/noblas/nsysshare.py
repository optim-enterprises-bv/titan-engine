#!/usr/bin/env python3
"""nsysshare.py SQLITE...: GPU-time share of cuBLAS / cuBLASLt kernels in an nsys trace (cuda + cublas NVTX).
A kernel is cuBLAS's when its launch call (runtime/driver API, same thread) lies inside a cuBLAS NVTX range; kernels
replayed from CUDA graphs have no launch call, so every kernel whose name was ever launched from a cuBLAS range
counts as cuBLAS. Prints total kernel time, cuBLAS time and share, and the top cuBLAS kernels."""
import sqlite3, sys, bisect, collections
for db in sys.argv[1:]:
    c = sqlite3.connect(db)
    names = dict(c.execute("select id, value from StringIds"))
    rng = collections.defaultdict(list)  # tid -> [(start, end)]
    for s, e, t, tid in c.execute("select start, end, coalesce(text, (select value from StringIds where id = textId)), globalTid from NVTX_EVENTS where end is not null"):
        if t and (t.startswith("cublas") and t != "cublasCreate_v2" and "Heuristic" not in t and t != "cublasLtCreate"):
            rng[tid].append((s, e))
    for v in rng.values():
        v.sort()
    starts = {t: [r[0] for r in v] for t, v in rng.items()}
    api = {}
    for s, corr, tid in c.execute("select start, correlationId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME"):
        api[corr] = (s, tid)
    def in_range(s, tid):
        v = rng.get(tid)
        if not v:
            return False
        i = bisect.bisect_right(starts[tid], s) - 1
        return i >= 0 and v[i][0] <= s <= v[i][1]
    kern = list(c.execute("select start, end, correlationId, shortName, graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL"))
    blas_names = set()
    for s, e, corr, nid, g in kern:
        a = api.get(corr)
        if a and in_range(*a):
            blas_names.add(nid)
    tot = sum(e - s for s, e, *_ in kern)
    per = collections.Counter()
    cnt = collections.Counter()
    for s, e, corr, nid, g in kern:
        if nid in blas_names:
            per[names[nid]] += e - s
            cnt[names[nid]] += 1
    b = sum(per.values())
    print(f"{db.split('/')[-1]:34s} kernels {len(kern):7d}  GPU {tot/1e6:9.1f} ms  cuBLAS {b/1e6:8.2f} ms  share {100*b/max(tot,1):5.2f}%  "
          f"(ranges {sum(len(v) for v in rng.values())}, kernel span {(max(e for _, e, *_ in kern) - min(s for s, *_ in kern))/1e9:.1f} s)")
    for n, t in per.most_common(6):
        print(f"      {t/1e6:8.2f} ms {cnt[n]:6d}x  {n[:110]}")
