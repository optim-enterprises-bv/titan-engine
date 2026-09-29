#!/usr/bin/env python3
"""GPU idle-gap attribution for batch-1 decode from an nsys sqlite export.

For every GPU op (kernel / memcpy / memset) in the decode window, the idle gap before it is split into:
  queued    - its launch call was issued before the previous op ended (device-side latency only)
  post_sync - the host issued it late, after returning from a blocking call (DtoH / stream sync)
              that waited on the GPU: CPU experts, routing decode, host logic after a sync
  launch    - the host issued it late with no sync in between: candle/Rust + driver launch overhead
and each op is assigned to a region: 'expert' (routing DtoH .. last routed-expert op of the layer)
or 'trunk' (everything else: norms, attention/GDN, router, shared expert, combine, lm head).
Graphs remove trunk 'launch' gaps (and shrink trunk 'queued' gaps); nothing else.
usage: gaps.py file.sqlite [min_step_ms max_step_ms]
"""
import sqlite3, sys, collections, statistics
db = sys.argv[1]
lo_ms = float(sys.argv[2]) if len(sys.argv) > 2 else 0
hi_ms = float(sys.argv[3]) if len(sys.argv) > 3 else 1e9
c = sqlite3.connect(db)
names = dict(c.execute("select id, value from StringIds"))
LOGITS_MIN = 200000  # DtoH of a full logits row (vocab * 4 bytes) marks the end of a step

ops = []  # (start, end, corr, kind, name, bytes)
for s, e, corr, nid in c.execute("select start,end,correlationId,shortName from CUPTI_ACTIVITY_KIND_KERNEL"):
    ops.append((s, e, corr, 'k', names[nid], 0))
for s, e, corr, kind, b in c.execute("select start,end,correlationId,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY"):
    ops.append((s, e, corr, {1: 'h2d', 2: 'd2h', 8: 'd2d'}.get(kind, 'cp'), '', b))
for s, e, corr in c.execute("select start,end,correlationId from CUPTI_ACTIVITY_KIND_MEMSET"):
    ops.append((s, e, corr, 'set', '', 0))
ops.sort()
api = {}
syncs = []  # (start, end) of host calls that block on the GPU
SYNC = ('Synchronize', 'cuMemcpyDtoH', 'cudaMemcpy', 'cuMemcpy2D', 'cuStreamWaitEvent')
for s, e, corr, nid in c.execute("select start,end,correlationId,nameId from CUPTI_ACTIVITY_KIND_RUNTIME"):
    n = names[nid]
    api[corr] = (s, e, n)
    if 'Synchronize' in n or n.startswith('cuMemcpyDtoH'):
        syncs.append((s, e))
syncs.sort()
import bisect
sync_ends = [e for _, e in syncs]

# steps: between consecutive logits DtoH
marks = [i for i, o in enumerate(ops) if o[3] == 'd2h' and o[5] >= LOGITS_MIN]
steps = [(marks[j] + 1, marks[j + 1]) for j in range(len(marks) - 1)]
steps = [(a, b) for a, b in steps if lo_ms <= (ops[b][1] - ops[a - 1][1]) / 1e6 <= hi_ms]

EXPERT_K = ('moe_gemv', 'quantize_q8_1', 'scatter_rows')
def region_marks(a, b):
    """per op in [a, b]: True if in an expert region"""
    reg = [False] * (b - a + 1)
    i = a
    while i <= b:
        o = ops[i]
        if o[3] == 'd2h' and o[5] < 4096 and o[5] % 4 == 0 and o[5] <= 256:  # routing ids (topk u32 per row)
            j, gemv = i, 0
            while j <= b:
                oj = ops[j]
                if oj[3] == 'k' and 'moe_gemv' in oj[4]:
                    gemv += 1
                reg[j - a] = True
                if gemv >= 3:
                    # absorb trailing miss traffic of the down projection
                    while j + 1 <= b and (ops[j + 1][3] in ('d2h', 'h2d') or 'scatter_rows' in ops[j + 1][4]):
                        j += 1
                        reg[j - a] = True
                    break
                j += 1
            i = j + 1
        else:
            i += 1
    return reg

tot = collections.Counter()
per_step = []
nk = collections.Counter()
for a, b in steps:
    reg = region_marks(a, b)
    st = collections.Counter()
    for i in range(a, b + 1):
        s, e, corr, kind, name, _ = ops[i]
        r = 'expert' if reg[i - a] else 'trunk'
        st[r + '.busy'] += e - s
        nk[r + '.ops'] += 1
        prev_end = ops[i - 1][1]
        gap = max(0, s - prev_end)
        L = api.get(corr, (s, s, ''))[0]
        if gap == 0:
            continue
        if L <= prev_end:
            st[r + '.queued'] += gap
        else:
            k = bisect.bisect_left(sync_ends, prev_end)
            if k < len(sync_ends) and sync_ends[k] <= L:
                st[r + '.post_sync'] += gap
            else:
                st[r + '.launch'] += gap
        st[r + '.api'] += 1
    wall = ops[b][1] - ops[a - 1][1]
    st['wall'] = wall
    per_step.append(st)
    tot.update(st)
n = len(per_step)
if n == 0:
    sys.exit("no decode steps found")
ms = lambda v: v / n / 1e6
print(f"{db}: {n} steps, median step {statistics.median(s['wall'] for s in per_step)/1e6:.2f} ms, mean {ms(tot['wall']):.2f} ms  ({1e3/ms(tot['wall']):.1f} steps/s)")
med = lambda k: statistics.median(s[k] for s in per_step) / 1e6
print("medians ms/step: " + ", ".join(f"{k} {med(k):.3f}" for k in ['wall','trunk.busy','trunk.launch','trunk.queued','trunk.post_sync','expert.busy','expert.launch','expert.queued','expert.post_sync']))
print(f"launch calls per step: trunk {tot['trunk.api']/n:.0f} expert {tot['expert.api']/n:.0f}")
print(f"ops per step: trunk {nk['trunk.ops']/n:.0f}, expert {nk['expert.ops']/n:.0f}")
for k in ['trunk.busy', 'trunk.launch', 'trunk.queued', 'trunk.post_sync', 'expert.busy', 'expert.launch', 'expert.queued', 'expert.post_sync']:
    print(f"  {k:18s} {ms(tot[k]):7.3f} ms/step  {100*tot[k]/tot['wall']:5.1f}%")
acc = sum(tot[k] for k in tot if k != 'wall' and not k.endswith('.api'))
print(f"  {'unaccounted':18s} {ms(tot['wall']-acc):7.3f} ms/step")
# upper bound: graphs remove trunk launch gaps and leave ~1 us per trunk op (node latency) of the queued gaps
ub = tot['trunk.launch']
print(f"graph upper bound (all trunk launch-bound idle): {ms(ub):.3f} ms/step = {100*ub/tot['wall']:.1f}% of decode wall")
