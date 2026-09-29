"""Idle budget per decode step from an nsys sqlite: kernel busy (accurate under CUPTI), CPU-miss passes
(host compute between the gemv launches and the miss-row upload), and the rest of the idle time."""
import sqlite3, sys, statistics, collections
db = sys.argv[1]; c = sqlite3.connect(db)
names = dict(c.execute("select id, value from StringIds"))
ops = []
for s, e, nid in c.execute("select start,end,shortName from CUPTI_ACTIVITY_KIND_KERNEL"): ops.append((s, e, 'k', names[nid], 0))
for s, e, k, b in c.execute("select start,end,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY"): ops.append((s, e, {1:'h2d',2:'d2h',8:'d2d'}.get(k,'cp'), '', b))
for s, e in c.execute("select start,end from CUPTI_ACTIVITY_KIND_MEMSET"): ops.append((s, e, 'set', '', 0))
ops.sort()
marks = [i for i, o in enumerate(ops) if o[2] == 'd2h' and o[4] >= 200000]
rows = []
for j in range(len(marks) - 1):
    if ops[marks[j+1]][1] - ops[marks[j]][1] < 3e6: continue
    a, b = marks[j] + 1, marks[j + 1]
    wall = ops[b][1] - ops[a-1][1]
    busy = sum(o[1] - o[0] for o in ops[a:b+1] if o[2] == 'k')
    idle = wall - sum(o[1]-o[0] for o in ops[a:b+1])
    cpu = 0; big = collections.Counter(); nsync = 0
    for i in range(a, b + 1):
        g = ops[i][0] - ops[i-1][1]
        if ops[i][2] == 'h2d' and g > 15000: cpu += g   # CPU miss pass ends with the row upload
        if ops[i][2] == 'd2h': nsync += 1
        if g > 15000 and ops[i][2] != 'h2d': big[ops[i][3] or ops[i][2]] += g
    rows.append((wall, busy, idle, cpu, nsync, sum(big.values()), len(ops[a:b+1])))
m = lambda k: statistics.median(r[k] for r in rows) / 1e6
print(f"{db}: {len(rows)} steps; medians ms: wall {m(0):.2f} kernel-busy {m(1):.2f} idle {m(2):.2f} cpu-miss-passes {m(3):.2f} other>15us-gaps {m(5):.2f}; d2h syncs {statistics.median(r[4] for r in rows):.0f}, gpu ops {statistics.median(r[6] for r in rows):.0f}")
