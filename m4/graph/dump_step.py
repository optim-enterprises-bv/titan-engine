import sqlite3, sys
db, step_no = sys.argv[1], int(sys.argv[2]); limit = int(sys.argv[3]) if len(sys.argv) > 3 else 400
c = sqlite3.connect(db)
names = dict(c.execute("select id, value from StringIds"))
# find logits d2h marks, pick window
marks = [r for r in c.execute("select start,end from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=2 and bytes>=200000 order by start")]
t0, t1 = marks[step_no][1], marks[step_no + 1][1]
ops = []
for s, e, corr, nid in c.execute("select start,end,correlationId,shortName from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and start<=?", (t0, t1)):
    ops.append((s, e, corr, names[nid]))
for s, e, corr, k, b in c.execute("select start,end,correlationId,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and start<=?", (t0, t1)):
    ops.append((s, e, corr, f"memcpy{k} {b}B"))
for s, e, corr in c.execute("select start,end,correlationId from CUPTI_ACTIVITY_KIND_MEMSET where start>=? and start<=?", (t0, t1)):
    ops.append((s, e, corr, "memset"))
ops.sort()
api = {corr: (s, e, names[n]) for s, e, corr, n in c.execute("select start,end,correlationId,nameId from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and start<=?", (t0 - 50_000_000, t1))}
apis = sorted((s, e, names[n]) for s, e, n in c.execute("select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and start<=?", (t0, t1)))
prev = t0
print(f"step {step_no}: {(t1-t0)/1e6:.2f} ms, {len(ops)} ops")
for s, e, corr, n in ops[:limit]:
    L = api.get(corr, (0, 0, '?'))
    between = [a[2] for a in apis if prev <= a[0] < L[0] and ('Sync' in a[2] or 'DtoH' in a[2])]
    print(f"{(s-t0)/1e3:9.1f} gap {(s-prev)/1e3:7.1f} dur {(e-s)/1e3:6.1f} launch@{(L[0]-t0)/1e3:9.1f} {L[2][:22]:22s} {n[:40]:40s} {' '.join(between)[:40]}")
    prev = e
