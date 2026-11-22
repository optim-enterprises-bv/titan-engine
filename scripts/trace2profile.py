# trace2profile.py <TITAN_TIERED_TRACE file> <profile out>: "layer batch topk ids..." lines -> "layer expert count"
import sys, collections
c = collections.Counter()
for line in open(sys.argv[1]):
    f = line.split()
    layer = int(f[0])
    for e in f[3:]:
        c[(layer, int(e))] += 1
with open(sys.argv[2], "w") as o:
    for (l, e), n in sorted(c.items()):
        o.write(f"{l} {e} {n}\n")
print(len(c), "layer/expert pairs,", sum(c.values()), "routings")
