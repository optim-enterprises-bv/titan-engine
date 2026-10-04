#!/usr/bin/env python3
"""invsum.py GLOG...: TITAN_GEMM_LOG inventory per model and phase (load = before the warm-up marker).
Shape key: kind (G candle matmul / L cuBLASLt / R cuRAND), dtype, batch, m, n, k, ops and leading dims, batch strides."""
import re, sys, collections
rows = []
for f in sys.argv[1:]:
    model = re.sub(r"^.*inv-|\.glog$", "", f)
    phase = "load"
    cnt = collections.Counter()
    for l in open(f):
        l = l.strip()
        if l.startswith("M "):
            phase = l.split()[1]
            continue
        p = l.split()
        if p[0] == "G":
            kv = dict(x.split("=", 1) for x in p[2:] if "=" in x)
            key = f"G {p[1]} b={kv['b']} m={kv['m']} n={kv['n']} k={kv['k']} {kv['ta']}{kv['tb']} lda={kv['lda']} ldb={kv['ldb']} sa={kv['sa']} sb={kv['sb']}"
        elif p[0] == "L":
            kv = {}
            for x in p[2:]:
                if "=" in x:
                    kv.setdefault(x.split("=", 1)[0], x.split("=", 1)[1])
            dt = p[1].split("::")[-1]
            key = (f"L {dt} b={kv['b']} m={kv['m']} n={kv['n']} k={kv['k']} transa={kv['transa']} lda={kv['lda']} sa={kv['sa']} "
                   f"sb={kv['sb']} alpha={kv['alpha']} beta={kv['beta']} bias={kv['bias']} c={kv['c']} heur={kv['heur']}")
        else:
            key = " ".join(p[:3])
        cnt[(phase, key)] += 1
    for (ph, key), c in sorted(cnt.items(), key=lambda x: (x[0][0], -x[1])):
        print(f"{model:22s} {ph:5s} {c:6d}  {key}")
