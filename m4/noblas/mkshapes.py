#!/usr/bin/env python3
"""mkshapes.py GLOG... > shapes.txt: the inventory's cuBLAS / cuBLASLt calls (warm, short and long phases; not the
load phase) as row-major problems for the oxide gemm bench (oxide-kernels/gemm/src/bench.rs), with counts per model.
G (candle matmul, cuBLAS column-major: A' = rhs, B' = lhs): D = lhs @ rhs, lhs = A (m x k), rhs = B (k x n).
L (cuBLASLt fwd: out (batch, n, m) = b a^T): A = b (n x k rows), B = a^T (transa: a (m, k) rows, else a (k, m))."""
import re, sys, collections
cnt = collections.Counter()
for f in sys.argv[1:]:
    model = re.sub(r"^.*inv-|\.glog$", "", f)
    phase = "load"
    for l in open(f):
        p = l.split()
        if not p:
            continue
        if p[0] == "M":
            phase = p[1]
            continue
        if phase == "load" or p[0] not in "GL":
            continue
        kv = {}
        for x in p[2:]:
            if "=" in x:
                kv.setdefault(x.split("=", 1)[0], x.split("=", 1)[1])
        I = lambda k: int(kv[k])
        if p[0] == "G":
            dt = p[1]
            b, m, n, k = I("b"), I("m"), I("n"), I("k")
            lda, ldb, sa_, sb_ = I("lda"), I("ldb"), I("sa"), I("sb")
            # rhs = B: ta N -> n-contiguous (b_s0 = lda), T -> k-contiguous (b_s1 = lda); lhs = A: tb N -> k-contig
            b_s0, b_s1 = (lda, 1) if kv["ta"] == "N" else (1, lda)
            a_s0, a_s1 = (ldb, 1) if kv["tb"] == "N" else (1, ldb)
            key = (model, dt, b, m, n, k, a_s0, a_s1, b_s0, b_s1, sb_, sa_, 1.0, 0.0, 0, "G")
        else:
            dt = {"half::bfloat::bf16": "bf16", "half::binary16::f16": "f16", "f32": "f32"}[p[1]]
            b, mL, nL, k = I("b"), I("m"), I("n"), I("k")
            lda, ldb = I("lda"), I("ldb")
            transa = kv["transa"] == "true"
            a_s0, a_s1 = ldb, 1
            b_s0, b_s1 = (1, lda) if transa else (lda, 1)
            key = (model, dt, b, nL, mL, k, a_s0, a_s1, b_s0, b_s1, I("sb"), I("sa"), float(kv["alpha"]), float(kv["beta"]),
                   1 if kv["bias"] == "true" else 0, "L")
        cnt[key] += 1
for key, c in sorted(cnt.items(), key=lambda x: (x[0][0], -x[1])):
    model, *rest, tag = key
    print(model, c, " ".join(str(x) for x in rest), tag)
