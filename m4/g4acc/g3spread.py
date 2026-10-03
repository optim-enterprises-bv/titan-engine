#!/usr/bin/env python3
"""G3 against llama.cpp's own self-variation.

usage: g3spread.py REF.json LLAMA_VARIANT.json[,...] -- TITAN.json [TITAN2.json ...]
REF is the llama.cpp reference (m4/g4s/out/ref-g12-g3.json or -g3l), the variants are the same llama.cpp binary and
precision with only batching changed (-ub 256, -ub 128, -b 512: g3spread.sh produces them), each scored against REF
exactly as cmp.py does (top-1 agreement, median |dlogprob| of REF's top-1 token, top-20 KL median). A titan run is
"within llama.cpp's spread" when its top-1 agreement is >= the worst variant's and its median |dlogprob| and KL median
are <= the worst variant's. Prints one row per run and the verdict."""
import json, math, statistics, sys

def score(t, l):
    agree, dlp, kls = 0, [], []
    for a, b in zip(t, l):
        ta, lb = a["top"][0], b["top"][0]
        tm = dict((k, v) for k, v in ta)
        floor = min(v for _, v in ta)
        agree += ta[0][0] == lb[0][0]
        dlp.append(abs(tm.get(lb[0][0], floor) - lb[0][1]))
        kls.append(sum(math.exp(lp) * (lp - tm.get(tok, floor)) for tok, lp in lb))
    return agree, statistics.median(dlp), statistics.median(kls)

ref = json.load(open(sys.argv[1]))
sep = sys.argv.index("--")
variants = [f for a in sys.argv[2:sep] for f in a.split(",")]
titans = sys.argv[sep + 1:]
rows = [(f, score(json.load(open(f)), ref)) for f in variants]
n = len(ref)
w_top = min(r[1][0] for r in rows)
w_dlp = max(r[1][1] for r in rows)
w_kl = max(r[1][2] for r in rows)
print(f"reference {sys.argv[1].split('/')[-1]} ({n} prompts)")
for f, (a, d, k) in rows:
    print(f"  llama variant {f.split('/')[-1]:24} top-1 {a}/{n}  dlp {d:.4f}  KL {k:.5f}")
print(f"  llama.cpp self-spread (worst):           top-1 {w_top}/{n}  dlp {w_dlp:.4f}  KL {w_kl:.5f}")
for f in titans:
    a, d, k = score(json.load(open(f)), ref)
    ok = a >= w_top and d <= w_dlp and k <= w_kl
    print(f"  titan {f.split('/')[-1]:32} top-1 {a}/{n}  dlp {d:.4f}  KL {k:.5f}  -> {'WITHIN' if ok else 'OUTSIDE'} llama.cpp's spread"
          f"; absolute G3 (dlp < 0.05, KL < 0.01): {'PASS' if d < 0.05 and k < 0.01 else 'FAIL'}")
