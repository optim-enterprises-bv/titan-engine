#!/usr/bin/env python3
"""Offline routing analysis for the 80B from TITAN_TIERED_TRACE files (`layer batch topk id...`, gate only).
usage: analyze.py CALIB_TRACE EVAL_TRACE PROFILE NUM_RESIDENT [FLEX...]
  1. decode hit rate of the profile placement (NUM_RESIDENT hottest per layer) on the eval trace
  2. next-layer predictors, precision / recall on eval decode steps, over all routed experts and over
     non-resident ones: co-occurrence (layer L ids -> L+1, counts from the calib trace), previous token
     (same layer), and the static profile (the hottest non-resident experts)
  3. P2 fingerprint: per request, the FLEX coldest slots take the hottest non-core experts of its prefill
"""
import collections, sys
import numpy as np

def requests(path):
    """[(prefill {layer: Counter}, [token {layer: [ids]}])], a new request at each prefill."""
    reqs, cur, tok = [], None, None
    for line in open(path):
        f = line.split()
        if len(f) < 4:
            continue
        layer, batch, ids = int(f[0]), int(f[1]), [int(x) for x in f[3:]]
        if batch > 1:
            if cur is None or cur[1]:
                cur = (collections.defaultdict(collections.Counter), []); reqs.append(cur)
            cur[0][layer].update(ids)
            continue
        if cur is None:
            cur = (collections.defaultdict(collections.Counter), []); reqs.append(cur)
        if layer == 0 or tok is None or layer in tok:
            tok = {}; cur[1].append(tok)
        tok[layer] = ids
    return reqs

calib, evl = requests(sys.argv[1]), requests(sys.argv[2])
prof = collections.defaultdict(collections.Counter)
for line in open(sys.argv[3]):
    l, e, c = map(int, line.split()); prof[l][e] += c
n_res = int(sys.argv[4]); flexes = [int(x) for x in sys.argv[5:]] or [8, 16, 32]
layers = sorted({l for r in evl for t in r[1] for l in t})
E = 1 + max(max(ids) for r in evl + calib for t in r[1] for ids in t.values())
order = {l: sorted(range(E), key=lambda e: (-prof[l][e], e)) for l in layers}
res = {l: set(order[l][:n_res]) for l in layers}
toks = [t for r in evl for t in r[1]]
print(f"eval: {len(evl)} requests, {len(toks)} decode steps, {len(layers)} layers, {E} experts, {n_res} resident")

h = m = 0
for t in toks:
    for l, ids in t.items():
        k = sum(e in res[l] for e in ids); h += k; m += len(ids) - k
print(f"1. profile placement, offline decode hit rate {100*h/(h+m):.1f}%")

co = {l: np.zeros((E, E), np.float32) for l in layers}
for r in calib:
    for t in r[1]:
        for l in layers:
            if l in t and l + 1 in t:
                co[l][np.ix_(t[l], t[l + 1])] += 1
def score(name, pairs):
    s = np.zeros(6)
    for pred, act, l in pairs:
        pred, act = set(pred), set(act); nr = lambda x: {e for e in x if e not in res[l]}
        s += [len(pred & act), len(pred), len(act), len(nr(pred) & nr(act)), len(nr(pred)), len(nr(act))]
    print(f"2. {name}: precision {100*s[0]/s[1]:.1f}% recall {100*s[0]/s[2]:.1f}% "
          f"(non-resident: precision {100*s[3]/max(s[4],1):.1f}% recall {100*s[3]/max(s[5],1):.1f}%)")
k = len(next(iter(toks[0].values())))
pairs = []
for t in toks:
    for l in layers:
        if l in t and l + 1 in t:
            sc = co[l][t[l]].sum(0); pairs.append((np.argsort(-sc, kind="stable")[:k].tolist(), t[l + 1], l + 1))
score("co-occurrence L -> L+1", pairs)
pairs = []
for t in toks:
    for l in layers:
        if l in t and l + 1 in t:
            sc = co[l][t[l]].sum(0)
            cand = [e for e in np.argsort(-sc, kind="stable").tolist() if e not in res[l + 1]]
            pairs.append((cand[:k], t[l + 1], l + 1))
score("co-occurrence L -> L+1, top-k non-resident candidates only", pairs)
pairs = []
for r in evl:
    for a, b in zip(r[1], r[1][1:]):
        pairs += [(a[l], b[l], l) for l in layers if l in a and l in b]
score("previous token, same layer", pairs)
pairs = [([e for e in order[l][n_res:n_res + k]], t[l], l) for t in toks for l in layers if l in t]
score("static profile (next-hottest non-resident)", pairs)

for flex in flexes:
    h = m = 0
    for pre, dec in evl:
        cur = {}
        for l in layers:
            core = set(order[l][:n_res - flex])
            cand = sorted(order[l][n_res - flex:], key=lambda e: (-pre[l][e], order[l].index(e)))
            cur[l] = core | set(cand[:flex])
        for t in dec:
            for l, ids in t.items():
                kk = sum(e in cur[l] for e in ids); h += kk; m += len(ids) - kk
    print(f"3. fingerprint, {flex} flex slots of {n_res}: offline decode hit rate {100*h/(h+m):.1f}%")
