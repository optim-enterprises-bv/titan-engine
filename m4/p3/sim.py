#!/usr/bin/env python3
"""P3 offline cache simulator: GPU expert-slot policies replayed over TITAN_TIERED_TRACE routing, with costs.

Trace lines: `layer batch topk id id ...`, one per MoE forward (gate projection). batch > 1 is prefill.
Each layer is its own cache of C slots (gate/up/down share the routing, so they make the same decisions).
The cache persists across requests, as in the server; requests are replayed in trace order.

Cost model per decode token (ms): base + misses * c_miss + uploads * c_up (+ hits * 0: the GPU gemv is one
launch per projection whatever the hit count). An upload is a synchronous pageable memcpy_htod of the whole
expert (gate+up+down) on the decode thread, as in touch_and_admit / reseed today; `async` rows set c_up = 0
(an upload hidden behind compute, still usable only from the next token).

usage: sim.py CONFIG.json   (see run.sh)
"""
import collections, json, math, sys, time
import numpy as np


# ---------------------------------------------------------------- traces
def read_trace(path):
    """-> list of requests: (prefill [L,E] counts dict layer->Counter, decode ids [T, L, k] int16, layers)."""
    reqs, cur, tok = [], None, None
    for line in open(path):
        f = line.split()
        if len(f) < 4:
            continue
        layer, batch = int(f[0]), int(f[1])
        ids = [int(x) for x in f[3:]]
        if batch > 1:
            if cur is None or cur[1]:
                cur = (collections.defaultdict(collections.Counter), [])
                reqs.append(cur)
            cur[0][layer].update(ids)
            continue
        if cur is None:
            cur = (collections.defaultdict(collections.Counter), [])
            reqs.append(cur)
        if tok is None or not cur[1] or cur[1][-1] is not tok or layer <= max(tok):
            tok = {}
            cur[1].append(tok)
        tok[layer] = ids
    return reqs


def to_arrays(reqs, layers, E):
    L = len(layers)
    li = {l: i for i, l in enumerate(layers)}
    out = []
    for pre, dec in reqs:
        dec = [t for t in dec if len(t) == L]
        if not dec:
            continue
        k = len(next(iter(dec[0].values())))
        a = np.zeros((len(dec), L, k), np.int32)
        for ti, t in enumerate(dec):
            for l, ids in t.items():
                a[ti, li[l]] = ids
        p = np.zeros((L, E), np.float32)
        for l, c in pre.items():
            if l in li:
                for e, n in c.items():
                    p[li[l], e] = n
        out.append((p, a))
    return out


def profile_counts(reqs, layers, E):
    li = {l: i for i, l in enumerate(layers)}
    c = np.zeros((len(layers), E), np.float64)
    for _, dec in reqs:
        for t in dec:
            for l, ids in t.items():
                if l in li:
                    np.add.at(c[li[l]], ids, 1)
    return c


def arr_counts(arrs, E):
    L = arrs[0][1].shape[1]
    c = np.zeros((L, E), np.float64)
    for _, a in arrs:
        for l in range(L):
            np.add.at(c[l], a[:, l].ravel(), 1)
    return c


def profile_file(path, layers, E):
    li = {l: i for i, l in enumerate(layers)}
    c = np.zeros((len(layers), E), np.float64)
    for line in open(path):
        l, e, n = map(int, line.split())
        if l in li:
            c[li[l], e] += n
    return c


def seed_resident(prof, C):
    """C hottest per layer, ties to the lower id (profile_order in titan_tiered.rs)."""
    L, E = prof.shape
    res = np.zeros((L, E), bool)
    for l in range(L):
        order = sorted(range(E), key=lambda e: (-prof[l, e], e))
        res[l, order[:C]] = True
    return res


# ---------------------------------------------------------------- features for the learned policy
DECAYS = (0.5, 0.8, 0.95)
NF = 9


class Feat:
    """Policy-independent per-(layer, expert) state, updated once per decode token."""

    def __init__(self, prof, L, E):
        self.L, self.E = L, E
        lp = np.log1p(prof)
        self.prior = (lp / np.maximum(lp.max(1, keepdims=True), 1e-9)).astype(np.float32)
        self.dec = np.zeros((len(DECAYS), L, E), np.float32)
        self.last = np.full((L, E), -10_000, np.int64)
        self.fp = np.zeros((L, E), np.float32)
        self.now = np.zeros((L, E), np.float32)
        self.t = 0
        self.layer = np.broadcast_to((np.arange(L, dtype=np.float32) / max(L - 1, 1))[:, None], (L, E))
        self.rows = np.arange(L)[:, None]

    def new_request(self, prefill):
        tot = prefill.sum(1, keepdims=True)
        self.fp = (prefill * self.E / np.maximum(tot, 1)).astype(np.float32)  # 1 = uniform share
        self.fp = np.log1p(self.fp)

    def step(self, ids):
        """ids [L, k] of this token."""
        self.t += 1
        self.now[:] = 0
        self.now[self.rows, ids] = 1
        for i, d in enumerate(DECAYS):
            self.dec[i] *= d
            self.dec[i][self.rows, ids] += 1
        self.last[self.rows, ids] = self.t

    def matrix(self):
        """[L, E, NF] float32."""
        rec = np.log1p(np.minimum(self.t - self.last, 4096)).astype(np.float32) / math.log1p(4096)
        cols = [self.prior, *(np.log1p(self.dec[i]) for i in range(len(DECAYS))), rec, self.fp, self.now, self.layer,
                self.prior * self.now]
        return np.stack(cols, -1)


class Model:
    """Logistic regression or a one-hidden-layer MLP on standardized features."""

    def __init__(self, kind, mu, sd, W):
        self.kind, self.mu, self.sd, self.W = kind, mu, sd, W

    def score(self, X):
        Z = (X - self.mu) / self.sd
        if self.kind == "logreg":
            w, b = self.W
            s = Z @ w + b
        else:
            W1, b1, W2, b2 = self.W
            h = np.maximum(Z @ W1 + b1, 0)
            s = h @ W2 + b2
        return 1.0 / (1.0 + np.exp(-s))


def build_dataset(arrs, prof, N, rng, neg_per_layer=24):
    L, E = prof.shape
    F = Feat(prof, L, E)
    Xs, ys = [], []
    # label: (l, e) used again within the next N tokens of the stream. NU[l, e] = next use after token t.
    stream = list(arrs)
    all_ids = np.concatenate([a for _, a in stream], 0)  # [T, L, k]
    T = all_ids.shape[0]
    rows = np.arange(L)[:, None]
    nxt_ev = np.empty(all_ids.shape, np.int64)
    seen = np.full((L, E), 1 << 40, np.int64)
    for tt in range(T - 1, -1, -1):
        nxt_ev[tt] = seen[rows, all_ids[tt]]
        seen[rows, all_ids[tt]] = tt
    NU = seen
    t = 0
    for pre, a in stream:
        F.new_request(pre)
        for ti in range(a.shape[0]):
            F.step(a[ti])
            NU[rows, a[ti]] = nxt_ev[t]
            X = F.matrix()
            neg = rng.integers(0, E, size=(L, neg_per_layer))
            sel = np.concatenate([a[ti], neg], 1)
            Xs.append(X[rows, sel].reshape(-1, NF))
            ys.append(((NU[rows, sel] - t) <= N).astype(np.float32).reshape(-1))
            t += 1
    return np.concatenate(Xs), np.concatenate(ys)


def train_model(kind, X, y, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    if len(y) > 1_000_000:
        keep = np.random.default_rng(seed).choice(len(y), 1_000_000, replace=False)
        X, y = X[keep], y[keep]
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    if kind == "logreg":
        m = LogisticRegression(max_iter=500, C=1.0).fit(Z, y)
        W = (m.coef_[0].astype(np.float32), np.float32(m.intercept_[0]))
    else:
        m = MLPClassifier(hidden_layer_sizes=(16,), max_iter=60, random_state=seed, early_stopping=True,
                          batch_size=4096, learning_rate_init=3e-3).fit(Z, y)
        W = (m.coefs_[0].astype(np.float32), m.intercepts_[0].astype(np.float32),
             m.coefs_[1][:, 0].astype(np.float32), np.float32(m.intercepts_[1][0]))
    return Model(kind, mu.astype(np.float32), sd.astype(np.float32), W)


# ---------------------------------------------------------------- simulation
def simulate(arrs, prof, C, policy, params=None, model=None):
    """-> dict(hits, misses, uploads, tokens, layer_passes)."""
    params = params or {}
    L, E = prof.shape
    rows = np.arange(L)
    res = seed_resident(prof, C) if policy != "empty-lru" else np.zeros((L, E), bool)
    if policy == "empty-lru":
        policy = "lru"
    stream = arrs
    k = stream[0][1].shape[2]
    hits = misses = uploads = passes = tokens = 0
    t = 0
    last = np.zeros((L, E), np.int64)
    lfu = np.zeros((L, E), np.float64)
    W = params.get("window", 16)
    ring = np.zeros((W, L, E), np.int8) if policy == "admit" else None
    F = Feat(prof, L, E) if policy == "learned" else None
    # Belady: next-use times over the whole stream
    if policy == "belady":
        all_ids = np.concatenate([a for _, a in stream], 0)
        T = all_ids.shape[0]
        nxt_ev = np.full((T, L, k), 1 << 40, np.int64)  # next use after this event
        seen = np.full((L, E), 1 << 40, np.int64)
        for tt in range(T - 1, -1, -1):
            ids = all_ids[tt]
            nxt_ev[tt] = seen[rows[:, None], ids]
            seen[rows[:, None], ids] = tt
        NU = seen.copy()  # first use of each (l, e)
    for pre, a in stream:
        if F is not None:
            F.new_request(pre)
        for ti in range(a.shape[0]):
            ids = a[ti]  # [L, k]
            t += 1
            tokens += 1
            was = res[rows[:, None], ids]  # hits against the cache before this token's admissions
            h = int(was.sum())
            hits += h
            misses += ids.size - h
            passes += int((~was).any(1).sum())
            # stats update
            last[rows[:, None], ids] = t
            if policy == "lfu":
                lfu *= params["decay"]
                lfu[rows[:, None], ids] += 1
            if policy == "admit":
                ring[t % W] = 0
                ring[t % W][rows[:, None], ids] = 1
                recent = ring.sum(0)
            if policy == "belady":
                NU[rows[:, None], ids] = nxt_ev[t - 1]
            if F is not None:
                F.step(ids)
                sc = model.score(F.matrix())  # [L, E]
            if policy == "static":
                continue
            used_now = np.zeros((L, E), bool)
            used_now[rows[:, None], ids] = True
            # victim score (lower = evict first); +inf for non-resident / used this token
            if policy == "lru":
                vs = last.astype(np.float64)
            elif policy == "lfu":
                vs = lfu.copy()
            elif policy == "admit":
                vs = last.astype(np.float64)
            elif policy == "learned":
                vs = sc.astype(np.float64).copy()
            elif policy == "belady":
                vs = -NU.astype(np.float64)
            vs[~res | used_now] = np.inf
            for j in range(k):
                c = ids[:, j]
                miss = ~was[:, j]
                if not miss.any():
                    continue
                v = vs.argmin(1)
                vmin = vs[rows, v]
                ok = miss & np.isfinite(vmin)
                if policy == "lfu":
                    ok &= lfu[rows, c] > vmin + params.get("margin", 0.0)
                elif policy == "admit":
                    ok &= recent[rows, c] >= params["k"]
                elif policy == "learned":
                    ok &= sc[rows, c] > vmin + params["margin"]
                elif policy == "belady":
                    ok &= -NU[rows, c] > vmin  # candidate's next use is sooner than the victim's
                if not ok.any():
                    continue
                r = rows[ok]
                res[r, v[ok]] = False
                res[r, c[ok]] = True
                vs[r, v[ok]] = np.inf
                uploads += int(ok.sum())
    return dict(hits=hits, misses=misses, uploads=uploads, tokens=tokens, passes=passes)


def cost(r, cm, layer_scale=1.0):
    """ms/token from a sim result and a cost model {base_ms, c_miss_us, c_up_us}."""
    m = r["misses"] / r["tokens"] * layer_scale
    u = r["uploads"] / r["tokens"] * layer_scale
    return cm["base_ms"] + (m * cm["c_miss_us"] + u * cm["c_up_us"]) / 1000.0


def main():
    cfg = json.load(open(sys.argv[1]))
    rng = np.random.default_rng(0)
    t0 = time.time()
    tr = read_trace(cfg["train"])
    ev = read_trace(cfg["eval"])
    if cfg.get("split"):  # 2-fold by prompt from one trace: even requests train, odd held out
        allr = tr
        tr, ev = allr[0::2], allr[1::2]
        if cfg["split"] == "swap":
            tr, ev = ev, tr
    layers = sorted({l for r in tr + ev for t in r[1] for l in t})
    E = cfg["experts"]
    L = len(layers)
    tra, eva = to_arrays(tr, layers, E), to_arrays(ev, layers, E)
    prof = profile_file(cfg["profile"], layers, E) if cfg.get("profile") else profile_counts(tr, layers, E)
    hind = profile_counts(ev, layers, E)
    C = cfg["slots"]
    scale = cfg.get("layer_scale", 1.0)
    print(f"# {cfg['name']}: train {len(tra)} req / {sum(a.shape[0] for _, a in tra)} tok, held-out {len(eva)} req / "
          f"{sum(a.shape[0] for _, a in eva)} tok, {L} layers, {E} experts, top-{eva[0][1].shape[2]}, {C} slots/layer "
          f"({C / E:.1%}), read in {time.time() - t0:.0f}s", flush=True)

    # cost model: base so the static row reproduces the measured tok/s
    st = simulate(eva, prof, C, "static")
    cms = {}
    for name, cm in cfg["costs"].items():
        cm = dict(cm)
        cm["base_ms"] = 1000.0 / cm["measured_static_tok_s"] - st["misses"] / st["tokens"] * scale * cm["c_miss_us"] / 1000.0
        cms[name] = cm
        print(f"# cost {name}: base {cm['base_ms']:.2f} ms/token, miss {cm['c_miss_us']} us/expert, upload {cm['c_up_us']} us/expert"
              f" (static calibrated to {cm['measured_static_tok_s']} tok/s)", flush=True)
    main_cm = cms[cfg["main_cost"]]

    # Cross-fitting on the train prompts only: the placement profile of one half, tuning/training data from the
    # other half (and vice versa), so a tuned policy sees an out-of-sample profile, as on the held-out prompts.
    halves = (tra[0::2], tra[1::2])
    hprof = [arr_counts(h, E) for h in halves]
    folds = ((hprof[0], halves[1]), (hprof[1], halves[0]))

    def train_cost(policy, p, models=(None, None)):
        tot = ms = 0.0
        for (pf, arr), m in zip(folds, models):
            r = simulate(arr, pf, C, policy, p, m)
            tot += r["tokens"]
            ms += cost(r, main_cm, scale) * r["tokens"]
        return ms / tot

    def tune(policy, grid, models=(None, None)):
        best = min(((train_cost(policy, p, models), i) for i, p in enumerate(grid)))
        return grid[best[1]], best[0]

    rows = []

    def add(label, r, note=""):
        rows.append((label, r, note))
        print(f"{label:48s} hit {r['hits'] / (r['hits'] + r['misses']):6.1%}  uploads/tok {r['uploads'] / r['tokens'] * scale:6.2f}  " +
              "  ".join(f"{n} {cost(r, cm, scale):6.2f} ms" for n, cm in cms.items()) + (f"  [{note}]" if note else ""), flush=True)

    add("(a) static profile (current default)", st)
    add("(b) LRU promotion (profile seed)", simulate(eva, prof, C, "lru"))
    g = [dict(decay=d, margin=m) for d in (0.5, 0.65, 0.8, 0.9, 0.95, 0.98) for m in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0)]
    print(f"# cross-fit train: static {train_cost('static', {}):.2f} ms", flush=True)
    p, _ = tune("lfu", g)
    add("(c) LFU with decay", simulate(eva, prof, C, "lfu", p), f"tuned on train: {p}")
    g = [dict(k=kk, window=w) for w in (4, 8, 16, 32) for kk in (2, 3, 4, 6, 8, 12, 16, 24) if kk <= w]
    p, _ = tune("admit", g)
    add("(d) static + admission (k hits in window)", simulate(eva, prof, C, "admit", p), f"tuned on train: {p}")
    for kind in ("logreg", "mlp"):
        best = None
        for N in cfg.get("horizons", (4, 16, 64)):
            # fold models (for tuning the margin out of sample) and the final model on all train data
            fm = []
            for pf, arr in folds:
                X, y = build_dataset(arr, pf, N, rng)
                fm.append(train_model(kind, X, y))
            p, c = tune("learned", [dict(margin=mm) for mm in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)], (fm[1], fm[0]))  # each fold scored by the model of the other
            Xa = [build_dataset(arr, pf, N, rng) for pf, arr in folds]
            X, y = np.concatenate([x for x, _ in Xa]), np.concatenate([v for _, v in Xa])
            m = train_model(kind, X, y)
            print(f"#   {kind} N={N}: {len(y)} samples, positive {y.mean():.1%}, cross-fit train {c:.2f} ms at {p}", flush=True)
            if best is None or c < best[0]:
                best = (c, N, p, m)
        _, N, p, m = best
        add(f"(e) learned {kind} (reuse within {N} tok)", simulate(eva, prof, C, "learned", p, m), f"tuned on train: {p}")
        if kind == "logreg":
            np.savez(cfg["out"] + f".{kind}.npz", mu=m.mu, sd=m.sd, w=m.W[0], b=m.W[1], N=N, margin=p["margin"])
    add("(f) Belady (next use, bypass), upper bound", simulate(eva, prof, C, "belady"), "hit-rate optimal, not cost optimal")
    add("    static, hindsight profile of held-out", simulate(eva, hind, C, "static"), "best static placement bound")
    json.dump([dict(label=l, note=n, **r, ms={k: cost(r, cm, scale) for k, cm in cms.items()}) for l, r, n in rows],
              open(cfg["out"] + ".json", "w"), indent=1)
    print(f"# done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
