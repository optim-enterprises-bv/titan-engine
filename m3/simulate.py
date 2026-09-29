#!/usr/bin/env python3
"""Replay titan-engine routing traces against GPU expert-cache policies.

Trace lines (TITAN_TIERED_TRACE): `layer batch topk id id ...`, one per MoE forward.
A forward with batch > 1 is prefill; batch == 1 is a decode step. Requests are split
where a prefill line for layer 0 appears.

Usage: simulate.py CALIB_TRACE EVAL_TRACE SLOTS_PER_LAYER [SLOTS ...]
CALIB builds the static profile; EVAL is held out. Hit rate counts decode-step
expert uses served from the GPU cache (prefill streams every expert anyway).
"""
import collections
import sys


def read(path):
    requests, cur = [], []
    for line in open(path):
        f = line.split()
        if len(f) < 4:
            continue
        layer, batch, topk = int(f[0]), int(f[1]), int(f[2])
        ids = list(map(int, f[3:]))
        if layer == 0 and batch > 1 and cur:
            requests.append(cur)
            cur = []
        cur.append((layer, batch, topk, ids))
    if cur:
        requests.append(cur)
    return requests


def static_profile(requests):
    counts = collections.defaultdict(collections.Counter)
    for req in requests:
        for layer, batch, _, ids in req:
            if batch == 1:
                counts[layer].update(ids)
    return counts


def simulate(requests, policy, slots, profile):
    hits = total = 0
    for req in requests:
        cache = {}  # layer -> ordered dict expert -> last use tick
        tick = 0
        for layer, batch, topk, ids in req:
            c = cache.get(layer)
            if c is None:
                seed = [e for e, _ in profile[layer].most_common(slots)] if policy in ("static", "static+lru", "prefill+lru") else []
                c = cache[layer] = collections.OrderedDict((e, 0) for e in seed)
            if batch > 1:
                if policy == "prefill+lru":
                    # re-seed from this request's own prefill routing counts
                    top = [e for e, _ in collections.Counter(ids).most_common(slots)]
                    c.clear()
                    for e in top:
                        c[e] = tick
                continue
            for e in ids:
                total += 1
                tick += 1
                if e in c:
                    hits += 1
                    if policy != "static":
                        c.move_to_end(e)
                elif policy != "static":
                    c[e] = tick
                    if len(c) > slots:
                        c.popitem(last=False)
    return hits / max(total, 1), total


def main():
    calib, evalt = read(sys.argv[1]), read(sys.argv[2])
    profile = static_profile(calib)
    layers = sorted({l for r in evalt for l, *_ in r})
    num_experts = max(max(ids) for r in evalt for *_, ids in r) + 1
    print(f"calib {len(calib)} requests, eval {len(evalt)} requests, {len(layers)} layers, {num_experts} experts")
    for slots in map(int, sys.argv[3:]):
        row = [f"slots/layer {slots:3d} ({slots / num_experts:5.1%})"]
        for policy in ("lru", "static", "static+lru", "prefill+lru"):
            hr, n = simulate(evalt, policy, slots, profile)
            row.append(f"{policy} {hr:6.1%}")
        print("  ".join(row), f"({n} decode expert uses)")


if __name__ == "__main__":
    main()
