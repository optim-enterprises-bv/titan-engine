#!/usr/bin/env python3
"""Rank matmul op classes by the GPU time a llama.cpp-speed kernel would save in the 35B service.

share  = % of GPU kernel time of the op in the MTP=2 service profile (weights.py)
ratio  = ours / llama.cpp at the decode batch the service mostly runs (b3 = MTP verify; b1 also shown)
saving = share * (1 - 1/ratio) when ratio > 1: the % of total GPU kernel time removed if the op ran at
         llama.cpp's speed. Upper-bound tok/s gain if decode were GPU-bound: 1/(1 - saving) - 1.
"""
import json, os

K = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(f"{K}/out/results.json"))
w = json.load(open(f"{K}/out/weights.json"))["share_pct"]


def ratio(op, b):
    d = res.get(f"{op}/b{b}", {})
    L, O = d.get("L", {}).get("nsys_us"), d.get("O", {}).get("nsys_us")
    return (O / L, L, O) if L and O else (None, L, O)


rows = []
for op, share in w.items():
    if "/" in op or not op.startswith("q35."):
        continue
    r3, l3, o3 = ratio(op, 3)
    r1, l1, o1 = ratio(op, 1)
    r = r3 if r3 else r1
    saving = share * (1 - 1 / r) if r and r > 1 else 0.0
    rows.append((saving, op, share, r1, r3, l1, o1, l3, o3))
rows.sort(reverse=True)
tot = sum(r[0] for r in rows)
print("| rank | op (35B) | share of service GPU time | ours/llama b1 | ours/llama b3 | GPU time saved at llama speed |")
print("|---:|---|---:|---:|---:|---:|")
for i, (s, op, sh, r1, r3, *_ ) in enumerate(rows, 1):
    f = lambda x: f"{x:.2f}" if x else "-"
    print(f"| {i} | {op} | {sh:.1f}% | {f(r1)} | {f(r3)} | {s:.1f}% |")
print(f"\nsum of savings: {tot:.1f}% of GPU kernel time -> <= {100 * (1 / (1 - tot / 100) - 1):.0f}% more tok/s if GPU-bound")

# prefill: per 512-token chunk, the whole model's matmuls (40 layers; 30 GDN + 10 attention)
per_layer = {  # op -> launches per layer (all 40 layers unless noted)
    "q35.exp_gate_up.q4_K.k2048n512": 2 * 40, "q35.exp_down.q5_K.k512n2048": 36, "q35.exp_down.q6_K.k512n2048": 4,
    "q35.qkv.q8_0.k2048n8192": 30 + 10, "q35.attn_gate.q8_0.k2048n4096": 30, "q35.ssm_out.q8_0.k4096n2048": 30 + 10,
    "q35.shexp_gate_up.q8_0.k2048n512": 2 * 40 + 2 * 10, "q35.shexp_down.q8_0.k512n2048": 40,
}
for b in (512, 2048, 8192):
    L = O = 0.0
    for op, n in per_layer.items():
        d = res.get(f"{op}/b{b}", {})
        l, o = d.get("L", {}).get("nsys_us"), d.get("O", {}).get("nsys_us")
        if l and o:
            L += n * l
            O += n * o
    if L:
        print(f"prefill b{b}: matmuls per forward: llama.cpp {L/1e3:.1f} ms, ours {O/1e3:.1f} ms ({O/L:.2f}x)"
              f" -> {b / (L/1e6):.0f} vs {b / (O/1e6):.0f} tok/s matmul-only")
