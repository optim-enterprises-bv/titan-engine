#!/usr/bin/env python3
"""Layer-by-layer diff of the prompt pass: llama.cpp llama-eval-callback (l_out-N) vs mistral.rs TITAN_LAYER_DUMP.
usage: layerdiff.py EVALCB_TXT MISTRAL_DUMP"""
import re, sys
def llama(path):
    lines = open(path).read().split('\n'); out = {}; i = 0
    while i < len(lines):
        m = re.search(r'\bl_out-(\d+) = ', lines[i])
        if m:
            layer = int(m.group(1)); vals = []; i += 1
            while i < len(lines) and 'sum =' not in lines[i]:
                row = re.findall(r'-?\d+\.\d+', lines[i])
                if len(row) == 6: vals.extend(map(float, row))
                i += 1
            out[layer] = (float(lines[i].split('=')[1]), vals)
        i += 1
    return out
def mistral(path):
    out = {}
    for line in open(path):
        f = line.split(); out.setdefault(int(f[0]), (float(f[1]), list(map(float, f[2:]))))
    return out
a, b = llama(sys.argv[1]), mistral(sys.argv[2])
print(f"{'layer':>5} {'sum llama':>11} {'sum mistral':>11} {'max |diff| sampled':>18} {'rel sum diff':>12}")
worst = 0
for l in sorted(set(a) & set(b)):
    (sa, va), (sb, vb) = a[l], b[l]
    md = max((abs(x - y) for x, y in zip(va, vb)), default=float('nan'))
    rel = abs(sa - sb) / max(abs(sa), 1e-6); worst = max(worst, md)
    print(f"{l:5d} {sa:11.4f} {sb:11.4f} {md:18.4f} {rel:12.2e}")
print(f"layers compared: {len(set(a) & set(b))}; worst sampled |diff| {worst:.4f}")
