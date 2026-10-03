#!/usr/bin/env python3
"""kern.py NCU_RAW.csv [MIN_ROW] -- GPU time per kernel name (ms, share, launches), top 40, from an
`ncu --import X --csv --page raw` export (gpu__time_duration.sum); rows before MIN_ROW (warm-up) skipped,
plus a coarse class table (MoE experts / attention / dense matmul / norms+elementwise / other)."""
import csv, re, sys, collections
rd = list(csv.reader(open(sys.argv[1])))
hdr, unit, data = rd[0], rd[1], rd[2:]
data = data[int(sys.argv[2]) if len(sys.argv) > 2 else 0:]
ci = {h: i for i, h in enumerate(hdr)}
t = ci["gpu__time_duration.sum"]
sc = {"ns": 1e-6, "nsecond": 1e-6, "us": 1e-3, "usecond": 1e-3, "ms": 1.0, "msecond": 1.0, "s": 1e3}[unit[t]]
agg = collections.defaultdict(lambda: [0.0, 0])
for r in data:
    n = re.sub(r"\(.*", "", r[ci["Kernel Name"]]).replace("void ", "").strip()
    agg[n][0] += float(r[t]) * sc
    agg[n][1] += 1
tot = sum(v[0] for v in agg.values())
print(f"total GPU {tot:.1f} ms in {sum(v[1] for v in agg.values())} launches")
for n, (ms, c) in sorted(agg.items(), key=lambda x: -x[1][0])[:40]:
    print(f"{ms:9.1f} ms {100 * ms / tot:5.1f}% {c:7d}  {n[:110]}")
def cls(n):
    if re.search(r"moe|mmq_.*_x|indexed|dispatch|weighted_reduce|iq4_nl_mmq|quantize_mmq_glu|quantize_q8_1", n): return "MoE experts (+ their activation quant)"
    if re.search(r"flash|softmax|attn|gemm|Kernel|cutlass|sm\d\d_|badd|bmm|rotary|rope", n): return "attention (incl. cuBLAS)"
    if re.search(r"mmq|mmvq|gemv|quantize", n): return "dense quantized matmul"
    if re.search(r"rms|norm|affine|binary|unary|cast|copy|fill|where|gelu|silu|add|mul", n): return "norms / elementwise / copies"
    return "other"
c = collections.defaultdict(float)
for n, (ms, _) in agg.items():
    c[cls(n)] += ms
for k, v in sorted(c.items(), key=lambda x: -x[1]):
    print(f"CLASS {v:9.1f} ms {100 * v / tot:5.1f}%  {k}")
