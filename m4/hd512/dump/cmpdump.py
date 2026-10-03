#!/usr/bin/env python3
"""cmpdump.py TITAN_DIR LLAMA_DIR [LLAMA2_DIR]: per layer, relative L2 error of titan's last-token rows (kqv_out,
attn_out, l_out) against llama.cpp's; with LLAMA2_DIR also llama-vs-llama (e.g. bf16 KV or FA off) for scale."""
import os, sys, numpy as np
t, l = sys.argv[1], sys.argv[2]
l2 = sys.argv[3] if len(sys.argv) > 3 else None
def rd(d, n):
    f = os.path.join(d, n + ".f32")
    return np.fromfile(f, dtype=np.float32).astype(np.float64) if os.path.exists(f) else None
def rel(a, b):
    if a is None or b is None or a.shape != b.shape: return float("nan")
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
print("layer  " + "  ".join(f"{n:>9}" + (f" {n+'(ll)':>11}" if l2 else "") for n in ("kqv_out", "attn_out", "l_out")))
for i in range(64):
    if rd(l, f"l_out-{i}") is None: break
    row = []
    for n in ("kqv_out", "attn_out", "l_out"):
        row.append(f"{rel(rd(t, f'{n}-{i}'), rd(l, f'{n}-{i}')):9.2e}")
        if l2: row.append(f"{rel(rd(l2, f'{n}-{i}'), rd(l, f'{n}-{i}')):11.2e}")
    print(f"{i:5d}  " + "  ".join(row))
