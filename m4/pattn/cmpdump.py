#!/usr/bin/env python3
"""cmpdump.py N A_DIR B_DIR LAYER [BIN]: compare the last N dumped attention-output rows (positions 0..N-1 of the last
prompt) of LAYER between two TITAN_PATTN_DUMP dirs: relative L2 error per position, summarised per BIN-position band
(default 512), and the first position whose error exceeds 0.05."""
import sys, struct
import numpy as np
def load(d, layer):
    raw = open(f"{d}/L{layer}.bin", "rb").read()
    out, i = [], 0
    while i < len(raw):
        rows, width = struct.unpack_from("<II", raw, i); i += 8
        out.append(np.frombuffer(raw, dtype="<f4", count=rows * width, offset=i).reshape(rows, width)); i += rows * width * 4
    return np.concatenate(out, 0)
n, a, b, layer = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
binw = int(sys.argv[5]) if len(sys.argv) > 5 else 512
A, B = load(a, layer)[-n:], load(b, layer)[-n:]
err = np.linalg.norm(A - B, axis=1) / np.maximum(np.linalg.norm(B, axis=1), 1e-12)
bad = np.nonzero(err > 0.05)[0]
print(f"L{layer} {a.split('/')[-1]} vs {b.split('/')[-1]}: {n} rows; first row with rel err > 0.05: {bad[0] if len(bad) else None}; "
      f"rows > 0.05: {len(bad)}; max {err.max():.4f}")
print("   band medians: " + " ".join(f"[{s}:{min(s + binw, n)}) {np.median(err[s:s + binw]):.4f}" for s in range(0, n, binw)))
