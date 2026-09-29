#!/usr/bin/env python3
"""Extract Q6_K ffn_down_exps tensors from a GGUF into data/ for the kernel gate.
Reads only the header and the requested tensor bytes (never the whole file)."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/titan-engine/m4"))
from qwen35_gguf_shapes import read_header

path = os.path.expanduser("~/ai/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
version, meta, tensors, offsets, data_start = read_header(path)
for layer in sys.argv[1:] or ["1", "34", "38", "39"]:
    name = f"blk.{layer}.ffn_down_exps.weight"
    dims, ty = tensors[name]
    assert ty == "Q6_K", (name, ty)
    k, n, e = dims
    nbytes = n * e * (k // 256) * 210
    with open(path, "rb") as f:
        f.seek(data_start + offsets[name])
        b = f.read(nbytes)
    assert len(b) == nbytes
    dst = os.path.join(out, f"blk.{layer}.ffn_down_exps.q6k")
    with open(dst, "wb") as g:
        g.write(b)
    print(name, dims, ty, nbytes, "->", dst)
