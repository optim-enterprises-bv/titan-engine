#!/usr/bin/env python3
"""Extract the first 32 experts of Qwen3.6-35B-A3B UD-Q4_K_XL expert tensors (the service shapes) into
data/ for the kernel gate: gate/up Q4_K k2048 n512, down Q5_K / Q6_K k512 n2048.
Reads only the header and the requested bytes."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/titan-engine/m4"))
from qwen35_gguf_shapes import read_header
BB = {"Q4_K": 144, "Q5_K": 176, "Q6_K": 210}
path = os.path.expanduser("~/ai/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
version, meta, tensors, offsets, data_start = read_header(path)
kinds = {}
for name, (dims, ty) in tensors.items():
    if "_exps.weight" in name:
        kinds.setdefault((name.split(".")[2], ty), []).append(int(name.split(".")[1]))
for key in sorted(kinds):
    print(key, sorted(kinds[key])[:6], len(kinds[key]), "layers")
want = [("ffn_gate_exps", "Q4_K"), ("ffn_up_exps", "Q4_K"), ("ffn_down_exps", "Q5_K"), ("ffn_down_exps", "Q6_K")]
for proj, ty in want:
    if (proj, ty) not in kinds:
        continue
    layer = sorted(kinds[(proj, ty)])[0]
    name = f"blk.{layer}.{proj}.weight"
    dims, t = tensors[name]
    k, n, e = dims
    nbytes = n * 32 * (k // 256) * BB[ty]
    with open(path, "rb") as f:
        f.seek(data_start + offsets[name])
        b = f.read(nbytes)
    assert len(b) == nbytes
    dst = os.path.join(out, f"q35.blk{layer}.{proj}.{ty.lower().replace('_', '')}.k{k}n{n}")
    open(dst, "wb").write(b)
    print(name, dims, ty, nbytes, "->", dst)
