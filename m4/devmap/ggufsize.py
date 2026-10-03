#!/usr/bin/env python3
"""Per-layer / non-layer byte inventory of a GGUF (exact file bytes per tensor), plus key metadata.
usage: ggufsize.py FILE.gguf [--types]"""
import sys, re, collections
sys.path.insert(0, "~/ai/llama.cpp/gguf-py")
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
arch = bytes(r.fields["general.architecture"].parts[-1]).decode()
def md(k):
    f = r.fields.get(f"{arch}.{k}")
    if f is None: return None
    v = f.parts[f.data[0]] if len(f.data) == 1 else [f.parts[i][0] for i in f.data]
    return v.tolist()[0] if hasattr(v, "tolist") and len(f.data) == 1 else (v if not hasattr(v, "tolist") else v.tolist())
layers = collections.defaultdict(int); other = collections.defaultdict(int); types = collections.Counter()
for t in r.tensors:
    m = re.match(r"blk\.(\d+)\.", t.name)
    if m: layers[int(m.group(1))] += int(t.n_bytes)
    else: other[t.name] += int(t.n_bytes)
    types[(t.tensor_type.name, t.name.split(".")[-2] if m else t.name)] += 1
L = sum(layers.values()); O = sum(other.values())
print(f"arch {arch}; block_count {md('block_count')}; embd {md('embedding_length')}; ffn {md('feed_forward_length')}; heads {md('attention.head_count')}; kv_heads {md('attention.head_count_kv')}; key_len {md('attention.key_length')}; val_len {md('attention.value_length')}; swa {md('attention.sliding_window')}")
print(f"layers: {len(layers)}, sum {L/2**20:.1f} MiB, min {min(layers.values())/2**20:.1f} max {max(layers.values())/2**20:.1f} MiB")
print("non-layer:", ", ".join(f"{k} {v/2**20:.1f}" for k, v in other.items()), f"= {O/2**20:.1f} MiB")
print(f"total {(L+O)/2**20:.1f} MiB")
if "--types" in sys.argv:
    for (ty, n), c in sorted(types.items()): print(f"  {ty:8s} {n:30s} x{c}")
