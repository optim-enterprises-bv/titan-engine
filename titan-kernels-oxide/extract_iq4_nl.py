#!/usr/bin/env python3
"""Extract IQ4_NL (GGML type 20) expert tensors from the requantized test model into data/ for the
kernel gate. The model is qwen3coder30b-first24.gguf with every *_exps tensor requantized to IQ4_NL:
  llama-quantize --allow-requantize --tensor-type ffn_gate_exps=iq4_nl --tensor-type ffn_up_exps=iq4_nl \
      --tensor-type ffn_down_exps=iq4_nl qwen3coder30b-first24.gguf qwen3coder30b-first24-iq4_nlexps.gguf Q4_K_M 6
Reads only the header and the requested tensor bytes."""
import os, struct, sys

path = os.path.expanduser("~/titan-engine/m1/data/qwen3coder30b-first24-iq4_nlexps.gguf")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def rd(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))
def rstr(f):
    (n,) = rd(f, "Q"); return f.read(n).decode("utf-8", "replace")
def rval(f, t):
    m = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    if t in m: return rd(f, m[t])[0]
    if t == 8: return rstr(f)
    if t == 9:
        (at, n) = rd(f, "IQ"); return [rval(f, at) for _ in range(n)]
    raise ValueError(t)

with open(path, "rb") as f:
    assert f.read(4) == b"GGUF"
    _, nt, nkv = rd(f, "IQQ")
    kv = {}
    for _ in range(nkv):
        k = rstr(f); (t,) = rd(f, "I"); kv[k] = rval(f, t)
    tensors = {}
    for _ in range(nt):
        name = rstr(f); (nd,) = rd(f, "I"); dims = rd(f, "Q" * nd); ty, off = rd(f, "IQ")
        tensors[name] = (dims, ty, off)
    al = kv.get("general.alignment", 32)
    base = (f.tell() + al - 1) // al * al
    for name in sys.argv[1:] or ["blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight", "blk.23.ffn_up_exps.weight"]:
        dims, ty, off = tensors[name]
        assert ty == 20, (name, ty)
        k, n, e = dims
        nbytes = e * n * (k // 32) * 18
        f.seek(base + off)
        b = f.read(nbytes)
        assert len(b) == nbytes
        dst = os.path.join(out, name.replace(".weight", ".iq4_nl"))
        open(dst, "wb").write(b)
        print(name, dims, nbytes, "->", dst)
