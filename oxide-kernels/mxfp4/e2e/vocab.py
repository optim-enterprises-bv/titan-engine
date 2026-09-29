#!/usr/bin/env python3
"""vocab.py <gguf> <out.json>: tokenizer.ggml.tokens as a JSON list (for decoding mistral.rs top-logprob ids)."""
import json, struct, sys
def rd(f, fmt): return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))
def rstr(f):
    (n,) = rd(f, "Q"); return f.read(n).decode("utf-8", "replace")
def rval(f, t):
    m = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    if t in m: return rd(f, m[t])[0]
    if t == 8: return rstr(f)
    if t == 9:
        (at, n) = rd(f, "IQ"); return [rval(f, at) for _ in range(n)]
    raise ValueError(t)
with open(sys.argv[1], "rb") as f:
    assert f.read(4) == b"GGUF"
    _, nt, nkv = rd(f, "IQQ")
    for _ in range(nkv):
        k = rstr(f); (t,) = rd(f, "I"); v = rval(f, t)
        if k == "tokenizer.ggml.tokens":
            json.dump(v, open(sys.argv[2], "w"), ensure_ascii=False); print(len(v), "tokens"); break
