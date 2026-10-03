#!/usr/bin/env python3
"""tokmap_gguf.py MODEL.gguf FILE.json...: m4/g4s/tokmap.py for any byte-level-BPE GGUF vocabulary (qwen35):
titan's top_logprobs carry token ids; rewrite them in place as decoded pieces (llama.cpp reports pieces)."""
import json, sys, types
sys.modules.setdefault("yaml", types.ModuleType("yaml")); sys.path.insert(0, "~/ai/llama.cpp/gguf-py")
from gguf import GGUFReader
f = GGUFReader(sys.argv[1]).fields["tokenizer.ggml.tokens"]
vocab = [bytes(f.parts[i]).decode("utf-8") for i in f.data]
bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)); cs = bs[:]; n = 0
for x in range(256):
    if x not in bs: bs.append(x); cs.append(256 + n); n += 1
inv = {chr(c): x for x, c in zip(bs, cs)}
def dec(t):
    return bytes(inv.get(ch, 63) for ch in vocab[t]).decode("utf-8", "replace") if not vocab[t].startswith("<|") else vocab[t]
for fn in sys.argv[2:]:
    rows = json.load(open(fn)); n = 0
    for r in rows:
        for step in r.get("top", []):
            for pair in step:
                if isinstance(pair[0], int):
                    pair[0] = dec(pair[0]); n += 1
    json.dump(rows, open(fn, "w"), ensure_ascii=False)
    print(fn, "mapped", n)
