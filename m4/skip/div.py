#!/usr/bin/env python3
"""div.py REF NAME...: per prompt, first token position where out/NAME.json diverges from out/REF.json (texts
re-tokenized with the Qwen2.5 BPE, same vocabulary as Qwen3-Next); 64 = identical through the 64-token cap."""
import glob, json, statistics, sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(glob.glob("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/tokenizer.json")[0])
ref = json.load(open(f"out/{sys.argv[1]}.json"))
for n in sys.argv[2:]:
    b = json.load(open(f"out/{n}.json"))
    pos = []
    for x, y in zip(ref, b):
        tx, ty = tok.encode(x).ids, tok.encode(y).ids
        d = next((i for i, (p, q) in enumerate(zip(tx, ty)) if p != q), min(len(tx), len(ty)))
        pos.append(64 if x == y else d)
    print(f"{n} vs {sys.argv[1]}: identical {sum(x == y for x, y in zip(ref, b))}/{len(ref)}, first divergence median {statistics.median(pos):.0f}, "
          f"mean {statistics.mean(pos):.1f}, <8: {sum(p < 8 for p in pos)}, <32: {sum(p < 32 for p in pos)}")
