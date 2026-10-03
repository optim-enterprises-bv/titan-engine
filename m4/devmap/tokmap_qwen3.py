# Titan's top_logprobs carry token ids; rewrite them in place as decoded pieces (llama.cpp reports pieces), qwen3
# vocabulary (byte-level BPE from the GGUF; decoding a single id needs no pre-tokenizer). usage: tokmap_qwen3.py FILE.json...
import sys, json
sys.path.insert(0, "~/titan-engine/m4/g4s/tok")
import proto
from tokenizers import Tokenizer, models, decoders
toks, merges, types, md = proto.load("~/ai/models/qwen3-14b/Qwen3-14B-vanilla-Q5_K_M.gguf")
tok = Tokenizer(models.BPE({t: i for i, t in enumerate(toks)}, [tuple(x.split(" ", 1)) for x in merges], ignore_merges=False))
tok.decoder = decoders.ByteLevel()
proto.specials(tok, toks, types)
for f in sys.argv[1:]:
    rows = json.load(open(f))
    n = 0
    for r in rows:
        for step in r.get("top", []):
            for pair in step:
                if isinstance(pair[0], int):
                    pair[0] = tok.decode([pair[0]], skip_special_tokens=False); n += 1
    json.dump(rows, open(f, "w"), ensure_ascii=False)
    print(f, n, "ids mapped")
