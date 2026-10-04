source $(dirname $0)/tmcmp.sh true
Q=$HOME/ai/models/Qwen3.6-35B-A3B-MTP/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
for f in "$@"; do cp ../$f.json . ; done
tm gguf:$Q $(for f in "$@"; do echo $f.json; done)
python3 $(dirname $0)/../../sp.py $(for f in "$@"; do echo $f.json; done)
for f in "$@"; do g=${f##*-}; echo "$f vs llama: $(cmpg $g $f.json ../lref-q35-$g.json)"; done
python3 - "$@" <<'PY'
import json,sys
L=json.load(open("../lref-q35-g3.json"))
for f in sys.argv[1:]:
    if not f.endswith("g3"): continue
    T=json.load(open(f+".json"))
    print(f, "prompt 3:", [(x[0], round(x[1],3)) for x in T[3]["top"][0][:3]])
print("llama prompt 3:", [(x[0], round(x[1],3)) for x in L[3]["top"][0][:3]])
PY
