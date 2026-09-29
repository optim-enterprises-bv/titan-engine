#!/bin/bash
# MoE test model (qwen3coder30b-first24, experts requantized to <fmt>) through mistral.rs TieredExperts:
# GPU fraction 1 / 0.5 / 0.5 permuted / 0 must give byte-identical greedy completions (GPU kernel and
# CPU twin bit-identical).  usage: moe_tiered.sh <fmt>   (e.g. iq4_nl; model qwen3coder30b-first24-<fmt>exps.gguf)
cd "$(dirname "$0")"
FMT=${1:?fmt}
D=$HOME/titan-engine/m1/data; F=qwen3coder30b-first24-${FMT}exps.gguf
mkdir -p out
for cfg in "1 0" "0.5 0" "0.5 1" "0 0"; do
  set -- $cfg
  since=$(date '+%Y-%m-%d %H:%M:%S')
  ./run_mistral.sh 18283 $D $F --setenv=TITAN_TIERED=1 --setenv=TITAN_TIERED_GPU_FRACTION=$1 --setenv=TITAN_TIERED_PROBE=$2 || { echo "start failed"; journalctl --user -u fmt-mistral --no-pager --since "$since" | grep -v INFO | tail -20; exit 1; }
  journalctl --user -u fmt-mistral --no-pager --since "$since" | grep -m1 "titan tiered experts" | sed 's/.*titan_tiered: //'
  timeout 1200 python3 query.py http://127.0.0.1:18283 out/${FMT}-moe-f$1-p$2.json > /dev/null 2>&1; echo "fraction=$1 probe=$2 query rc=$?"
  systemctl --user stop fmt-mistral
done
FMT=$FMT python3 - <<'PY'
import json, glob, os
fmt = os.environ["FMT"]
base = json.load(open(f'out/{fmt}-moe-f1-p0.json'))
for f in sorted(glob.glob(f'out/{fmt}-moe-f*.json')):
    r = json.load(open(f))
    same = all(x['text'] == y['text'] and x['top4'] == y['top4'] for x, y in zip(base, r))
    print(f, 'identical to fraction 1:', same, 'decode tok/s', [round(x['usage'].get('avg_compl_tok_per_sec', 0), 1) for x in r])
print(repr(base[0]['text'][:100]))
PY
