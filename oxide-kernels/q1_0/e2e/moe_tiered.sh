#!/bin/bash
# Q1_0 MoE test model through mistral.rs TieredExperts: GPU fraction 1 / 0.5 / 0.5 permuted / 0 must give
# byte-identical greedy completions (GPU kernel and CPU twin bit-identical).
cd "$(dirname "$0")"
D=$HOME/titan-engine/m1/data; F=qwen3coder30b-first24-q1_0exps.gguf
for cfg in "1 0" "0.5 0" "0.5 1" "0 0"; do
  set -- $cfg
  since=$(date '+%Y-%m-%d %H:%M:%S')
  ./run_mistral.sh 18183 $D $F --setenv=TITAN_TIERED=1 --setenv=TITAN_TIERED_GPU_FRACTION=$1 --setenv=TITAN_TIERED_PROBE=$2 || { echo "start failed"; exit 1; }
  journalctl --user -u q1-mistral --no-pager --since "$since" | grep -m1 "titan tiered experts" | sed 's/.*titan_tiered: //'
  timeout 1200 python3 query.py http://127.0.0.1:18183 moe-f$1-p$2.json > /dev/null 2>&1; echo "fraction=$1 probe=$2 query rc=$?"
  systemctl --user stop q1-mistral
done
python3 - <<'PY'
import json, glob
base = json.load(open('moe-f1-p0.json'))
for f in sorted(glob.glob('moe-f*.json')):
    r = json.load(open(f))
    same = all(x['text'] == y['text'] and x['top4'] == y['top4'] for x, y in zip(base, r))
    print(f, 'identical to fraction 1:', same, 'decode tok/s', [round(x['usage'].get('avg_compl_tok_per_sec', 0), 1) for x in r])
print(repr(base[0]['text'][:100]))
PY
