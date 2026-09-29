#!/bin/bash
# M6 gate + A/B suite, back to back on an otherwise idle GPU
E=$HOME/titan-engine; M=$HOME/ai/models/Qwen3.6-35B-A3B-MTP; F=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
cmp() { python3 -c "import json,sys;a,b=json.load(open(sys.argv[1])),json.load(open(sys.argv[2]));print(sys.argv[2].split('/')[-1],'vs',sys.argv[1].split('/')[-1],':',sum(x==y for x,y in zip(a,b)),'/',len(a))" $1 $2; }
unset PROMPTS
$E/m6/run.sh g-off $M $F
for n in 1 2 3 4; do $E/m6/run.sh g-mtp$n $M $F TITAN_MTP=$n; cmp $E/m6/out/g-off.json $E/m6/out/g-mtp$n.json; done
$E/m6/run.sh g-oldfile $HOME/ai/models $F; cmp $E/m3/out/q35-prof.json $E/m6/out/g-oldfile.json
export PROMPTS=$E/m6/prompts-5.txt
for c in "s-nod TITAN_MTP_NODRAFT=1" "s-m2" "s-nod-skip TITAN_MTP_NODRAFT=1 TITAN_TIERED_DEBUG_SKIP_MISSES=1" "s-m2-skip TITAN_TIERED_DEBUG_SKIP_MISSES=1"; do
  set -- $c; n=$1; shift
  $E/m6/run.sh $n $M $F TITAN_MTP=2 TITAN_MTP_PROF=wall "$@"; grep -o "titan mtp prof.*" $E/m6/out/$n.server.log | tail -1
done
unset PROMPTS
$E/m6/llama.sh l-off
for n in 1 2 3; do $E/m6/llama.sh l-mtp$n --spec-type draft-mtp --spec-draft-n-max $n; done
echo SUITE DONE
