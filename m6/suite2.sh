#!/bin/bash
# M6 gates + A/B on the final binary, back to back (run.sh waits for a quiet CPU/GPU before each run).
E=$HOME/titan-engine; M=$HOME/ai/models/Qwen3.6-35B-A3B-MTP; F=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
cmp() { python3 -c "import json,sys;a,b=json.load(open(sys.argv[1])),json.load(open(sys.argv[2]));print('GATE',sys.argv[2].split('/')[-1],'vs',sys.argv[1].split('/')[-1],':',sum(x==y for x,y in zip(a,b)),'/',len(a))" $1 $2; }
stats() { grep -o "titan mtp stats.*" $E/m6/out/$1.server.log | tail -1; }
unset PROMPTS
$E/m6/run.sh f-oldfile $HOME/ai/models $F; cmp $E/m3/out/q35-prof.json $E/m6/out/f-oldfile.json
$E/m6/run.sh f-off $M $F
for n in 1 2 3 4; do $E/m6/run.sh f-mtp$n $M $F TITAN_MTP=$n; cmp $E/m6/out/f-off.json $E/m6/out/f-mtp$n.json; stats f-mtp$n; done
$E/m6/run.sh f-nodraft $M $F TITAN_MTP=2 TITAN_MTP_NODRAFT=1; cmp $E/m6/out/f-off.json $E/m6/out/f-nodraft.json
NCPUMOE=${NCPUMOE:-18}
$E/m6/llama.sh fl-off
for n in 1 2 3 4; do $E/m6/llama.sh fl-mtp$n --spec-type draft-mtp --spec-draft-n-max $n; done
echo SUITE DONE
