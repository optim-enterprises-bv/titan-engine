#!/bin/bash
# M6 final: gates + mistral.rs A/B on the final binary (g-*), llama.cpp draft 4 at n-cpu-moe 15 (14 OOMs).
E=$HOME/titan-engine; M=$HOME/ai/models/Qwen3.6-35B-A3B-MTP; F=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
cmp() { python3 -c "import json,sys;a,b=json.load(open(sys.argv[1])),json.load(open(sys.argv[2]));print('GATE',sys.argv[2].split('/')[-1],'vs',sys.argv[1].split('/')[-1],':',sum(x==y for x,y in zip(a,b)),'/',len(a))" $1 $2; }
stats() { grep -o "titan mtp stats.*" $E/m6/out/$1.server.log | tail -2 | head -1; }
unset PROMPTS
$E/m6/run.sh h-oldfile $HOME/ai/models $F; cmp $E/m3/out/q35-prof.json $E/m6/out/h-oldfile.json
$E/m6/run.sh h-off $M $F; cmp $E/m6/out/f-off.json $E/m6/out/h-off.json
for n in 1 2 3 4; do $E/m6/run.sh h-mtp$n $M $F TITAN_MTP=$n; cmp $E/m6/out/h-off.json $E/m6/out/h-mtp$n.json; stats h-mtp$n; done
MBIN=$E/target-mtp-oxide/release/mistralrs $E/m6/run.sh h-oxide-mtp2 $M $F TITAN_MTP=2; cmp $E/m6/out/h-off.json $E/m6/out/h-oxide-mtp2.json
NCPUMOE=15 $E/m6/llama.sh fl15-mtp4 --spec-type draft-mtp --spec-draft-n-max 4
NCPUMOE=15 $E/m6/llama.sh fl15-off
echo SUITE DONE
