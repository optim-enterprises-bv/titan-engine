#!/bin/bash
E=$HOME/titan-engine; cd $E/m3
./collect.sh q35-clean-auto $E/m4/prompts-eval.txt 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt 2>&1 | grep -v INFO | grep tok/s
grep -o 'titan tiered auto.*' out/q35-clean-auto.server.log | head -1; grep -o 'decode hit rate.*' out/q35-clean-auto.server.log | tail -1
python3 -c "import json;a,b=json.load(open('out/q35-prof.json')),json.load(open('out/q35-clean-auto.json'));print('identical to q35-prof:',sum(x==y for x,y in zip(a,b)),'/',len(a))"
$E/m4/llama-chat.sh
