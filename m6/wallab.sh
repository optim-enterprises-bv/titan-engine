#!/bin/bash
# fixed-fraction wall-time A/B of decode vs verify forwards (5 prompts)
E=$HOME/titan-engine; M=$HOME/ai/models/Qwen3.6-35B-A3B-MTP; F=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
export PROMPTS=$E/m6/prompts-5.txt
FR=${FR:-0.55}
for cfg in "$@"; do
  n=wab-$cfg
  case $cfg in nod) extra="TITAN_MTP=1 TITAN_MTP_NODRAFT=1";; *) extra="TITAN_MTP=$cfg";; esac
  $E/m6/run.sh $n $M $F $extra TITAN_MTP_PROF=wall TITAN_TIERED_GPU_FRACTION=$FR 2>&1 | grep -E "tok/s"
  grep -o "titan mtp prof.*" $E/m6/out/$n.server.log | tail -1
  grep -o "titan mtp stats.*" $E/m6/out/$n.server.log | tail -1
done
