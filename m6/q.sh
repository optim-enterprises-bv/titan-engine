#!/bin/bash
# quick 5-prompt profile runs: q.sh NAME [ENV...]  (MTP file, TITAN_MTP_PROF=wall, tiered timing)
E=$HOME/titan-engine; M=$HOME/ai/models/Qwen3.6-35B-A3B-MTP; F=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
n=$1; shift
PROMPTS=${PROMPTS:-$E/m6/prompts-5.txt} $E/m6/run.sh $n $M $F TITAN_MTP_PROF=wall TITAN_TIERED_TIMING=1 "$@" 2>&1 | grep -v "^gpu\|tiered auto"
grep -o "titan mtp prof.*" $E/m6/out/$n.server.log | tail -1
grep -o "titan mtp stats.*" $E/m6/out/$n.server.log | tail -1
grep -o "titan tiered timing.*" $E/m6/out/$n.server.log | tail -1
