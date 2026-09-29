#!/bin/bash
E=$HOME/titan-engine; M=$HOME/ai/models/Qwen3.6-35B-A3B-MTP; F=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
export PROMPTS=$E/m6/prompts-5.txt
r() { n=$1; shift; $E/m6/run.sh $n $M $F "$@" TITAN_MTP_PROF=wall 2>&1 | grep -E "tok/s|auto|gpu used"; grep -o "titan mtp prof.*" $E/m6/out/$n.server.log | tail -1; grep -o "titan mtp stats.*" $E/m6/out/$n.server.log | tail -1; }
r x-nod TITAN_MTP=2 TITAN_MTP_NODRAFT=1
r x-m2 TITAN_MTP=2
r x-nod-skip TITAN_MTP=2 TITAN_MTP_NODRAFT=1 TITAN_TIERED_DEBUG_SKIP_MISSES=1
r x-m2-skip TITAN_MTP=2 TITAN_TIERED_DEBUG_SKIP_MISSES=1
