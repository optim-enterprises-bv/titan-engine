#!/bin/bash
# M6: 40 eval prompts x 256 tokens through m6/collect.sh with the M6 binary, under a memory cap.
# usage: run.sh NAME FILE_DIR FILE [ENV...]
E=$HOME/titan-engine; n=$1 dir=$2 file=$3; shift 3
while [ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -lt 13 ]; do sleep 20; done
# shared GPU: wait (up to 90 min) until no other process holds VRAM
for i in $(seq 1 540); do [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) -lt 700 ] && break; sleep 10; done
# quiet CPU: wait (up to 30 min) for the 1-minute load average to drop below 2
for i in $(seq 1 180); do awk '{exit !($1 < 2)}' /proc/loadavg && break; sleep 10; done
echo "gpu used before: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"; echo "loadavg before $n: $(cat /proc/loadavg)"
for try in 1 2 3 4 5 6; do
systemd-run --user --unit=mtp-run-$n --collect --wait -q -p MemoryMax=14G -p MemorySwapMax=0 -p WorkingDirectory=$E/m3 \
  -p StandardOutput=truncate:$E/m6/$n.log -p StandardError=truncate:$E/m6/$n.log \
  env BIN=${MBIN:-$E/target-mtp/release/mistralrs} DIR=$dir FILE=$file OUT=$E/m6/out $E/m6/collect.sh $n ${PROMPTS:-$E/m4/prompts-eval.txt} 256 \
  TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt "$@"
  grep -q OUT_OF_MEMORY $E/m6/out/$n.server.log || break
  echo "CUDA OOM (shared GPU), retry $try"; sleep 60
done
echo "loadavg after $n: $(cat /proc/loadavg)"
grep -v INFO $E/m6/$n.log | grep -E "tok/s|died|rror" | head -3
grep -o 'titan tiered auto.*' $E/m6/out/$n.server.log | head -1
grep -o 'titan mtp.*' $E/m6/out/$n.server.log | tail -2
