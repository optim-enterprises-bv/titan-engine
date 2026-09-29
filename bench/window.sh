#!/bin/bash
# GPU window for one bench run: service gap, stop titan-mistral, run the suite, restart titan-mistral on exit.
# The caller holds the GPU lock (the main session's gpu-queue.sh flocks ~/titan-engine/.gpu.lock); this script
# does not flock. Usage: bench/window.sh <binary> <label>
set -u
E=$HOME/titan-engine; D=$E/bench
BIN=${1:?usage: window.sh <binary> <label>}; LABEL=${2:?usage: window.sh <binary> <label>}
mkdir -p $D/runs
exec > >(tee -a $D/runs/window-$(date +%Y%m%d-%H%M)-$LABEL.log) 2>&1
echo "== bench window $LABEL: lock held, $(date -Is)"
# 30-minute service gap (RULES-agents.md 2026-09-28 15:10); ~/titan-engine/.nogap turns it off, as in the other scripts.
idle=0
while :; do
  if [ "$(systemctl --user is-active titan-mistral)" != active ]; then
    idle=$((idle + 1)); [ $idle -ge 20 ] && { echo "titan-mistral inactive for 10 min: starting it"; systemctl --user start titan-mistral; idle=0; }
  else
    t=$(systemctl --user show titan-mistral -p ActiveEnterTimestampMonotonic --value)
    now=$(awk '{print int($1*1000000)}' /proc/uptime)
    [ $(( (now - t) / 1000000 )) -ge $( [ -e $E/.nogap ] && echo 0 || echo 1800 ) ] && break
  fi
  sleep 30
done
echo "== service gap satisfied $(date -Is); stopping titan-mistral"
trap 'systemctl --user start titan-mistral; echo "== titan-mistral restarted $(date -Is)"' EXIT
trap 'exit 130' INT TERM HUP
systemctl --user stop titan-mistral
for i in $(seq 1 30); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 700 ] && break; sleep 2; done
t0=$(date +%s)
timeout 1500 bash $D/run.sh "$BIN" "$LABEL"
echo "== bench window $LABEL: run.sh rc=$? after $(( $(date +%s) - t0 ))s (service down $(( $(date +%s) - t0 ))s + stop)"
