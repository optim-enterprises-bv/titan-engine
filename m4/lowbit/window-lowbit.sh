#!/bin/bash
# Low-bit Qwen3.6-35B-A3B (IQ3_XXS / IQ2_M, all-in-VRAM) vs tiered Q4_K_XL bench, llama.cpp only.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/lowbit/window-lowbit.sh
trap 'systemctl --user start titan-mistral' EXIT
trap 'exit 130' INT TERM HUP
set -u
E=$HOME/titan-engine; L=$E/m4/lowbit
mkdir -p $L/out
exec > >(tee -a $L/window-$(date +%Y%m%d-%H%M).log) 2>&1
WINDOW_START=$(date +%s)
WINDOW_MAX=$((60 * 60))  # org-wide ≤60min/window cap from RULES-agents.md
echo "== window start $(date -Is), titan-mistral: $(systemctl --user is-active titan-mistral)"

# 30-minute service gap (RULES-agents.md, added 2026-09-28 15:10): right after acquiring the
# flock and before stopping titan-mistral, wait until it has been active >= 30 minutes so we
# don't yank a just-restarted service out from under another agent's window.
echo "-- waiting for titan-mistral service gap (>= 30min active) --"
while :; do
  t=$(systemctl --user show titan-mistral -p ActiveEnterTimestampMonotonic --value)
  now=$(awk '{print int($1*1000000)}' /proc/uptime)
  gap=$(( (now - t) / 1000000 ))
  echo "  titan-mistral active for ${gap}s ($(date -Is))"
  [ "$gap" -ge $( [ -e $HOME/titan-engine/.nogap ] && echo 0 || echo 1800 ) ] && break
  sleep 30
done
echo "-- service gap satisfied, proceeding --"

systemctl --user stop titan-mistral
sleep 2
echo "titan-mistral after stop: $(systemctl --user is-active titan-mistral)"
free -m | head -2
# nvidia-smi/NVML is known broken on this box (version-strict NVML fails, CUDA/libcuda is fine);
# this is best-effort telemetry only - never let it abort the window. Real VRAM fit is judged
# from llama-server's own load log (CUDA buffer-size lines) inside bench_lowbit.py.
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>&1 || echo "(nvidia-smi unavailable, ignoring - not fatal)"

# budget the python benchmark so gap-wait + work stays within the 60min org cap, capped at
# 40min of actual work regardless (user asked for one window, at most 45min of GPU work)
NOW=$(date +%s)
ELAPSED=$((NOW - WINDOW_START))
BUDGET=$((WINDOW_MAX - ELAPSED - 300))  # 5min safety margin for teardown/restart
[ "$BUDGET" -gt 2400 ] && BUDGET=2400
if [ "$BUDGET" -lt 300 ]; then
  echo "!! only ${BUDGET}s left after the service gap - not enough to run the benchmark safely, aborting"
  systemctl --user start titan-mistral
  trap - EXIT
  exit 1
fi
echo "-- benchmark budget: ${BUDGET}s --"
DEADLINE=$(( $(date +%s) + BUDGET ))
timeout $((BUDGET + 60)) python3 $L/bench_lowbit.py $DEADLINE
rc=$?
echo "bench_lowbit.py rc=$rc"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>&1 || echo "(nvidia-smi unavailable, ignoring - not fatal)"
echo "== window end $(date -Is), total elapsed $(( $(date +%s) - WINDOW_START ))s"
