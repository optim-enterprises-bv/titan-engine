#!/bin/bash
# fm window 2: bench A/B/A bracketing to handle load contamination -- deployed (pre-fm), candidate (fm),
# deployed again (pre-fm-2), back to back.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/fm/window2.sh
set -u
E=$HOME/titan-engine; W=$E/m4/fm
exec > >(tee -a $W/window2-$(date +%Y%m%d-%H%M).log) 2>&1
echo "== lock acquired $(date -Is); no service gap (.nogap)"
trap 'systemctl --user start titan-mistral; echo "== titan-mistral restart requested $(date -Is)"' EXIT
trap 'exit 130' INT TERM HUP
systemctl --user stop titan-mistral
for p in $(ps -eo pid=,ppid=,comm= | awk '$2 == 878468 && $3 ~ /rust-analyzer/ {print $1}'); do echo "killing rust-analyzer $p"; kill $p; done
free -m | head -2
T0=$(date +%s); el() { echo "$(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s"; }
echo "== window2 start $(date -Is), titan-mistral: $(systemctl --user is-active titan-mistral)"

DEPLOYED=$E/bin/mistralrs-titan-kv
CAND=$E/bin/mistralrs-titan-fm-candidate
[ -x "$CAND" ] || { echo "no candidate binary at $CAND -- run window1 first"; exit 1; }

echo "== [1/3] bench pre-fm (deployed) ($(el))"
timeout 1200 bash $E/bench/run.sh "$DEPLOYED" pre-fm
echo "pre-fm rc=$? ($(el))"

echo "== [2/3] bench fm (candidate) ($(el))"
BENCH_SRC=$E/mr-094 timeout 1200 bash $E/bench/run.sh "$CAND" fm
echo "fm rc=$? ($(el))"

echo "== [3/3] bench pre-fm-2 (deployed) ($(el))"
timeout 1200 bash $E/bench/run.sh "$DEPLOYED" pre-fm-2
echo "pre-fm-2 rc=$? ($(el))"

echo "== window2 done $(el) ($(date -Is))"
