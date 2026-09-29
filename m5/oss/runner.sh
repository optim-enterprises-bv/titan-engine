trap 'systemctl --user start titan-mistral' EXIT
systemctl --user stop titan-mistral
# M5 gpt-oss window runner. Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m5/oss/runner.sh
# Runs queued jobs q/NN-name.sh in order (each under timeout, capped by the window deadline) until q/END
# exists and the queue is empty, or 57 minutes have passed. q/HOLD (set by a failed build) pauses the queue.
trap 'exit 130' INT TERM HUP
set -u
O=$HOME/titan-engine/m5/oss; Q=$O/q; mkdir -p $Q/done $O/logs
LOG=$O/logs/runner-$(date +%Y%m%d-%H%M).log
exec > >(tee -a $LOG) 2>&1
T0=$(date +%s); DEADLINE=$((T0 + 57 * 60))
el() { echo "$(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s"; }
echo "== window start $(date -Is), titan-mistral: $(systemctl --user is-active titan-mistral)"
free -m | head -2
for p in $(ps -eo pid=,ppid=,comm= | awk '$2 == 878468 && $3 ~ /rust-analyzer/ {print $1}'); do echo "killing rust-analyzer $p"; kill $p; done
while :; do
  left=$(( DEADLINE - $(date +%s) ))
  [ $left -le 40 ] && { echo "== deadline reached"; break; }
  [ -e $Q/HOLD ] && { sleep 5; continue; }
  job=$(ls $Q/*.sh 2>/dev/null | sort | head -1)
  if [ -z "$job" ]; then [ -e $Q/END ] && break; sleep 5; continue; fi
  name=$(basename $job .sh); mv $job $Q/done/$name.sh
  echo "== job $name start ($(el)), ${left}s left"
  DEADLINE=$DEADLINE timeout -k 15 $(( left - 25 )) bash $Q/done/$name.sh
  echo "== job $name rc=$? ($(el))"
done
for u in $(systemctl --user list-units 'oss-*' --plain --no-legend | awk '{print $1}'); do echo "stopping leftover $u"; systemctl --user stop $u; done
rm -f $Q/END
echo "== window end $(date -Is), downtime $(el)"
