# 30-minute service gap (RULES-agents.md 2026-09-28 15:10): after taking the flock, before stopping titan-mistral
echo "== lock acquired $(date -Is), waiting for titan-mistral >= 30 min active"
while :; do t=$(systemctl --user show titan-mistral -p ActiveEnterTimestampMonotonic --value)
  now=$(awk '{print int($1*1000000)}' /proc/uptime); [ $(( (now - t) / 1000000 )) -ge $( [ -e $HOME/titan-engine/.nogap ] && echo 0 || echo 1800 ) ] && break; sleep 30; done
trap 'systemctl --user start titan-mistral' EXIT
systemctl --user stop titan-mistral
# miss-skip window runner. Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/skip/runner.sh
# Executes queued jobs q/NN-name.sh in order (each under timeout, capped by the window deadline) until
# q/END exists and the queue is empty, or 57 minutes have passed. Jobs source lib.sh.
trap 'exit 130' INT TERM HUP
set -u
N=$HOME/titan-engine/m4/skip; Q=$N/q; mkdir -p $Q/done $N/logs
LOG=$N/logs/runner-$(date +%Y%m%d-%H%M).log
exec > >(tee -a $LOG) 2>&1
T0=$(date +%s); DEADLINE=$((T0 + 57 * 60))
el() { echo "$(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s"; }
echo "== window start $(date -Is), titan-mistral: $(systemctl --user is-active titan-mistral)"
free -m | head -2
for p in $(ps -eo pid=,ppid=,comm= | awk '$2 == 878468 && $3 ~ /rust-analyzer/ {print $1}'); do echo "killing rust-analyzer $p"; kill $p; done
while :; do
  left=$(( DEADLINE - $(date +%s) ))
  [ $left -le 40 ] && { echo "== deadline reached"; break; }
  [ -e $Q/HOLD ] && { sleep 5; continue; }  # a failed build holds the queue until fixed
  job=$(ls $Q/*.sh 2>/dev/null | sort | head -1)
  if [ -z "$job" ]; then [ -e $Q/END ] && break; sleep 5; continue; fi
  name=$(basename $job .sh); mv $job $Q/done/$name.sh
  echo "== job $name start ($(el)), ${left}s left"
  DEADLINE=$DEADLINE timeout -k 15 $(( left - 25 )) bash $Q/done/$name.sh
  echo "== job $name rc=$? ($(el))"
done
for u in $(systemctl --user list-units 'sk-*' --plain --no-legend | awk '{print $1}'); do echo "stopping leftover $u"; systemctl --user stop $u; done
rm -f $Q/END
echo "== window end $(date -Is), downtime $(el)"
