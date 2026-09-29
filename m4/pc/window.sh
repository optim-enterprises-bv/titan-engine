trap 'systemctl --user stop "pc-*" 2>/dev/null; systemctl --user start titan-mistral' EXIT
systemctl --user stop titan-mistral
# Prefix-cache window: holds ~/titan-engine/.gpu.lock with titan-mistral stopped and runs the step
# scripts dropped into q/next.sh one after another (each under `timeout`), until q/end appears or
# 57 minutes pass. Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/pc/window.sh
trap 'exit 130' INT TERM HUP
set -u
E=$HOME/titan-engine; P=$E/m4/pc; Q=$P/q
exec > >(tee -a $P/window-$(date +%Y%m%d-%H%M).log) 2>&1
T0=$(date +%s); DEADLINE=$((T0 + 57 * 60))
el() { echo "$(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s"; }
echo "== window start $(date -Is), titan-mistral: $(systemctl --user is-active titan-mistral)"
for p in $(ps -eo pid=,ppid=,comm= | awk '$2 == 878468 && $3 ~ /rust-analyzer/ {print $1}'); do echo "killing rust-analyzer $p"; kill $p; done
free -m | head -2
rm -f $Q/end $Q/stepdone
while :; do
  now=$(date +%s)
  [ $now -ge $((DEADLINE - 30)) ] && { echo "== deadline"; break; }
  [ -e $Q/end ] && { rm -f $Q/end; echo "== end requested"; break; }
  if [ -e $Q/next.sh ]; then
    mv $Q/next.sh $Q/run.sh
    left=$((DEADLINE - now - 25))
    echo "== step $(head -1 $Q/run.sh) at $(el), ${left}s left"
    DEADLINE=$DEADLINE timeout --kill-after=20 $left bash $Q/run.sh
    echo "== step rc=$? at $(el)"
    mv $Q/run.sh $Q/done-$(date +%H%M%S).sh
    systemctl --user stop "pc-*" 2>/dev/null
    touch $Q/stepdone
  fi
  sleep 2
done
free -m | head -2
echo "== window end $(el)"
