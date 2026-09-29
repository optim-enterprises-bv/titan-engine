trap 'systemctl --user start titan-mistral' EXIT
systemctl --user stop titan-mistral
# titan-094 window: holds the GPU lock with titan-mistral stopped and runs request scripts dropped
# into ctl/ (NAME.req -> NAME.req.log / NAME.req.rc, then renamed NAME.done) until ctl/stop appears
# or the deadline passes. Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/sync/w094/window.sh N
trap 'exit 130' INT TERM HUP
set -u
N=${1:-1}
E=$HOME/titan-engine; W=$E/sync/w094; C=$W/ctl
exec > >(tee -a $W/logs/window$N-$(date +%Y%m%d-%H%M).log) 2>&1
T0=$(date +%s); DEADLINE=$((T0 + 57 * 60))
el() { echo "$(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
echo "== window$N start $(date -Is), titan-mistral: $(systemctl --user is-active titan-mistral)"
for p in $(ps -eo pid=,ppid=,comm= | awk '$2 == 878468 && $3 ~ /rust-analyzer/ {print $1}'); do echo "killing rust-analyzer $p"; kill $p; done
free -m | head -2
rm -f $C/stop
echo $$ > $C/window.pid
touch $C/window.up
while [ $(left) -gt 30 ] && [ ! -f $C/stop ]; do
  for r in $(ls $C/*.req 2>/dev/null | sort | head -1); do
    n=$(basename $r .req)
    echo "== request $n start ($(el), $(left)s left)"
    timeout $(( $(left) - 20 )) bash $r > $C/$n.req.log 2>&1
    rc=$?
    echo $rc > $C/$n.req.rc
    mv $r $C/$n.done
    echo "== request $n rc=$rc ($(el))"
  done
  sleep 3
done
rm -f $C/window.up $C/window.pid
echo "== window$N end $(el) ($(date -Is))"
