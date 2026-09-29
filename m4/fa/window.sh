# flash-decode window: stops titan-mistral, then runs the request scripts in
# ctl/ (NAME.req -> NAME.req.log / NAME.req.rc, renamed NAME.done) in name order until ctl/stop or the deadline.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/fa/window.sh
set -u
E=$HOME/titan-engine; W=$E/m4/fa; C=$W/ctl
exec > >(tee -a $W/window-$(date +%Y%m%d-%H%M).log) 2>&1
echo "== lock acquired $(date -Is); no service gap (.nogap)"
trap 'systemctl --user start titan-mistral; echo "== titan-mistral restarted $(date -Is)"' EXIT
trap 'exit 130' INT TERM HUP
systemctl --user stop titan-mistral
T0=$(date +%s); DEADLINE=$((T0 + 57 * 60))
el() { echo "$(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
echo "== window start $(date -Is), titan-mistral: $(systemctl --user is-active titan-mistral)"
for p in $(ps -eo pid=,ppid=,comm= | awk '$2 == 878468 && $3 ~ /rust-analyzer/ {print $1}'); do echo "killing rust-analyzer $p"; kill $p; done
free -m | head -2
rm -f $C/stop
touch $C/window.up
while [ $(left) -gt 30 ] && [ ! -f $C/stop ]; do
  for r in $(ls $C/*.req 2>/dev/null | sort | head -1); do
    n=$(basename $r .req)
    echo "== request $n start ($(el), $(left)s left)"
    DEADLINE=$DEADLINE timeout $(( $(left) - 20 )) bash $r > $C/$n.req.log 2>&1
    rc=$?
    echo $rc > $C/$n.req.rc
    mv $r $C/$n.done
    echo "== request $n rc=$rc ($(el))"
    tail -25 $C/$n.req.log
  done
  sleep 3
done
rm -f $C/window.up
echo "== window end $(el) ($(date -Is))"
