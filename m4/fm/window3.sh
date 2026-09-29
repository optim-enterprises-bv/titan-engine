#!/bin/bash
# fm window 3: long-context decode compare (candidate vs deployed, ~13k and ~27.7k prompts, MTP=2, 256 decode
# tokens), then the deploy decision, and the deploy itself if all criteria hold.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/fm/window3.sh
set -u
E=$HOME/titan-engine; W=$E/m4/fm
exec > >(tee -a $W/window3-$(date +%Y%m%d-%H%M).log) 2>&1
echo "== lock acquired $(date -Is); no service gap (.nogap)"
trap 'systemctl --user start titan-mistral; echo "== titan-mistral restart requested $(date -Is)"' EXIT
trap 'exit 130' INT TERM HUP
systemctl --user stop titan-mistral
for p in $(ps -eo pid=,ppid=,comm= | awk '$2 == 878468 && $3 ~ /rust-analyzer/ {print $1}'); do echo "killing rust-analyzer $p"; kill $p; done
free -m | head -2
T0=$(date +%s); el() { echo "$(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s"; }
echo "== window3 start $(date -Is), titan-mistral: $(systemctl --user is-active titan-mistral)"

CAND=$E/bin/mistralrs-titan-fm-candidate
DEPLOYED=$E/bin/mistralrs-titan-kv
[ -x "$CAND" ] || { echo "no candidate binary at $CAND -- run window1 first"; exit 1; }

echo "== [1/3] long-context: deployed ($(el))"
timeout 900 bash $W/longctx.sh deployed "$DEPLOYED"
echo "== [2/3] long-context: candidate ($(el))"
timeout 900 bash $W/longctx.sh candidate "$CAND"

echo "== [3/3] decision ($(el))"
python3 $W/decide.py $W/out/decision.json "$CAND"
DEPLOY=$(python3 -c "import json;print(json.load(open('$W/out/decision.json'))['deploy'])")
echo "deploy=$DEPLOY"
if [ "$DEPLOY" = "True" ]; then
  DEST=$E/bin/mistralrs-titan-fm
  cp -a "$CAND" "$DEST"
  sha256sum "$DEST"
  for f in $E/deploy/titan-mistral.service $HOME/.config/systemd/user/titan-mistral.service; do
    cp "$f" "$f.bak-fm"
    sed -i 's#bin/mistralrs-titan-kv#bin/mistralrs-titan-fm#' "$f"
    grep -n "ExecStart=\|^Environment=" "$f"
  done
  systemctl --user daemon-reload
  echo "deployed $DEST, unit files updated, daemon-reload done; trap will restart the service on exit"
else
  echo "NOT deploying: $(python3 -c "import json;print(json.load(open('$W/out/decision.json'))['reasons'])")"
fi
echo "== window3 done $(el) ($(date -Is))"
