#!/bin/bash
# titan-094 integration: deploy step only (build+bench+monitor already done and passed the gate in
# m4/094-int/decision.json). Run as: flock ~/titan-engine/.gpu.lock bash <this script>
set -u
E=$HOME/titan-engine; LOGDIR=$E/m4/094-int
exec > >(tee -a "$LOGDIR/deploy-$(date +%Y%m%d-%H%M).log") 2>&1
echo "== lock acquired $(date -Is)"
DEC=$LOGDIR/decision.json
DEPLOY=$(python3 -c "import json;print(json.load(open('$DEC'))['deploy'])")
echo "decision.deploy=$DEPLOY"
[ "$DEPLOY" = "True" ] || { echo "decision says do not deploy -- aborting"; exit 1; }

CAND=$E/bin/mistralrs-titan-094-pfsmon-candidate
[ -x "$CAND" ] || { echo "candidate binary missing: $CAND"; exit 1; }
DEST=$E/bin/mistralrs-titan-094b
cp -a "$CAND" "$DEST"
sha256sum "$CAND" "$DEST"

for f in "$E/deploy/titan-mistral.service" "$HOME/.config/systemd/user/titan-mistral.service"; do
  cp "$f" "$f.bak-094int"
  sed -i 's#bin/mistralrs-titan-pfs#bin/mistralrs-titan-094b#' "$f"
  echo "-- $f --"
  grep -n "ExecStart=\|^Environment=" "$f"
done
systemctl --user daemon-reload
echo "daemon-reload done $(date -Is)"

STOP_TS=$(date -Is)
systemctl --user stop titan-mistral
echo "== titan-mistral stopped $STOP_TS, starting new binary"
systemctl --user start titan-mistral
for i in $(seq 1 120); do curl -sf -m 2 -o /dev/null localhost:1234/v1/models && break; sleep 2; done
START_TS=$(date -Is)
echo "== titan-mistral up (or timed out waiting) $START_TS after $((i*2))s"
systemctl --user is-active titan-mistral
systemctl --user status titan-mistral --no-pager | head -10
echo "== deploy done $(date -Is)"
