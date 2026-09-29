#!/bin/bash
# fm window 1: build the titan-094 integration candidate (flash-decode + mmvq-moe, f5458a45c) nvcc-free into
# the warm target-kv-oxide, copy it to bin/mistralrs-titan-fm-candidate, then run the two 40-prompt identity
# gates (MTP off vs q35-prof, MTP=2 vs h-off).
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/fm/window1.sh
set -u
E=$HOME/titan-engine; W=$E/m4/fm
exec > >(tee -a $W/window1-$(date +%Y%m%d-%H%M).log) 2>&1
echo "== lock acquired $(date -Is); no service gap (.nogap, RULES-agents.md 2026-09-28 23:14)"
trap 'systemctl --user start titan-mistral; echo "== titan-mistral restart requested $(date -Is)"' EXIT
trap 'exit 130' INT TERM HUP
systemctl --user stop titan-mistral
for p in $(ps -eo pid=,ppid=,comm= | awk '$2 == 878468 && $3 ~ /rust-analyzer/ {print $1}'); do echo "killing rust-analyzer $p"; kill $p; done
free -m | head -2
T0=$(date +%s); el() { echo "$(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s"; }
echo "== window1 start $(date -Is), titan-mistral: $(systemctl --user is-active titan-mistral)"

echo "== [1/2] build ($(el))"
bash $W/build.sh w1build
BUILD_RC=$?
BIN=$E/target-kv-oxide/release/mistralrs
if [ "$BUILD_RC" -ne 0 ] || [ ! -x "$BIN" ] || [ ! -f "$W/ctl/build.ok" ]; then
  echo "BUILD FAILED or not fresh - aborting window1"
  exit 1
fi
CAND=$E/bin/mistralrs-titan-fm-candidate
cp -a "$BIN" "$CAND"
sha256sum "$CAND" | tee "$W/ctl/candidate.sha256"

echo "== [2/2] identity gates ($(el))"
bash $W/gate.sh fmw1 "$CAND"

echo "== gate summaries =="
cat $W/out/gate-m2.json 2>&1; echo
cat $W/out/gate-m0.json 2>&1; echo
echo "== window1 done $(el) ($(date -Is))"
