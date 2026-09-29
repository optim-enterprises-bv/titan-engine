#!/bin/bash
# titan-engine release benchmark campaign: ours (mistral.rs titan-094, the deployed service binary) vs llama.cpp
# acecd56 on this machine, per model, resumable per item. Config: campaign.json. Core: lib/camp.py.
#
#   campaign.sh plan               items still to run, time estimate, split into <= 55-min windows   (CPU only)
#   campaign.sh status             done / unstable / failed / pending per item                        (CPU only)
#   campaign.sh precheck           backup/maintenance timers and heavy jobs; exit 1 if not quiet       (CPU only)
#   campaign.sh report             rebuild results.md + results.json from results/*.json              (CPU only)
#   campaign.sh redo ITEM...       discard items' results so the next window runs them again          (CPU only)
#   campaign.sh dry                rehearsal with fake engines into dryrun/ (no GPU, no service stop) (CPU only)
#   campaign.sh window [ITEM...]   one GPU window: runs pending items (or only the named ones) for up to 55 min
#
# A window needs the GPU lock and the go-ahead file:
#   flock ~/titan-engine/.gpu.lock bash ~/titan-engine/release/bench/campaign.sh window
# (without an outer flock it takes the lock itself and waits for it). It refuses to start unless
# release/bench/GO exists: the main session creates it once the MTP-consistency fix is deployed
# (deploy/titan-mistral.service ExecStart = the binary to benchmark). The first window pins that binary's sha256;
# a later window with a different binary refuses to run (results would mix builds).
# No service gap (RULES-agents.md 2026-09-28 23:14). Stops only titan-mistral, restarts it on exit.
# Env: CAMPAIGN_POST=staging-build  after the items, build release/staging nvcc-free + smoke test (staging-build.sh)
#      CAMPAIGN_OPTIONAL=1 also runs optional items (Qwen3-14B, gpt-oss-120b at 28k).
#      CAMPAIGN_BUDGET_MIN=N shorter window budget (default 55).
set -u
E=$HOME/titan-engine; D=$E/release/bench; LOCK=$E/.gpu.lock
cmd=${1:-plan}; shift || true
py() { # py MAX_SECONDS script args...: python in a 4G-capped transient unit, killed after MAX_SECONDS
  local t=$1; shift
  systemd-run --user --collect --wait --pipe -q -p MemoryMax=4G -p MemorySwapMax=0 -p RuntimeMaxSec=$t \
         --setenv=CAMPAIGN_DRY=${CAMPAIGN_DRY:-0} --setenv=CAMPAIGN_OPTIONAL=${CAMPAIGN_OPTIONAL:-0} \
         ${CAMPAIGN_BUDGET_MIN:+--setenv=CAMPAIGN_BUDGET_MIN=$CAMPAIGN_BUDGET_MIN} \
         ${CAMPAIGN_ENGINE:+--setenv=CAMPAIGN_ENGINE=$CAMPAIGN_ENGINE} ${CAMPAIGN_MODEL:+--setenv=CAMPAIGN_MODEL=$CAMPAIGN_MODEL} \
         ${CAMPAIGN_REPIN:+--setenv=CAMPAIGN_REPIN=$CAMPAIGN_REPIN} \
         python3 "$@"; }

lock_held_by_ancestor() { # is .gpu.lock flock()ed by this process or one of its ancestors?
  local ino pids p q
  ino=$(stat -c %i "$LOCK" 2>/dev/null) || return 1
  pids=$(awk -v ino="$ino" '$2 == "FLOCK" { n = split($6, a, ":"); if (a[n] == ino) print $5 }' /proc/locks)
  p=$$
  while [ "${p:-1}" -gt 1 ]; do
    for q in $pids; do [ "$q" = "$p" ] && return 0; done
    p=$(awk '{print $4}' /proc/$p/stat 2>/dev/null)
  done
  return 1
}

case $cmd in
  plan|status|redo|precheck)
    exec python3 $D/lib/camp.py $cmd "$@" ;;
  report)
    exec python3 $D/lib/report.py "${1:-$D}" ;;
  dry)
    rm -rf $D/dryrun
    export CAMPAIGN_DRY=1 CAMPAIGN_OPTIONAL=1 CAMPAIGN_BUDGET_MIN=${CAMPAIGN_BUDGET_MIN:-120}
    py $(( CAMPAIGN_BUDGET_MIN * 60 + 300 )) $D/lib/camp.py window "$@"
    rc=$?
    systemctl --user stop "camp-*.service" 2>/dev/null
    mkdir -p $D/dryrun; python3 $D/lib/camp.py status > $D/dryrun/status.txt 2>&1
    echo "dry run rc=$rc; report: $D/dryrun/results.md"; exit $rc ;;
  window)
    [ -e $D/GO ] || { echo "release/bench/GO is missing: the GPU campaign waits for the MTP-consistency fix to be deployed"; exit 4; }
    if ! lock_held_by_ancestor; then
      echo "taking $LOCK (waiting for it)"; exec flock "$LOCK" bash "$0" window "$@"
    fi
    mkdir -p $D/logs
    exec > >(tee -a $D/logs/window-$(date +%Y%m%d-%H%M).log) 2>&1
    echo "== campaign window $(date -Is); lock held"
    python3 $D/lib/camp.py precheck || { echo "machine not quiet: window not started"; exit 5; }
    echo "== filters: engine=${CAMPAIGN_ENGINE:-all} model=${CAMPAIGN_MODEL:-all}"
    trap 'systemctl --user stop "camp-*.service" 2>/dev/null; systemctl --user start titan-mistral; echo "== titan-mistral restarted $(date -Is)"' EXIT
    trap 'exit 130' INT TERM HUP
    systemctl --user stop titan-mistral
    WSTART=$(date +%s)
    echo "== titan-mistral stopped $(date -Is)"
    budget=${CAMPAIGN_BUDGET_MIN:-55}
    # the Python driver runs in a 4G-capped unit; the model servers it starts are separate 20G-capped units
    py $(( budget * 60 + 240 )) $D/lib/camp.py window "$@"
    echo "== campaign window done rc=$? $(date -Is)"
    if [ "${CAMPAIGN_POST:-}" = staging-build ]; then  # AUDIT.md B5, in the same window while time remains
      bash $D/staging-build.sh $(( WSTART + 58 * 60 ))
    fi ;;
  *)
    sed -n '2,24p' "$0"; exit 2 ;;
esac
