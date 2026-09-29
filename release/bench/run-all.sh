#!/bin/bash
# Run campaign windows back to back until no required item is pending, then one window for the staging build test
# (AUDIT.md B5). Each window takes ~/titan-engine/.gpu.lock and restarts titan-mistral when it ends. Between windows
# the lock is released for 60 s (other agents queue on it; titan-mistral serves meanwhile). A window that could not
# start or ended on a heavy job (restic, dnf, packagekit, ...) is retried after 10 min. At most MAX windows.
# One summary line per window goes to logs/progress.txt.
set -u
D=$HOME/titan-engine/release/bench; MAX=${MAX:-12}; P=$D/logs/progress.txt
mkdir -p $D/logs
pending() { python3 $D/lib/camp.py status | grep -v optional | grep -c -E "pending|FAILED x1"; }
n=0; built=0
while [ $n -lt $MAX ]; do
  before=$(ls $D/results/*.json 2>/dev/null | wc -l)
  left=$(pending)
  post=""
  if [ "$left" -eq 0 ]; then
    [ $built = 1 ] && break
    post=staging-build
  fi
  n=$((n + 1))
  t0=$(date +%s)
  CAMPAIGN_POST=$post flock $HOME/titan-engine/.gpu.lock bash $D/campaign.sh window > $D/logs/run-all-w$n.out 2>&1
  rc=$?
  after=$(ls $D/results/*.json 2>/dev/null | wc -l)
  redo=$(grep -c "REDO" $D/logs/run-all-w$n.out)
  why=$(grep -h -o -E "heavy job running[^:]*|machine not quiet|NOT QUIET.*|interrupted by the window deadline|FAILED [A-Za-z]+" $D/logs/run-all-w$n.out | sort | uniq -c | tr '\n' ';')
  sb=""; [ -n "$post" ] && [ -f $D/staging-build/result.json ] && sb=" staging-build: $(python3 -c "import json;d=json.load(open('$D/staging-build/result.json'));print('rc', d['build_rc'], 'smoke', {k: v.get('identity') for k, v in d['smoke'].items()})")"
  echo "$(date +%H:%M) window $n (rc $rc, $(( ($(date +%s) - t0) / 60 )) min): items done $((after - before)), total $after results, required pending $(pending), redo attempts $redo ${why:+[$why]}$sb" | tee -a $P
  if [ -n "$post" ]; then
    grep -q "build rc=0" $D/logs/run-all-w$n.out && built=1
    grep -q "build deferred\|build rc=" $D/logs/run-all-w$n.out || true
    [ $built = 1 ] && break
  fi
  if [ $((after - before)) -eq 0 ] && [ -z "$post" ]; then sleep 600; else sleep 60; fi
done
echo "$(date +%H:%M) run-all finished after $n windows" | tee -a $P
