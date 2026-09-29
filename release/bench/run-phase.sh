#!/bin/bash
# One campaign phase in short windows: run-phase.sh ENGINE [MODEL ...]   (ENGINE = llama | ours)
# One model per window (CAMPAIGN_MODEL), window budget campaign.json window_minutes (20); a model with items left
# after its window gets another (at most 3). Each window takes ~/titan-engine/.gpu.lock itself and releases it when it
# ends (titan-mistral restarted by campaign.sh's trap); 90 s pause between windows so other agents' windows slot in.
# A window that ran nothing (heavy job, not quiet, lock contention timing) waits 10 min before retrying.
# ENGINE=ours with POST=staging-build runs the staging build test (AUDIT.md B5) after the last ours window.
# One line per window in logs/progress.txt.
set -u
D=$HOME/titan-engine/release/bench; ENG=${1:?llama|ours}; shift
MODELS=${*:-$(python3 -c "import json;print(' '.join(m['id'] for m in json.load(open('$D/campaign.json'))['models'] if not m.get('optional')))")}
P=$D/logs/progress.txt; mkdir -p $D/logs
pending() { CAMPAIGN_ENGINE=$ENG CAMPAIGN_MODEL=$1 python3 $D/lib/camp.py status | grep -c -E " (pending|FAILED x1)"; }
w=0
for m in $MODELS; do
  tries=0
  while [ "$(pending $m)" -gt 0 ] && [ $tries -lt 3 ]; do
    tries=$((tries + 1)); w=$((w + 1)); t0=$(date +%s)
    before=$(ls $D/results/*.json 2>/dev/null | wc -l)
    CAMPAIGN_ENGINE=$ENG CAMPAIGN_MODEL=$m flock $HOME/titan-engine/.gpu.lock bash $D/campaign.sh window \
      > $D/logs/phase-$ENG-$m-$tries.out 2>&1
    after=$(ls $D/results/*.json 2>/dev/null | wc -l)
    redo=$(grep -c "REDO" $D/logs/phase-$ENG-$m-$tries.out)
    why=$(grep -h -o -E "heavy job running[^:]*|machine not quiet|interrupted by the window deadline|FAILED [A-Za-z]+|needs ~[0-9.]+ min" \
          $D/logs/phase-$ENG-$m-$tries.out | sort | uniq -c | tr -s ' ' | tr '\n' ';')
    echo "$(date +%H:%M) $ENG window $w ($m, $(( ($(date +%s) - t0) / 60 )) min): $((after - before)) items done, $(pending $m) left for $m, redo attempts $redo ${why:+[$why]}" | tee -a $P
    if [ $((after - before)) -eq 0 ]; then sleep 600; else sleep 90; fi
  done
done
if [ "$ENG" = ours ] && [ "${POST:-}" = staging-build ]; then
  w=$((w + 1)); t0=$(date +%s)
  CAMPAIGN_ENGINE=ours CAMPAIGN_MODEL=none CAMPAIGN_POST=staging-build flock $HOME/titan-engine/.gpu.lock \
    bash $D/campaign.sh window > $D/logs/phase-staging-build.out 2>&1
  echo "$(date +%H:%M) staging-build window ($(( ($(date +%s) - t0) / 60 )) min): $(cat $D/staging-build/result.json 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print('build rc', d['build_rc'], 'smoke', {k: v.get('identity') for k, v in d['smoke'].items()})" 2>&1)" | tee -a $P
fi
echo "$(date +%H:%M) $ENG phase finished after $w windows" | tee -a $P
