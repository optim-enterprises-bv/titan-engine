#!/bin/bash
set -u
start() {
  systemctl --user stop titan-dense.service 2>/dev/null; sleep 2
  systemctl --user reset-failed titan-dense.service 2>/dev/null
  systemd-run --user --unit=titan-dense --property=WorkingDirectory=/home/dingo/src/titan-engine "$@" >/dev/null 2>&1
  for i in $(seq 1 40); do sleep 3; ss -tln | grep -q ':1234' && break; done
}
for cap in 8 2 1; do
  echo "=== paged ON, batch 8, MMVQ_MAX_BATCH=$cap (mmq above it) ==="
  start --setenv=TITAN_MTP=0 --setenv=TITAN_PAGED=on --setenv=TITAN_BATCH=8 \
        --setenv=TITAN_RECURRENT_SLOTS=8 --setenv=TITAN_MMVQ_MAX_BATCH=$cap \
        -- /home/dingo/src/titan-engine/serve-dense.sh 32768
  python3 bench-concurrent.py 96
done
