#!/bin/bash
set -u
start() {
  systemctl --user stop titan-dense.service 2>/dev/null; sleep 2
  pkill -f "mistralrs ser[v]e" 2>/dev/null; sleep 4
  systemctl --user reset-failed titan-dense.service 2>/dev/null
  systemd-run --user --unit=titan-dense --property=WorkingDirectory=/home/dingo/src/titan-engine "$@" >/dev/null 2>&1
  for i in $(seq 1 40); do sleep 3; ss -tln | grep -q ':1234' && break; done
}
for c in 0 8192; do
  echo "=== TITAN_MMQ_SMALL_COLS=$c ==="
  start --setenv=TITAN_MTP=0 --setenv=TITAN_MMQ_SMALL_COLS=$c -- /home/dingo/src/titan-engine/serve-dense.sh 65536
  cd /tmp && python3 decode-once.py 16 >/dev/null 2>&1; python3 decode-once.py 256; cd /home/dingo/src/titan-engine
done
