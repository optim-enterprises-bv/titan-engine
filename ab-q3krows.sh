#!/bin/bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO/bench-lib.sh"
set -u
start() {
  bench_stop_prod; sleep 2
  pkill -f "mistralrs ser[v]e" 2>/dev/null; sleep 4
  systemctl --user reset-failed titan-dense.service 2>/dev/null
  systemd-run --user --unit=titan-dense --property=WorkingDirectory=/home/dingo/src/titan-engine "$@" >/dev/null 2>&1
  for i in $(seq 1 40); do sleep 3; ss -tln | grep -q ':1234' && break; done
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
}
for v in 1 0 1 0; do
  echo "=== TITAN_Q3K_ROWS=$v ==="
  start --setenv=TITAN_MTP=0 --setenv=TITAN_Q3K_ROWS=$v -- /home/dingo/src/titan-engine/serve-dense.sh 65536
  cd /tmp && python3 "$REPO/decode-once.py" 16 >/dev/null 2>&1; python3 "$REPO/decode-once.py" 256; cd /home/dingo/src/titan-engine
done
