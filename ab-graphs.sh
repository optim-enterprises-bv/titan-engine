#!/bin/bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO/bench-lib.sh"
set -u
start() {
  bench_stop_prod; sleep 2
  systemctl --user reset-failed titan-dense.service 2>/dev/null
  systemd-run --user --unit=titan-dense --property=WorkingDirectory=/home/dingo/src/titan-engine "$@" >/dev/null 2>&1
  for i in $(seq 1 40); do sleep 3; ss -tln | grep -q ':1234' && break; done
}
for g in 1 0; do
  echo "=== TITAN_CUDA_GRAPHS=$g ==="
  start --setenv=TITAN_MTP=0 --setenv=TITAN_CUDA_GRAPHS=$g -- /home/dingo/src/titan-engine/serve-dense.sh 65536
  cd /tmp && python3 "$REPO/decode-once.py" 16 >/dev/null 2>&1; python3 "$REPO/decode-once.py" 256; cd /home/dingo/src/titan-engine
done
