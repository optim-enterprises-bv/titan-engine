#!/bin/bash
set -u
start() { # extra env + serve flags
  systemctl --user stop titan-dense.service 2>/dev/null; sleep 2
  systemctl --user reset-failed titan-dense.service 2>/dev/null
  systemd-run --user --unit=titan-dense --property=WorkingDirectory=/home/dingo/src/titan-engine \
    --setenv=TITAN_MTP=0 "$@" >/dev/null 2>&1
  for i in $(seq 1 40); do sleep 3; ss -tln | grep -q ':1234' && break; done
}
echo "=== A: bf16 matvec ON, single stream + 8 concurrent ==="
start --setenv=TITAN_DENSE_BF16_MATVEC=1 -- /home/dingo/src/titan-engine/serve-dense.sh 32768
python3 bench-concurrent.py 128
echo "=== B: bf16 matvec OFF (f32), single stream + 8 concurrent ==="
start --setenv=TITAN_DENSE_BF16_MATVEC=0 -- /home/dingo/src/titan-engine/serve-dense.sh 32768
python3 bench-concurrent.py 128
