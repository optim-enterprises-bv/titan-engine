#!/bin/bash
set -u
start() {
  systemctl --user stop titan-dense.service 2>/dev/null; sleep 2
  systemctl --user reset-failed titan-dense.service 2>/dev/null
  systemd-run --user --unit=titan-dense --property=WorkingDirectory=/home/dingo/src/titan-engine "$@" >/dev/null 2>&1
  for i in $(seq 1 40); do sleep 3; ss -tln | grep -q ':1234' && break; done
}
echo "=== paged-attn ON, batch 8 ==="
start --setenv=TITAN_MTP=0 --setenv=TITAN_PAGED=on --setenv=TITAN_BATCH=8 -- /home/dingo/src/titan-engine/serve-dense.sh 32768
curl -s -m 5 http://127.0.0.1:1234/v1/models | head -c 120; echo
python3 bench-concurrent.py 128
echo "--- server log tail ---"
journalctl --user -u titan-dense.service --no-pager -n 60 2>&1 | grep -vE "titan step prof|engine prof" | tail -12
