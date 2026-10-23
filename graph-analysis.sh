#!/bin/bash
# Quantify the graph path's per-step overhead: segments, replays, input copies, and the step-time
# split with graphs on vs off (same prompt, same depth).
set -u
run() { # label extra env...
  local label=$1; shift
  systemctl --user stop titan-dense.service 2>/dev/null; sleep 2
  sudo -n pkill -f "mistralrs ser[v]e" 2>/dev/null; sleep 5
  systemctl --user reset-failed titan-dense.service 2>/dev/null
  systemd-run --user --unit=titan-dense --property=WorkingDirectory=/home/dingo/src/titan-engine \
    --setenv=TITAN_STEP_PROF=1 "$@" /home/dingo/src/titan-engine/serve-dense.sh 65536 >/dev/null 2>&1
  for i in $(seq 1 40); do sleep 3; ss -tln | grep -q ':1234' && break; done
  echo "=== $label ==="
  python3 - <<'PY'
import json, time, urllib.request
b={"model":"default","messages":[{"role":"user","content":"Write a very long detailed chapter about GPU memory hierarchies and bandwidth limits in inference engines."}],"temperature":0.0,"max_tokens":400}
r=urllib.request.Request("http://127.0.0.1:1234/v1/chat/completions",data=json.dumps(b).encode(),headers={"Content-Type":"application/json"})
t0=time.time()
with urllib.request.urlopen(r,timeout=1800) as f: d=json.loads(f.read().decode())
print(f"  decode={d['usage']['avg_compl_tok_per_sec']:.1f} tok/s")
PY
  journalctl --user -u titan-dense.service --no-pager --since "$(date -d '3 minutes ago' '+%Y-%m-%d %H:%M:%S')" 2>/dev/null \
    | grep -oE "(titan graphs: [0-9]+ segments.*|titan engine prof[^|]*)" | tail -3 | sed 's/^/  /'
}
run "graphs OFF" --setenv=TITAN_DENSE_GRAPHS=0
run "graphs ON"  --setenv=TITAN_DENSE_GRAPHS=1
