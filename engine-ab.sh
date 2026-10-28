#!/bin/bash
set -u
run() { # label extra env...
  local label=$1; shift
  systemctl --user stop titan-dense.service 2>/dev/null; sleep 2
  sudo -n pkill -f "mistralrs ser[v]e" 2>/dev/null; sleep 5
  systemctl --user reset-failed titan-dense.service 2>/dev/null
  systemd-run --user --unit=titan-dense --property=WorkingDirectory=/home/dingo/src/titan-engine "$@" \
    /home/dingo/src/titan-engine/serve-dense.sh 65536 >/dev/null 2>&1
  for i in $(seq 1 40); do sleep 3; ss -tln | grep -q ':1234' && break; done
  echo "=== $label ==="
  python3 - <<'PY'
import json, time, urllib.request
b={"model":"default","messages":[{"role":"user","content":"Write a very long detailed chapter about GPU memory hierarchies and bandwidth limits in inference engines."}],"temperature":0.0,"max_tokens":700}
r=urllib.request.Request("http://127.0.0.1:1234/v1/chat/completions",data=json.dumps(b).encode(),headers={"Content-Type":"application/json"})
t0=time.time()
with urllib.request.urlopen(r,timeout=1800) as f: d=json.loads(f.read().decode())
u=d["usage"]
print(f"  tok={u['completion_tokens']} decode={u['avg_compl_tok_per_sec']:.1f} tok/s wall={time.time()-t0:.1f}s")
PY
}
run "dense graphs OFF (current default)" --setenv=TITAN_DENSE_GRAPHS=0
run "dense graphs ON  (TITAN_DENSE_GRAPHS=1)" --setenv=TITAN_DENSE_GRAPHS=1
run "dense graphs ON, MTP off" --setenv=TITAN_DENSE_GRAPHS=1 --setenv=TITAN_MTP=0
