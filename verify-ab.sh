#!/bin/bash
set -u
run() { # label extra-env
  local label=$1; shift
  systemctl --user stop titan-dense.service 2>/dev/null; sleep 2
  sudo -n pkill -f "mistralrs ser[v]e" 2>/dev/null; sleep 5
  systemctl --user reset-failed titan-dense.service 2>/dev/null
  systemd-run --user --unit=titan-dense --property=WorkingDirectory=/home/dingo/src/titan-engine \
    --setenv=TITAN_MTP=1 "$@" /home/dingo/src/titan-engine/serve-dense.sh 65536 >/dev/null 2>&1
  for i in $(seq 1 40); do sleep 3; ss -tln | grep -q ':1234' && break; done
  echo "=== $label ==="
  python3 - <<'PY'
import json, time, urllib.request
b={"model":"default","messages":[{"role":"user","content":"Write a very long detailed technical chapter about GPU memory hierarchies and bandwiidth limits in inference engines."}],"temperature":0.0,"max_tokens":700}
r=urllib.request.Request("http://127.0.0.1:1234/v1/chat/completions",data=json.dumps(b).encode(),headers={"Content-Type":"application/json"})
t0=time.time()
with urllib.request.urlopen(r,timeout=1800) as f: d=json.loads(f.read().decode())
u=d["usage"]
print(f"  tok={u['completion_tokens']} decode={u['avg_compl_tok_per_sec']:.1f} tok/s wall={time.time()-t0:.1f}s")
PY
  journalctl --user -u titan-dense.service --no-pager --since "$(date -d '2 minutes ago' '+%Y-%m-%d %H:%M:%S')" 2>/dev/null \
    | grep -oE "titan mtp stats: .*" | tail -1 | sed 's/^/  /'
}
run "batched rows (default)"
run "per-row (TITAN_MTP_VERIFY=rowwise)" --setenv=TITAN_MTP_VERIFY=rowwise
