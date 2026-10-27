#!/bin/bash
# MTP acceptance by draft depth. The reporter fires every 200 verify steps, so each run must be
# long enough to trip it: 700 tokens at depth 1 = ~700 steps, at depth 3 = ~230 steps.
set -u
start() { # depth
  systemctl --user stop titan-dense.service 2>/dev/null; sleep 2
  sudo -n pkill -f "mistralrs ser[v]e" 2>/dev/null; sleep 5
  systemctl --user reset-failed titan-dense.service 2>/dev/null
  systemd-run --user --unit=titan-dense --property=WorkingDirectory=/home/dingo/src/titan-engine \
    --setenv=TITAN_MTP="$1" /home/dingo/src/titan-engine/serve-dense.sh 65536 >/dev/null 2>&1
  for i in $(seq 1 40); do sleep 3; ss -tln | grep -q ':1234' && break; done
}
bench() { python3 - <<'PY'
import json, time, urllib.request
b={"model":"default","messages":[{"role":"user","content":"Write a very long, detailed technical chapter about how GPU memory hierarchies and bandwidth limits shape LLM inference. Be exhaustive."}],"temperature":0.0,"max_tokens":700}
r=urllib.request.Request("http://127.0.0.1:1234/v1/chat/completions",data=json.dumps(b).encode(),headers={"Content-Type":"application/json"})
t0=time.time()
with urllib.request.urlopen(r,timeout=1800) as f: d=json.loads(f.read().decode())
u=d["usage"]
print(f"  tok={u['completion_tokens']} decode={u.get('avg_compl_tok_per_sec',0):.1f} tok/s wall={time.time()-t0:.1f}s")
PY
}
for depth in 0 1 2 3; do
  echo "=== TITAN_MTP=$depth ==="
  start "$depth"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/  VRAM: /'
  bench
  # --since the start of this depth's run: a plain `-n 4000` picks up the previous depth's lines
  journalctl --user -u titan-dense.service --no-pager --since "$(date -d '3 minutes ago' '+%Y-%m-%d %H:%M:%S')" 2>/dev/null \
    | grep -oE "titan mtp stats: .*" | tail -2 | sed 's/^/  /' 
done
