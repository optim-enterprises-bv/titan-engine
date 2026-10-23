#!/bin/bash
set -u
start() {
  systemctl --user stop titan-dense.service 2>/dev/null; sleep 2
  sudo -n pkill -f "mistralrs ser[v]e" 2>/dev/null; sleep 5
  systemctl --user reset-failed titan-dense.service 2>/dev/null
  systemd-run --user --unit=titan-dense --property=WorkingDirectory=/home/dingo/src/titan-engine "$@" >/dev/null 2>&1
  for i in $(seq 1 40); do sleep 3; ss -tln | grep -q ':1234' && break; done
}
cap() { python3 - "$1" <<'PY'
import json,sys,urllib.request
b={"model":"default","messages":[{"role":"user","content":"Write 120 words on why memory bandwidth limits LLM token generation."}],"temperature":0.0,"max_tokens":160}
r=urllib.request.Request("http://127.0.0.1:1234/v1/chat/completions",data=json.dumps(b).encode(),headers={"Content-Type":"application/json"})
with urllib.request.urlopen(r,timeout=600) as f: d=json.loads(f.read().decode())
m=d["choices"][0]["message"]
open(sys.argv[1],"w").write((m.get("content") or "")+(m.get("reasoning_content") or ""))
PY
}
for v in 1 0 1; do
  echo "=== TITAN_Q3K_ROWS=$v ==="
  start --setenv=TITAN_MTP=0 --setenv=TITAN_Q3K_ROWS=$v -- /home/dingo/src/titan-engine/serve-dense.sh 65536
  cd /tmp && python3 decode-once.py 16 >/dev/null 2>&1; python3 decode-once.py 256
  cap /tmp/final-$v-$(date +%s).txt
  ls -t /tmp/final-$v-*.txt | head -1 > /tmp/last-$v
  cd /home/dingo/src/titan-engine
done
python3 - <<'PY'
import difflib, glob
a=open(open('/tmp/last-1').read().strip()).read()
b=open(open('/tmp/last-0').read().strip()).read()
print("rows1 vs rows0 text:", "IDENTICAL" if a==b else f"DIFFERS {difflib.SequenceMatcher(None,a,b).ratio():.4f} ({len(a)} vs {len(b)} chars)")
PY
