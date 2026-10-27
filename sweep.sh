#!/bin/bash
# Decode/prefill sweep on the live titan unit: one restart per config, one
# 300-token greedy decode + one 4.6k-token prefill each. Prints a table.
set -u
CTX=${CTX:-65536}
PROMPT="Write a plain-text numbered list of things a Linux desktop agent should check on a machine before it claims the machine is healthy. Keep going until you have 60 items."
FILL=$(python3 -c "print('The quick brown fox jumps over the lazy dog. '*450)")

measure() { # label, extra VAR=VAL assignments
  local label=$1; shift
  local setenv=(--setenv=TITAN_PREFILL_BIG_CHUNK=0 --setenv=TITAN_RECURRENT_SLOTS=2
                --setenv=TITAN_ADMIT=log --setenv=TITAN_CUDA_GRAPHS=1)
  local kv
  for kv in "$@"; do setenv+=("--setenv=$kv"); done
  systemctl --user stop titan-dense.service 2>/dev/null
  sleep 2
  systemctl --user reset-failed titan-dense.service 2>/dev/null
  systemd-run --user --unit=titan-dense \
    --property=WorkingDirectory=/home/dingo/src/titan-engine \
    "${setenv[@]}" \
    /home/dingo/src/titan-engine/serve-dense.sh "$CTX" >/dev/null 2>&1
  for i in $(seq 1 40); do
    sleep 3
    ss -tln | grep -q ':1234' && break
  done
  python3 - "$label" "$PROMPT" "$FILL" <<'PY'
import json, sys, time, urllib.request
label, prompt, fill = sys.argv[1], sys.argv[2], sys.argv[3]
def call(text, mt):
    body = {"model":"default","messages":[{"role":"user","content":text}],"temperature":0.0,"max_tokens":mt}
    req = urllib.request.Request("http://127.0.0.1:1234/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type":"application/json"})
    t0=time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r: d=json.loads(r.read().decode())
    except Exception as e:
        return None, time.time()-t0, str(e)[:60]
    return d.get("usage",{}), time.time()-t0, ""
u1,w1,e1 = call(prompt, 300)
u2,w2,e2 = call(f"req{time.time_ns()} "+fill+"\nReply ok", 8)
dec = f"{u1.get('avg_compl_tok_per_sec',0):6.1f}" if u1 else f"  ERR({e1})"
pre = f"{u2.get('avg_prompt_tok_per_sec',0):7.1f}" if u2 else f"  ERR({e2})"
print(f"{label:34s} decode={dec} tok/s   prefill@{(u2 or {}).get('prompt_tokens',0):5d}={pre} tok/s   wall={w1:6.1f}s/{w2:6.1f}s")
PY
  systemctl --user stop titan-dense.service 2>/dev/null
  sleep 2
}

measure "MTP=2 (default)"
measure "MTP=0" TITAN_MTP=0
measure "MTP=1" TITAN_MTP=1
measure "MTP=2 no-cuda-graphs" TITAN_CUDA_GRAPHS=0
measure "MTP=0 slot1" TITAN_MTP=0 TITAN_RECURRENT_SLOTS=1
