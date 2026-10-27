#!/bin/bash
# A/B the zero-copy GDN state path on the same binary: default (in place) vs
# TITAN_GDN_DIRECT=0 (gather + scatter). Same prompts, greedy; diff the text.
set -u
CTX=${CTX:-65536}
run() { # label, extra setenv...
  local label=$1; shift
  systemctl --user stop titan-dense.service 2>/dev/null; sleep 2
  systemctl --user reset-failed titan-dense.service 2>/dev/null
  local se=(--setenv=TITAN_MTP=0)
  local kv; for kv in "$@"; do se+=("--setenv=$kv"); done
  systemd-run --user --unit=titan-dense --property=WorkingDirectory=/home/dingo/src/titan-engine \
    "${se[@]}" /home/dingo/src/titan-engine/serve-dense.sh "$CTX" >/dev/null 2>&1
  for i in $(seq 1 30); do sleep 3; ss -tln | grep -q ':1234' && break; done
  python3 - "$label" <<'PY'
import json, sys, time, urllib.request
label = sys.argv[1]
def call(text, mt, temp=0.0):
    body={"model":"default","messages":[{"role":"user","content":text}],"temperature":temp,"max_tokens":mt}
    req=urllib.request.Request("http://127.0.0.1:1234/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    t0=time.time()
    with urllib.request.urlopen(req,timeout=900) as r: d=json.loads(r.read().decode())
    return d.get("usage",{}), d["choices"][0]["message"].get("content") or "", time.time()-t0
call("say ok", 4)  # warm
u, txt, w = call("List 60 checks a Linux desktop agent should run before calling a machine healthy. Plain numbered list.", 300)
fill = "The quick brown fox jumps over the lazy dog. "*450
pu, _, _ = call(f"req{time.time_ns()} {fill}\nReply ok", 8)
print(f"{label:26s} decode={u.get('avg_compl_tok_per_sec',0):6.1f} tok/s ({u.get('completion_tokens')} tok, {w:.1f}s)  prefill@4.6k={pu.get('avg_prompt_tok_per_sec',0):6.0f} tok/s")
open(f"/tmp/gdn-{label}.txt","w").write(txt)
PY
  systemctl --user stop titan-dense.service 2>/dev/null; sleep 2
}
run "direct-default"
run "gather-scatter" TITAN_GDN_DIRECT=0
echo "--- greedy text diff ---"
diff -q /tmp/gdn-direct-default.txt /tmp/gdn-gather-scatter.txt >/dev/null && echo "IDENTICAL output" || { echo "OUTPUT DIFFERS:"; diff /tmp/gdn-direct-default.txt /tmp/gdn-gather-scatter.txt | head -10; }
