#!/bin/bash
# A/B the dense bf16-matvec patch on the same binary: default on vs
# TITAN_DENSE_BF16_MATVEC=0. Same prompts, greedy, one process at a time.
set -u
CTX=${CTX:-65536}
run() { # label, extra setenv pairs...
  local label=$1; shift
  systemctl --user stop titan-dense.service 2>/dev/null; sleep 2
  systemctl --user reset-failed titan-dense.service 2>/dev/null
  local se=(--setenv=TITAN_MTP=0 --setenv=TITAN_CUDA_GRAPHS=1
            --setenv=TITAN_PREFILL_BIG_CHUNK=0 --setenv=TITAN_RECURRENT_SLOTS=2)
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
    u=d.get("usage",{}); msg=d["choices"][0]["message"]
    return u, (msg.get("content") or ""), time.time()-t0
# warm
call("say ok", 4)
dec_u, dec_txt, dec_w   = call("List 60 checks a Linux desktop agent should run before calling a machine healthy. Plain numbered list.", 300)
fill = "The quick brown fox jumps over the lazy dog. "*450
pre_u, _, pre_w          = call(f"req{time.time_ns()} {fill}\nReply ok", 8)
long_u, long_txt, long_w = call(f"req{time.time_ns()} {fill}\n\nSummarise the sentence above in five words.", 60)
print(f"{label:28s} decode={dec_u.get('avg_compl_tok_per_sec',0):6.1f} tok/s "
      f"({dec_u.get('completion_tokens')} tok, {dec_w:.1f}s)  "
      f"prefill@4.6k={pre_u.get('avg_prompt_tok_per_sec',0):6.0f} tok/s  "
      f"long-decode@4.6k={long_u.get('avg_compl_tok_per_sec',0):6.1f} tok/s  "
      f"long-wall={long_w:.1f}s")
open(f"/tmp/out-{label.replace(' ','_')}.txt","w").write(dec_txt)
PY
  systemctl --user stop titan-dense.service 2>/dev/null; sleep 2
}
run "patched-default"
run "matvec-f32-old" TITAN_DENSE_BF16_MATVEC=0
echo "--- greedy output diff (first 40 lines each) ---"
diff <(head -40 "/tmp/out-patched-default.txt") <(head -40 "/tmp/out-matvec-f32-old.txt") > /tmp/patch-diff.txt && echo "IDENTICAL first 40 lines" || { echo "DIFFERS:"; head -20 /tmp/patch-diff.txt; }
