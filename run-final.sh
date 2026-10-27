#!/bin/bash
set -u
start() {
  systemctl --user stop titan-dense.service 2>/dev/null; sleep 2
  systemctl --user reset-failed titan-dense.service 2>/dev/null
  systemd-run --user --unit=titan-dense --property=WorkingDirectory=/home/dingo/src/titan-engine "$@" >/dev/null 2>&1
  for i in $(seq 1 40); do sleep 3; ss -tln | grep -q ':1234' && break; done
}
echo "=== default ceiling (mmvq 1 row), paged ON, batch 16 ==="
start --setenv=TITAN_MTP=0 --setenv=TITAN_PAGED=on --setenv=TITAN_BATCH=16 \
      --setenv=TITAN_RECURRENT_SLOTS=16 -- /home/dingo/src/titan-engine/serve-dense.sh 32768
python3 - <<'PY'
import json, sys, threading, time, urllib.request
URL="http://127.0.0.1:1234/v1/chat/completions"
P=("Write a long detailed technical explanation of how a Linux kernel scheduler decides which "
   "process runs next. Be thorough and concrete.")
def one(i,n,out):
    b={"model":"default","messages":[{"role":"user","content":f"req {i}: {P}"}],"temperature":0.0,"max_tokens":n}
    r=urllib.request.Request(URL,data=json.dumps(b).encode(),headers={"Content-Type":"application/json"})
    t0=time.time()
    try:
        with urllib.request.urlopen(r,timeout=1800) as f: d=json.loads(f.read().decode())
        out[i]=(d.get("usage",{}).get("completion_tokens",0),time.time()-t0,d)
    except Exception as e:
        out[i]=(0,time.time()-t0,None); print("  fail",e)
def run(k,n,keep=False):
    out={}; th=[threading.Thread(target=one,args=(i,n,out)) for i in range(k)]
    t0=time.time()
    for t in th: t.start()
    for t in th: t.join()
    w=time.time()-t0; toks=sum(v[0] for v in out.values()); per=[v[0]/v[1] for v in out.values() if v[1]>0]
    print(f"  {k:2d} concurrent: {toks:5d} tok in {w:6.1f}s -> aggregate {toks/w:6.1f} tok/s (per-stream {min(per):.1f}-{max(per):.1f})")
    return out
run(1,16); solo=run(1,96)
for k in (2,4,8,16): run(k,96)
open("/tmp/txt-single.txt","w").write(solo[0][2]["choices"][0]["message"]["content"])
PY
echo "--- text check: same prompt batched vs solo ---"
python3 - <<'PY'
import json, urllib.request, difflib
URL="http://127.0.0.1:1234/v1/chat/completions"
P=("Write a long detailed technical explanation of how a Linux kernel scheduler decides which "
   "process runs next. Be thorough and concrete.")
b={"model":"default","messages":[{"role":"user","content":f"req 0: {P}"}],"temperature":0.0,"max_tokens":96}
r=urllib.request.Request(URL,data=json.dumps(b).encode(),headers={"Content-Type":"application/json"})
with urllib.request.urlopen(r,timeout=900) as f: d=json.loads(f.read().decode())
a=open("/tmp/txt-single.txt").read(); c=d["choices"][0]["message"]["content"]
print("identical" if a==c else f"differ: {difflib.SequenceMatcher(None,a,c).ratio():.3f} similarity")
PY
