#!/bin/bash
# Profile llama.cpp's decode the same way titan was profiled, for a like-for-like comparison:
# same file, same ctx, greedy, 128 tokens, nsys kernel summary.
set -u
systemctl --user stop titan-dense.service 2>/dev/null
systemctl --user stop optim-llama.service 2>/dev/null
sleep 3
OUT=/tmp/llama-dec
rm -rf ${OUT}.nsys-rep ${OUT}.sqlite
cd /home/dingo
nsys profile -f true --trace=cuda -o "$OUT" --sample=none \
  /home/dingo/.unsloth/llama.cpp/build/bin/llama-server \
  -m /home/dingo/models/optim-Q3_K_M.gguf --alias optim --host 127.0.0.1 --port 8000 \
  -ngl 99 -ot "token_embd=CPU" -c 65536 -np 1 -ub 64 -b 128 \
  -ctk q4_0 -ctv q4_0 -fa on --jinja >/tmp/llama-nsys.log 2>&1 &
NPID=$!
for i in $(seq 1 60); do sleep 3; curl -s -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1 && break; done
python3 - <<'PY'
import json, time, urllib.request
URL="http://127.0.0.1:8000/v1/chat/completions"
def call(n):
    b={"model":"optim","messages":[{"role":"user","content":"Write a long detailed technical explanation of how a Linux kernel scheduler decides which process runs next. Be thorough and concrete."}],"temperature":0.0,"max_tokens":n}
    r=urllib.request.Request(URL,data=json.dumps(b).encode(),headers={"Content-Type":"application/json"})
    t0=time.time()
    with urllib.request.urlopen(r,timeout=1200) as f: d=json.loads(f.read().decode())
    return d["usage"], time.time()-t0
call(8)
u,w=call(128)
print(f"llama.cpp decode {u['completion_tokens']} tok wall {w:.1f}s -> {u.get('completion_tokens',0)/w:.1f} tok/s "
      f"(server-reported {u.get('avg_compl_tok_per_sec') or u.get('completion_tokens_per_second') or 'n/a'})")
PY
sleep 1
kill -INT $NPID 2>/dev/null; sleep 15; kill -9 $NPID 2>/dev/null
nsys stats --report cuda_gpu_kern_sum --format csv "$OUT".nsys-rep 2>/dev/null > /tmp/llama-kern.csv
echo "=== llama.cpp kernels (top by total) ==="
python3 - <<'PY'
rows=[]
for line in open('/tmp/llama-kern.csv'):
    line=line.strip()
    if ',' not in line or line.startswith(('Time','NOTICE','Process','SKIP')): continue
    p=line.split(',')
    try: tot=int(p[1]); n=int(p[2]); avg=float(p[3]); name=','.join(p[8:]).strip().strip('"')
    except: continue
    rows.append((tot,n,avg,name))
if not rows:
    print("no kernel data")
STEPS=136.0
grand=sum(r[0] for r in rows)
print(f"total GPU kernel time {grand/1e9:.2f}s over {int(STEPS)} decode steps -> {grand/1e9/STEPS*1000:.2f} ms/step")
for tot,n,avg,name in sorted(rows,key=lambda r:-r[0])[:8]:
    print(f"  {tot/1e9/STEPS*1000:6.2f} ms/step  {n/STEPS:6.0f}/step  {avg/1000:6.1f}us/call  {name[:56]}")
PY
