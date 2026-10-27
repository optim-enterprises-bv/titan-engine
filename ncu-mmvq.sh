#!/bin/bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO/bench-lib.sh"
# Per-launch DRAM throughput for the q3_k matvec, split by grid size (= matrix shape).
# Decides whether the tall-skinny projections are bandwidth-starved or latency-bound.
set -u
systemctl --user stop titan-dense.service 2>/dev/null; sleep 3
export TITAN_MTP=0 TITAN_GDN_DIRECT=1
OUT=/tmp/ncu-mmvq
rm -rf "$OUT"*
/opt/nvidia/nsight-compute/2026.3.1/ncu --target-processes all --csv \
  --kernel-name regex:mmvq_gguf_q3_k_bf16 \
  --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,launch__grid_size,launch__block_size,launch__registers_per_thread \
  --launch-count 400 \
  -o "$OUT" \
  /home/dingo/src/titan-engine/mistral.rs/target/release/mistralrs serve \
  -p 1234 --no-ui --paged-attn off --max-seq-len 8192 --format gguf \
  -m models-optim -f optim-Q3_K_M.gguf -n 0:64 >/tmp/ncu.log 2>&1 &
NPID=$!
for i in $(seq 1 40); do sleep 3; ss -tln | grep -q ':1234' && break; done
cd /tmp && python3 "$REPO/decode-once.py" 8 >/dev/null 2>&1
python3 "$REPO/decode-once.py" 12 >/dev/null 2>&1
sleep 2
kill -INT $NPID 2>/dev/null; sleep 20; kill -9 $NPID 2>/dev/null
ls -la "$OUT".ncu-rep 2>/dev/null | awk '{print "report bytes:", $5}'
/opt/nvidia/nsight-compute/2026.3.1/ncu --import "$OUT".ncu-rep --csv --page raw 2>/dev/null > /tmp/ncu.csv
python3 - <<'PY'
import csv, collections
try:
    rows=list(csv.DictReader(open('/tmp/ncu.csv')))
except Exception as e:
    print("parse failed:", e); rows=[]
if not rows:
    print("no rows"); raise SystemExit
def g(r,*keys):
    for k in keys:
        for kk in r:
            if kk and k.lower() in kk.lower(): return r[kk]
    return None
agg=collections.defaultdict(lambda:[0,0.0,0.0])
for r in rows:
    grid=g(r,'launch__grid_size'); dur=g(r,'gpu__time_duration.sum','gpu__time_duration')
    byt=g(r,'dram__bytes_read.sum'); pct=g(r,'dram__throughput.avg.pct_of_peak')
    if grid is None or dur is None: continue
    def num(x):
        if x is None: return 0.0
        x=str(x).replace(',','').strip()
        mult=1
        for suf,m in (('us',1e-6),('ms',1e-3),('ns',1e-9),('msecond',1e-3),('usecond',1e-6)):
            if x.endswith(suf): x=x[:-len(suf)]; mult=m; break
        try: return float(x)*mult
        except: return 0.0
    key=int(float(str(grid).replace(',','')))
    a=agg[key]; a[0]+=1; a[1]+=num(dur); a[2]+=float(str(pct).replace('%','') or 0)
print(f"{'grid(blocks)':>13} {'calls':>6} {'us/call':>8} {'DRAM GB/s/call':>15} {'%peak':>7}")
for k in sorted(agg):
    n,d,p=agg[k]
    us=d/n*1e6 if n else 0
    print(f"{k:13d} {n:6d} {us:8.1f} {'':15} {p/n:7.1f}")
PY
