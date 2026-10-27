#!/bin/bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO/bench-lib.sh"
# Q3_K kernel counters, run as root (GPU perf counters are admin-only on this box).
# Decides: DRAM-transaction bound (misaligned 16-bit reads) vs ALU/issue bound.
set -u
systemctl --user stop titan-dense.service 2>/dev/null
pkill -f "mistralrs ser[v]e" 2>/dev/null
sleep 6
free=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr -dc '0-9')
echo "GPU in use before start: ${free} MiB"
[ "${free:-9999}" -gt 2000 ] && { echo "GPU still busy, aborting"; exit 1; }

NCU=/opt/nvidia/nsight-compute/2026.3.1/ncu
rm -rf /tmp/ncu-q3k.ncu-repz /tmp/ncu-q3k.csv /tmp/ncu-q3k.log
sudo -n env TITAN_MTP=0 TITAN_GDN_DIRECT=1 \
  "$NCU" --target-processes all \
  --kernel-name regex:mmvq_gguf_q3_k_bf16_plain_cuda1 \
  --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,launch__grid_size \
  --launch-count 24 --kill yes \
  -f -o /tmp/ncu-q3k \
  /home/dingo/src/titan-engine/mistral.rs/target/release/mistralrs serve \
  -p 1234 --no-ui --paged-attn off --max-seq-len 8192 --format gguf \
  -m /home/dingo/src/titan-engine/models-optim -f optim-Q3_K_M.gguf -n 0:64 \
  > /tmp/ncu-q3k.log 2>&1 &
NPID=$!
for i in $(seq 1 60); do sleep 3; ss -tln | grep -q ':1234' && break; done
cd /tmp && python3 "$REPO/decode-once.py" 8 >/dev/null 2>&1; python3 "$REPO/decode-once.py" 12 >/dev/null 2>&1
for i in $(seq 1 40); do sleep 5; [ -f /tmp/ncu-q3k.ncu-repz ] && break; done
pkill -f "mistralrs ser[v]e" 2>/dev/null
sudo -n "$NCU" --import /tmp/ncu-q3k.ncu-repz --csv --page raw > /tmp/ncu-q3k.csv 2>/dev/null
python3 - <<'PY'
import csv, collections
rows=list(csv.DictReader(open('/tmp/ncu-q3k.csv'))) if open('/tmp/ncu-q3k.csv').read(200).strip() else []
if not rows:
    print("no counter rows — see /tmp/ncu-q3k.log"); 
    import subprocess; print(subprocess.run(["tail","-6","/tmp/ncu-q3k.log"],capture_output=True,text=True).stdout)
    raise SystemExit
def pick(r, frag):
    for k in r:
        if k and frag in k.lower(): return r[k]
    return None
agg=collections.defaultdict(lambda: [0,0.0,0.0,0.0,0.0])
for r in rows:
    g=pick(r,'grid_size'); 
    if not g: continue
    key=int(float(str(g).replace(',','')))
    def num(f):
        v=pick(r,f)
        try: return float(str(v).replace(',','').replace('%','').strip())
        except: return 0.0
    a=agg[key]; a[0]+=1; a[1]+=num('gpu__time_duration'); a[2]+=num('dram__bytes_read')
    a[3]+=num('dram__throughput'); a[4]+=num('sm__throughput')
print(f"{'grid':>6} {'n':>4} {'us':>7} {'MB read':>8} {'GB/s':>7} {'DRAM%':>6} {'SM%':>5}")
for k in sorted(agg):
    n,d,b,dp,sp=agg[k]
    us=d/n if d>1e3 else d*1e6/n     # ns or us
    us = d/n*1e6 if d/n < 1 else d/n/1e3
    mb=b/n/1e6; gbs=mb/ (us/1e3) /1e3
    print(f"{k:6d} {n:4d} {us:7.1f} {mb:8.2f} {gbs:7.0f} {dp/n:6.1f} {sp/n:5.1f}")
PY
