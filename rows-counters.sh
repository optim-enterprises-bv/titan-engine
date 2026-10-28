#!/bin/bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO/bench-lib.sh"
# Controlled comparison of the rows kernel at 1 vs 2 activation rows: same weights, same grid
# (the rows kernel always does 2 weight rows per block, so grid 8704 == the 17408-row FFN matrices
# in BOTH cases). ncu as root, DRAM vs SM throughput decides "bandwidth-amortised" vs "ALU-bound".
set -u
systemctl --user stop titan-dense.service 2>/dev/null
sudo -n pkill -f "mistralrs ser[v]e" 2>/dev/null
sleep 6
free=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr -dc '0-9')
[ "${free:-9999}" -gt 2000 ] && { echo "GPU busy (${free} MiB), aborting"; exit 1; }

NCU=/opt/nvidia/nsight-compute/2026.3.1/ncu
rm -rf /tmp/rowsq.ncu-rep* /tmp/rowsq.csv /tmp/rowsq.log
sudo -n env TITAN_MTP=1 TITAN_GDN_DIRECT=1 \
  "$NCU" --target-processes all \
  --kernel-name regex:mmvq_gguf_q3_k_bf16_rows2 \
  --metrics gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,launch__grid_size,launch__block_size \
  --launch-count 40 --kill yes \
  -f -o /tmp/rowsq \
  ${TITAN_BIN:-/usr/local/bin/mistralrs} serve \
  -p 1234 --no-ui --paged-attn off --max-seq-len 8192 --format gguf \
  -m /home/dingo/src/titan-engine/models-optim \
  -f Qwen3.8-27B-DevSecOps-QLoRA-abliterated-Q3_K_M.gguf -n 0:65 \
  > /tmp/rowsq.log 2>&1 &
for i in $(seq 1 60); do sleep 3; ss -tln | grep -q ':1234' && break; done
cd /tmp && python3 "$REPO/decode-once.py" 8 >/dev/null 2>&1; python3 "$REPO/decode-once.py" 30 >/dev/null 2>&1
for i in $(seq 1 40); do sleep 5; [ -f /tmp/rowsq.ncu-rep ] && break; [ -f /tmp/rowsq.ncu-repz ] && break; done
sudo -n "$NCU" --import /tmp/rowsq.ncu-repz --csv --page raw > /tmp/rowsq.csv 2>/dev/null
[ -s /tmp/rowsq.csv ] || sudo -n "$NCU" --import /tmp/rowsq.ncu-rep --csv --page raw > /tmp/rowsq.csv 2>/dev/null
sudo -n pkill -f "mistralrs ser[v]e" 2>/dev/null
python3 - <<'PY'
import csv, collections
rows=[r for r in csv.DictReader(open('/tmp/rowsq.csv')) if (r.get('Kernel Name') or '').strip()]
def f(r,k):
    try: return float((r.get(k) or '0').replace(',',''))
    except: return 0.0
agg=collections.defaultdict(lambda:[0,0.0,0.0,0.0])
for r in rows:
    key=(r['Kernel Name'].replace('mmvq_gguf_q3_k_bf16_',''), int(f(r,'launch__grid_size')))
    a=agg[key]; a[0]+=1
    a[1]+=f(r,'gpu__time_duration.sum'); a[2]+=f(r,'dram__throughput.avg.pct_of_peak_sustained_elapsed')
    a[3]+=f(r,'sm__throughput.avg.pct_of_peak_sustained_elapsed')
print(f"{'kernel':10s} {'grid':>6} {'n':>3} {'us':>7} {'DRAM%':>6} {'SM%':>6}")
for (k,g),(n,d,dr,sm) in sorted(agg.items(), key=lambda kv:-kv[1][0]):
    print(f"{k:10s} {g:6d} {n:3d} {d/n:7.1f} {dr/n:6.1f} {sm/n:6.1f}")
PY
