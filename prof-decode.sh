#!/bin/bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO/bench-lib.sh"
# Profile a decode-heavy window: server under nsys, 128-token greedy decode, then stats.
set -u
cd /home/dingo/src/titan-engine
bench_stop_prod; sleep 2
export TITAN_MTP=0 TITAN_GDN_DIRECT=1
OUT=/tmp/titan-dec
rm -rf ${OUT}.nsys-rep ${OUT}.sqlite
nsys profile -f true --trace=cuda -o "$OUT" --sample=none \
  /home/dingo/src/titan-engine/mistral.rs/target/release/mistralrs serve \
  -p 1234 --no-ui --paged-attn off --max-seq-len 65536 --format gguf \
  -m models-optim -f optim-Q3_K_M.gguf -n 0:64 >/tmp/titan-nsys.log 2>&1 &
NPID=$!
for i in $(seq 1 40); do sleep 3; ss -tln | grep -q ':1234' && break; done
cd /tmp && python3 "$REPO/decode-once.py" 8 >/dev/null 2>&1      # warm the model
python3 "$REPO/decode-once.py" 128
sleep 1
kill -INT $NPID 2>/dev/null; sleep 12; kill -9 $NPID 2>/dev/null
nsys stats --report cuda_gpu_kern_sum --format csv "$OUT".nsys-rep 2>/dev/null | head -40 > /tmp/kern.csv
nsys stats --report cuda_gpu_sum --format csv "$OUT".nsys-rep 2>/dev/null | head -6 > /tmp/gpu.csv
echo "=== KERNELS (csv: Time%, Total(ns), Instances, Avg(ns), Name) ==="
cat /tmp/kern.csv
echo "=== GPU ==="
cat /tmp/gpu.csv
