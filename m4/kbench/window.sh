#!/bin/bash
# kbench window: llama.cpp (acecd56 build, matmul code == upstream HEAD for sm_120) vs our cuda-oxide
# kernels on the 35B / 80B shapes. Measurement only, no engine changes.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/kbench/window.sh
set -u
E=$HOME/titan-engine; K=$E/m4/kbench; O=$K/out
NSYS=/opt/nvidia/nsight-compute/2026.2.1/host/target-linux-x64/nsys
NCU=/usr/local/cuda/bin/ncu
exec > >(tee -a $O/window-$(date +%Y%m%d-%H%M).log) 2>&1
echo "== kbench window: lock acquired $(date -Is)"
# 30-minute service gap (RULES-agents.md, 2026-09-28 15:10)
idle=0
while :; do
  if [ "$(systemctl --user is-active titan-mistral)" != active ]; then
    idle=$((idle + 1)); [ $idle -ge 20 ] && { echo "titan-mistral inactive for 10 min: starting it"; systemctl --user start titan-mistral; idle=0; }
  else
    t=$(systemctl --user show titan-mistral -p ActiveEnterTimestampMonotonic --value)
    now=$(awk '{print int($1*1000000)}' /proc/uptime)
    [ $(( (now - t) / 1000000 )) -ge $( [ -e $HOME/titan-engine/.nogap ] && echo 0 || echo 1800 ) ] && break
  fi
  sleep 30
done
echo "== service gap satisfied $(date -Is); stopping titan-mistral"
trap 'systemctl --user start titan-mistral; echo "== titan-mistral restarted $(date -Is)"' EXIT
trap 'exit 130' INT TERM HUP
systemctl --user stop titan-mistral
T0=$(date +%s); DEADLINE=$((T0 + 55 * 60))
el() { echo "$(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
echo "== window start $(date -Is), titan-mistral: $(systemctl --user is-active titan-mistral)"
for p in $(ps -eo pid=,ppid=,comm= | awk '$2 == 878468 && $3 ~ /rust-analyzer/ {print $1}'); do echo "killing rust-analyzer $p"; kill $p; done
free -m | head -2
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,power.draw,temperature.gpu,memory.used --format=csv 2>&1 | head -3
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 | head -5

echo "== build ($(el))"
timeout 300 systemd-run --user --collect --wait --pipe -q -p MemoryMax=4G -p MemorySwapMax=0 bash $K/build.sh
ls -la $K/kbench || { echo "no binary: ending the window"; exit 1; }

export CUDA_CACHE_MAXSIZE=4294967296
echo "== smoke ($(el))"
timeout 900 systemd-run --user --collect --wait --pipe -q -p MemoryMax=8G -p MemorySwapMax=0 --setenv=CUDA_CACHE_MAXSIZE=4294967296 \
  $K/kbench q35.exp_gate_up.q4_K.k2048n512/b1 q35.qkv.q8_0.k2048n8192/b3 q35.qkv.q8_0.k2048n8192/b512 2>&1 | grep -v "^ggml_cuda_init\|Device 0" | tail -12

echo "== nsys run ($(el))"
timeout 2100 systemd-run --user --collect --wait --pipe -q -p MemoryMax=10G -p MemorySwapMax=0 -p RuntimeMaxSec=2080 \
  --setenv=CUDA_CACHE_MAXSIZE=4294967296 --setenv=GGML_CUDA_DISABLE_GRAPHS=1 \
  $NSYS profile -t cuda,nvtx --sample=none --cpuctxsw=none -o $O/kbench -f true bash $K/runall.sh
echo "nsys rc=$? ($(el))"
timeout 600 $NSYS export --type sqlite -o $O/kbench.sqlite -f true $O/kbench.nsys-rep > /dev/null 2>&1; ls -la $O/kbench.sqlite
timeout 300 python3 $K/analyze.py 2>&1 | tail -60

echo "== ncu ($(el), $(left)s left)"
ncu1() { # ncu1 TAG CASE KERNEL_REGEX
  [ $(left) -gt 240 ] || { echo "skip ncu $1"; return; }
  timeout 300 sudo -n env KBENCH_NO_ALT=1 $NCU --section SpeedOfLight --section Occupancy --section LaunchStats --section MemoryWorkloadAnalysis \
    -k "regex:$3" --launch-skip 3 --launch-count 1 $K/kbench "$2" > $O/ncu-$1.txt 2>&1
  echo "ncu $1 rc=$?"; grep -E "Duration|Memory Throughput|DRAM Throughput|Achieved Occupancy|Theoretical Occupancy|Registers Per|Grid Size|Block Size" $O/ncu-$1.txt | head -12
}
ncu1 gu-b1-ours  q35.exp_gate_up.q4_K.k2048n512/b1 '^q4k_q8_1_moe_gemv$'
ncu1 gu-b1-llama q35.exp_gate_up.q4_K.k2048n512/b1 '^mul_mat_vec_q'
ncu1 dn-b1-ours  q35.exp_down.q5_K.k512n2048/b1 '^q5k_q8_1_moe_gemv_w$'
ncu1 dn-b1-llama q35.exp_down.q5_K.k512n2048/b1 '^mul_mat_vec_q'
ncu1 gu-b3-ours  q35.exp_gate_up.q4_K.k2048n512/b3 '^q4k_q8_1_moe_gemv$'
ncu1 gu-b3-llama q35.exp_gate_up.q4_K.k2048n512/b3 '^mul_mat_vec_q'
ncu1 gu-b512-ours  q35.exp_gate_up.q4_K.k2048n512/b512 '^q4k_q8_1_moe_gemv$'
ncu1 gu-b512-llama q35.exp_gate_up.q4_K.k2048n512/b512 '^mul_mat_q'
sudo -n chown -R "$(id -un):$(id -gn)" $O 2>/dev/null
free -m | head -2
echo "== window end $(el)"
