trap 'systemctl --user start titan-mistral' EXIT
systemctl --user stop titan-mistral
# Integration check window: mistral.rs fork, branch ktrace @ 8cff156 (chunked-prefill + cpu-twin-avx2 +
# fmt-mmq + qwen3next + gpt-oss-gguf + p1-prefetch merged; hand-resolved conflict in
# quantized_qwen3_moe.rs forward_act_lookahead). Run as:
#   flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/integ/window.sh > ~/titan-engine/m4/integ/window-$(date +%H%M).log 2>&1
# Steps: 1) cuda build (target-oss, warm)  2) nvcc-free/oxide build (target-oss-oxide, warm)
#        3) regression gates on the oxide binary: 35B vs m3/out/q35-prof.json, gpt-oss 20b vs fafce74,
#           80B 8-prompt vs m4/next/out/w2-8-final.json  4) deploy iff all gates pass.
trap 'exit 130' INT TERM HUP
set -u
E=$HOME/titan-engine; I=$E/m4/integ; M=$HOME/ai/models
mkdir -p $I/out $I/logs
CUDA_BIN=$E/target-oss/release/mistralrs
OXIDE_BIN=$E/target-oss-oxide/release/mistralrs
F80=$M/qwen3-next-80b/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf
G20=$M/gpt-oss-20b-F16.gguf
T0=$(date +%s); DEADLINE=$((T0 + 57 * 60))
el() { echo "$(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
cap() { local t=$1 l=$(( DEADLINE - $(date +%s) - 30 )); echo $(( t < l ? t : l )); }
cold80() { python3 -c "import os;fd=os.open('$F80',os.O_RDONLY);os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)"; }

echo "== window start $(date -Is), titan-mistral: $(systemctl --user is-active titan-mistral)"
free -m | head -2
for p in $(ps -eo pid=,ppid=,comm= | awk '$2 == 878468 && $3 ~ /rust-analyzer/ {print $1}'); do echo "killing rust-analyzer $p"; kill $p; done

BUILD_CUDA_RC=1; BUILD_OXIDE_RC=1
GATE_Q35_RC=1; GATE_OSS20_RC=1; GATE_80B_RC=1
Q35_TOKS=""; OSS20_STOCK_TOKS=""; OSS20_T50_TOKS=""; B80_TOKS=""

# ---- step 1: CUDA build ----
echo "== step 1: cuda build ($(el))"
L=$I/logs/build-cuda.log; t0=$(date +%s)
timeout $(cap 1530) systemd-run --user --unit=integ-build-cuda --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 \
  -p RuntimeMaxSec=1500 -p WorkingDirectory=$E/mistral.rs \
  --setenv=CARGO_BUILD_JOBS=2 --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 --setenv=CUDA_COMPUTE_CAP=120 \
  --setenv=CUDA_HOME=/usr/local/cuda --setenv=NVCC_CCBIN=/usr/bin/g++-15 --setenv=CUDA_NVCC_FLAGS=-Wno-template-body \
  "--setenv=RUSTFLAGS=-L $E/lib" --setenv=CARGO_TARGET_DIR=$E/target-oss \
  --setenv=PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:/usr/bin:/bin \
  -p StandardOutput=truncate:$L -p StandardError=truncate:$L nice -n 10 cargo build --release -p mistralrs-cli --features cuda
BUILD_CUDA_RC=$?; systemctl --user stop integ-build-cuda 2>/dev/null; grep -q "^error" $L && BUILD_CUDA_RC=1
echo "cuda build rc=$BUILD_CUDA_RC in $(( $(date +%s) - t0 ))s ($(el))"
grep -E "^(warning: unused|error)|Finished" -A9 $L | head -80
ls -la $CUDA_BIN 2>&1
if [ $BUILD_CUDA_RC != 0 ]; then
  echo "== CUDA BUILD FAILED: ending window here, no gates, no deploy"
  echo "== window end $(el), downtime so far"
  exit 1
fi

# ---- step 2: nvcc-free (oxide) build ----
echo "== step 2: oxide build ($(el))"
L=$I/logs/build-oxide.log; t0=$(date +%s)
command -v nvcc >/dev/null && echo "WARNING: nvcc reachable on PATH"
timeout $(cap 1530) systemd-run --user --unit=integ-build-oxide --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 \
  -p RuntimeMaxSec=1500 -p WorkingDirectory=$E/mistral.rs \
  --setenv=CARGO_TARGET_DIR=$E/target-oss-oxide --setenv=TITAN_OXIDE_DIR=$E/oxide-kernels --setenv=CUDA_HOME=$E/nocuda-bin \
  --setenv=CUDA_PATH=$E/nocuda-bin --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDARC_CUDA_VERSION=13030 --setenv=CARGO_BUILD_JOBS=2 \
  --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 "--setenv=RUSTFLAGS=-L $E/lib" --setenv=PATH=$HOME/.cargo/bin:/usr/bin:/bin \
  -p StandardOutput=truncate:$L -p StandardError=truncate:$L nice -n 10 cargo build --release -p mistralrs-cli --features oxide
BUILD_OXIDE_RC=$?; systemctl --user stop integ-build-oxide 2>/dev/null; grep -q "^error" $L && BUILD_OXIDE_RC=1
echo "oxide build rc=$BUILD_OXIDE_RC in $(( $(date +%s) - t0 ))s ($(el))"
grep "titan:" $L | sort -u
grep -iE "nvcc" $L | grep -v "nvcc not used\|titan:" | head -5
grep -E "Finished|^error" -A8 $L | head -40
ls -la $OXIDE_BIN 2>&1
if [ $BUILD_OXIDE_RC != 0 ]; then
  echo "== OXIDE BUILD FAILED: ending window here, no gates, no deploy"
  echo "== window end $(el)"
  exit 1
fi

# ---- step 3a: 35B regression gate (nvcc-free binary) ----
if [ $(left) -gt 700 ]; then
  echo "== step 3a: 35B gate vs m3/out/q35-prof.json ($(el))"
  n=integ-q35
  timeout $(cap 900) systemd-run --user --unit=$n --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 \
    --setenv=BIN=$OXIDE_BIN --setenv=DIR=$M --setenv=FILE=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    bash $E/m3/collect.sh $n $E/m4/prompts-eval.txt 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt \
    TITAN_TIERED_MMAP=1 TITAN_TIERED_POPULATE=1 TITAN_TIERED_PAGEOUT=1 TITAN_TIERED_LOOKAHEAD=2 TITAN_TIERED_LOOKAHEAD_POPULATE=1 TITAN_TIERED_TIMING=1 2>&1 | grep -v " INFO "
  grep -m1 -o 'titan tiered experts: .*' $E/m3/out/$n.server.log
  grep -o 'decode hit rate.*' $E/m3/out/$n.server.log | tail -1
  python3 -c "
import json,sys
ref=json.load(open('$E/m3/out/q35-prof.json'))
new=json.load(open('$E/m3/out/$n.json'))
n=len(ref); c=sum(x==y for x,y in zip(ref,new))
print(f'35B $n vs q35-prof: {c}/{n} identical (len(new)={len(new)})')
sys.exit(0 if (c==n and len(new)==n) else 1)
"
  GATE_Q35_RC=$?
else
  echo "== step 3a: skipped, $(left)s left"
fi

# ---- step 3b: gpt-oss 20b gate (nvcc-free binary) ----
if [ $(left) -gt 500 ]; then
  echo "== step 3b: gpt-oss 20b gate, stock vs tiered 50%, vs fafce74 baseline ($(el))"
  cd $E/m5/oss
  BIN=$OXIDE_BIN timeout $(cap 600) ./pair.sh $G20 0 integ-oss20-stock "integ-oss20-t50 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=0.5"
  python3 -c "
import json,sys
stock=json.load(open('out/integ-oss20-stock.json')); t50=json.load(open('out/integ-oss20-t50.json'))
ref_s=json.load(open('out/oss20-stock.json')); ref_t=json.load(open('out/oss20-t50.json'))
n=len(stock)
c1=sum(x==y for x,y in zip(stock,t50)); c2=sum(x==y for x,y in zip(stock,ref_s)); c3=sum(x==y for x,y in zip(t50,ref_t))
print(f'20b stock vs t50: {c1}/{n}; stock vs fafce74-stock: {c2}/{n}; t50 vs fafce74-t50: {c3}/{n}')
ok = c1==n and c2==n and c3==n and len(t50)==n and len(ref_s)==n and len(ref_t)==n
sys.exit(0 if ok else 1)
"
  GATE_OSS20_RC=$?
  cd $E
else
  echo "== step 3b: skipped, $(left)s left"
fi

# ---- step 3c: 80B gate, 8-prompt set, recommended config (nvcc-free binary) ----
if [ $(left) -gt 700 ]; then
  echo "== step 3c: 80B gate (P1 L+2 + pageout + lookahead-populate), 8-prompt set ($(el))"
  cd $E/m4/next
  cold80
  FIN="TITAN_TIERED_PROFILE=$E/m4/next/profile-next.txt TITAN_TIERED_LOOKAHEAD=2 TITAN_TIERED_LOOKAHEAD_POPULATE=1 TITAN_TIERED_PAGEOUT=1"
  BIN=$OXIDE_BIN timeout $(cap 900) ./pair.sh $F80 37 "integ-next8-final TITAN_TIERED=1 TITAN_TIERED_MMAP=1 TITAN_TIERED_GPU_FRACTION=auto $FIN"
  python3 -c "
import json,sys
ref=json.load(open('out/w2-8-final.json'))
new=json.load(open('out/integ-next8-final.json'))
n=len(ref); c=sum(x==y for x,y in zip(ref,new))
print(f'80B integ-next8-final vs w2-8-final: {c}/{n} identical (len(new)={len(new)})')
sys.exit(0 if (c==n and len(new)==n) else 1)
"
  GATE_80B_RC=$?
  cd $E
else
  echo "== step 3c: skipped, $(left)s left"
fi

echo "== gate summary ($(el)): cuda_build=$BUILD_CUDA_RC oxide_build=$BUILD_OXIDE_RC q35=$GATE_Q35_RC oss20=$GATE_OSS20_RC 80b=$GATE_80B_RC"

# ---- step 4: deploy iff every gate passed ----
DEPLOYED=0
if [ $BUILD_CUDA_RC = 0 ] && [ $BUILD_OXIDE_RC = 0 ] && [ $GATE_Q35_RC = 0 ] && [ $GATE_OSS20_RC = 0 ] && [ $GATE_80B_RC = 0 ]; then
  echo "== all gates passed: deploying oxide binary ($(el))"
  cp $OXIDE_BIN $E/bin/mistralrs-titan-integ && chmod +x $E/bin/mistralrs-titan-integ
  sed -i 's#/titan-engine/bin/mistralrs-titan-chunked#/titan-engine/bin/mistralrs-titan-integ#' $E/deploy/titan-mistral.service
  cp $E/deploy/titan-mistral.service $HOME/.config/systemd/user/titan-mistral.service
  systemctl --user daemon-reload
  DEPLOYED=1
  echo "DEPLOYED=1"
else
  echo "== one or more gates failed: NOT deploying; bin/ and unit left untouched ($(el))"
  echo "DEPLOYED=0"
fi
echo "== window end $(date -Is), downtime $(el)"
