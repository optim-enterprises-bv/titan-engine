#!/bin/bash
# IQ3_S window 1: build + gate the oxide-kernels/iq3_s port, stage its PTX into candle (top-iq3s),
# check candle's CPU dequantize against llama.cpp, then build mistral.rs (mr-iq3s, nvcc-free,
# --features oxide) into target-iq3s-oxide for as long as the window allows (cargo resumes later).
# User rule (2026-10-02): a window may stop titan-spark once it holds the lock and must restart titan-spark
# (never titan-mistral: its Conflicts= would kill spark) on exit.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/iq3s/window1.sh
set -u
E=$HOME/titan-engine; W=$E/m4/iq3s; C=$E/oxide-iq3s/iq3_s
exec > >(tee -a $W/window1-$(date +%Y%m%d-%H%M).log) 2>&1
T0=$(date +%s); DEADLINE=$((T0 + 57 * 60))
el() { echo "$(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
SPARK=$(systemctl --user is-active titan-spark)
trap 'systemctl --user stop iq3s-ox iq3s-build iq3s-quantb 2>/dev/null; [ "$SPARK" = active ] && systemctl --user start titan-spark; echo "== window1 exit $(date -Is): titan-spark $(systemctl --user is-active titan-spark), titan-mistral $(systemctl --user is-active titan-mistral)"' EXIT
trap 'exit 130' INT TERM HUP
echo "== lock acquired $(date -Is); titan-spark $SPARK, titan-mistral $(systemctl --user is-active titan-mistral) (titan-mistral is never started)"
[ "$SPARK" = active ] && systemctl --user stop titan-spark && echo "stopped titan-spark for the window"
sleep 3
for p in $(ps -eo pid=,ppid=,comm= | awk '$2 == 878468 && $3 ~ /rust-analyzer/ {print $1}'); do echo "killing rust-analyzer $p"; kill $p; done
free -m | head -2; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

echo "== [1] oxide-kernels/iq3_s build ($(el))"
timeout 1500 systemd-run --user --unit=iq3s-ox --collect --wait -q -p MemoryMax=6G -p MemorySwapMax=0 -p RuntimeMaxSec=1480 \
  -p WorkingDirectory=$C --setenv=CUDA_OXIDE_BACKEND=$E/cuda-oxide-fast/librustc_codegen_cuda.so \
  --setenv=CARGO_BUILD_JOBS=2 --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 --setenv=PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:/usr/bin:/bin \
  -p StandardOutput=truncate:$C/build.log -p StandardError=truncate:$C/build.log nice -n 10 cargo oxide build --arch sm_120
rc=$?; echo "oxide build rc=$rc ($(el))"
grep -E "^error|^warning: unused|not unrolled" -A12 $C/build.log | head -60
ls -la $C/iq3_s.ptx $C/target/release/iq3_s 2>&1

echo "== [2] kernel gate ($(el))"
GATE=1
if [ -x $C/target/release/iq3_s ] && [ $C/iq3_s.ptx -nt $C/src/main.rs ]; then
  (cd $C && timeout 600 ./target/release/iq3_s > $W/out/gate-iq3s.log 2>&1); GATE=$?
  grep -E "FAIL|mutation|cpu oracle|PASS|launches" $W/out/gate-iq3s.log | head -40
  echo "gate rc=$GATE"
else
  echo "no fresh kernel binary/PTX: gate not run"
fi

echo "== [3] stage PTX into candle + candle CPU dequantize check ($(el))"
if [ -f $C/iq3_s.ptx ]; then
  cp $C/iq3_s.ptx $E/top-iq3s/candle/candle-core/src/quantized/iq3_s_oxide.ptx
  [ $GATE = 0 ] || echo "WARNING: staged a PTX whose gate did not pass (build still warms the dependency cache)"
fi
if [ -f $C/ref/cpu_deq.bin ]; then
  (cd $W/cpucheck && python3 make.py && timeout 300 rustc -O --edition 2021 check.rs -o check 2>&1 | tail -5 && timeout 120 ./check) | tee $W/out/cpucheck.log
fi
[ -f $E/top-iq3s/candle/candle-core/src/quantized/iq3_s_oxide.ptx ] || { echo "no iq3_s_oxide.ptx: cannot build candle; ending window"; exit 1; }

echo "== [3b] background: 9B IQ3_M variant with an IQ3_S token_embd (CPU, nice 15, 8G cap) ($(el))"
[ -f $W/Qwen3.5-9B-IQ3_M-embIQ3S.gguf ] || systemd-run --user --unit=iq3s-quantb --collect -q -p MemoryMax=8G -p MemorySwapMax=0 -p RuntimeMaxSec=2400 \
  -p StandardOutput=truncate:$W/quantb.log -p StandardError=truncate:$W/quantb.log nice -n 15 \
  sh -c "$HOME/ai/llama.cpp/build/bin/llama-quantize --allow-requantize --leave-output-tensor --token-embedding-type iq3_s \
  $HOME/ai/models/Qwen3.5-9B-Claude-Distill-v2-Q8_0.gguf $W/embIQ3S.tmp IQ3_M 6 && mv $W/embIQ3S.tmp $W/Qwen3.5-9B-IQ3_M-embIQ3S.gguf"

echo "== [4] mistral.rs build, $(left)s left ($(el))"
BT=$(( $(left) - 120 )); [ $BT -gt 300 ] || { echo "too little time for the build"; exit 0; }
L=$W/build-mistral.log
timeout $((BT + 30)) systemd-run --user --unit=iq3s-build --collect --wait -q -p MemoryMax=12G -p MemoryHigh=8G -p MemorySwapMax=0 -p RuntimeMaxSec=$BT \
  -p WorkingDirectory=$E/mr-iq3s --setenv=CARGO_TARGET_DIR=$E/target-iq3s-oxide --setenv=TITAN_OXIDE_DIR=$E/oxide-kernels \
  --setenv=CUDA_HOME=$E/nocuda-bin --setenv=CUDA_PATH=$E/nocuda-bin --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDARC_CUDA_VERSION=13030 \
  --setenv=CARGO_BUILD_JOBS=2 --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 "--setenv=RUSTFLAGS=-L $E/lib" \
  --setenv=PATH=$HOME/.cargo/bin:/usr/bin:/bin -p StandardOutput=append:$L -p StandardError=append:$L \
  nice -n 10 cargo build --release -p mistralrs-cli --features oxide
rc=$?; echo "mistral.rs build rc=$rc ($(el))"
grep -E "^error" -A10 $L | head -60; grep -E "titan:|Finished" $L | sort -u | tail -5
ls -la $E/target-iq3s-oxide/release/mistralrs 2>&1
free -m | head -2
echo "-- quant B: $(systemctl --user is-active iq3s-quantb) $(tail -c 300 $W/quantb.log | tr '\n' ' ')"
for i in $(seq 1 60); do systemctl --user -q is-active iq3s-quantb || break; [ $(left) -gt 30 ] || break; sleep 10; done
tail -2 $W/quantb.log
echo "== window1 end $(el)"
