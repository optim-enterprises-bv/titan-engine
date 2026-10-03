#!/bin/bash
# IQ3_S window 3 (control): (resume the mistral.rs build if needed), end-to-end vs llama.cpp on the requantized
# Qwen3.5-9B IQ3_M files, then the 35B byte-identity gates (whole card: titan-spark stopped for the window).
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/iq3s/window2.sh [stages]   (stages: build e2e reg; default all)
set -u
E=$HOME/titan-engine; W=$E/m4/iq3s; M=$HOME/ai/models
STAGES=${1:-e2e}
exec > >(tee -a $W/window3-$(date +%Y%m%d-%H%M).log) 2>&1
T0=$(date +%s); DEADLINE=$((T0 + 57 * 60))
el() { echo "$(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
SPARK=$(systemctl --user is-active titan-spark)
trap 'systemctl --user stop iq3s-build iq3s-llama iq3s-mistral 2>/dev/null; for u in $(systemctl --user list-units --plain --no-legend "iq3s-reg-*" | cut -d" " -f1); do systemctl --user stop $u; done; [ "$SPARK" = active ] && systemctl --user start titan-spark; echo "== window3 exit $(date -Is): titan-spark $(systemctl --user is-active titan-spark), titan-mistral $(systemctl --user is-active titan-mistral)"' EXIT
trap 'exit 130' INT TERM HUP
echo "== lock acquired $(date -Is); stages: $STAGES; titan-spark $SPARK, titan-mistral $(systemctl --user is-active titan-mistral) (never started)"
[ "$SPARK" = active ] && systemctl --user stop titan-spark && echo "stopped titan-spark for the window"
for p in $(ps -eo pid=,ppid=,comm= | awk '$2 == 878468 && $3 ~ /rust-analyzer/ {print $1}'); do echo "killing rust-analyzer $p"; kill $p; done
sleep 3; free -m | head -2; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
BIN=$E/target-iq3s-oxide/release/mistralrs
export LD_LIBRARY_PATH=$E/lib:/usr/local/cuda/lib64

if [[ " $STAGES " == *" build "* ]]; then
  echo "== [build] mistral.rs ($(el))"
  BT=$(( $(left) - 25 * 60 )); [ $BT -gt 600 ] || BT=600
  L=$W/build-mistral.log
  timeout $((BT + 30)) systemd-run --user --unit=iq3s-build --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 -p RuntimeMaxSec=$BT \
    -p WorkingDirectory=$E/mr-iq3s --setenv=CARGO_TARGET_DIR=$E/target-iq3s-oxide --setenv=TITAN_OXIDE_DIR=$E/oxide-kernels \
    --setenv=CUDA_HOME=$E/nocuda-bin --setenv=CUDA_PATH=$E/nocuda-bin --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDARC_CUDA_VERSION=13030 \
    --setenv=CARGO_BUILD_JOBS=2 --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 "--setenv=RUSTFLAGS=-L $E/lib" \
    --setenv=PATH=$HOME/.cargo/bin:/usr/bin:/bin -p StandardOutput=append:$L -p StandardError=append:$L \
    nice -n 10 cargo build --release -p mistralrs-cli --features oxide
  echo "build rc=$? ($(el))"; grep -E "^error" -A10 $L | head -40; tail -2 $L
fi
ls -la $BIN || { echo "no binary: ending window"; exit 1; }
sha256sum $BIN | cut -c1-16

sw() { for i in $(seq 1 300); do curl -sf -m 2 -o /dev/null localhost:$1$2 && return 0; systemctl --user -q is-active $3 || return 1; sleep 2; done; return 1; }
e2e() { # e2e TAG FILE
  local T=$1 F=$2
  [ -f $F ] || { echo "skip $T: no $F"; return 1; }
  [ $(left) -gt 600 ] || { echo "skip $T: $(left)s left"; return 1; }
  echo "-- e2e $T: llama.cpp ($(el))"
  systemd-run --user --unit=iq3s-llama --collect -q -p MemoryMax=16G -p MemorySwapMax=0 -p StandardOutput=truncate:$W/out/${T}_llama.log -p StandardError=truncate:$W/out/${T}_llama.log \
    $HOME/ai/llama.cpp/build/bin/llama-server -m $F -ngl 99 -c 4096 --port 18491 --temp 0 --top-k 1 -np 1
  if sw 18491 /health iq3s-llama; then (cd $W && timeout 900 python3 e2e.py llama 18491 $T); else echo "llama died"; tail -5 $W/out/${T}_llama.log; fi
  systemctl --user stop iq3s-llama; sleep 2
  echo "-- e2e $T: mistral.rs ($(el))"
  systemd-run --user --unit=iq3s-mistral --collect -q -p MemoryMax=20G -p MemorySwapMax=0 --setenv=LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
    -p StandardOutput=truncate:$W/out/${T}_mistral.log -p StandardError=truncate:$W/out/${T}_mistral.log \
    $BIN --seed 0 serve -p 18492 --no-ui --paged-attn off --max-seq-len 4096 --format gguf -m $(dirname $F) -f $(basename $F)
  if sw 18492 /v1/models iq3s-mistral; then
    (cd $W && timeout 1500 python3 e2e.py mistral 18492 $T)
    grep -E "titan|GPU|device map|layers" $W/out/${T}_mistral.log | grep -v " DEBUG " | head -8 | cut -c1-200
  else echo "mistral died"; grep -vE " INFO " $W/out/${T}_mistral.log | tail -20; fi
  systemctl --user stop iq3s-mistral; sleep 2
  (cd $W && python3 e2e.py cmp $T $F)
}
if [[ " $STAGES " == *" e2e "* ]]; then
  echo "== [e2e] ($(el))"
  # control: the same harness on the source Q8_0 model (no IQ3_S tensors) calibrates how much
  # llama.cpp vs mistral.rs disagreement is not caused by the IQ3_S path
  e2e q8ctl $M/Qwen3.5-9B-Claude-Distill-v2-Q8_0.gguf
fi

coll() { # coll NAME BIN DIR ENV...
  local n=$1 b=$2 d=$3; shift 3
  [ $(left) -gt 560 ] || { echo "skip $n: $(left)s left"; return 1; }
  timeout 700 systemd-run --user --unit=iq3s-reg-$n --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=680 \
    --setenv=BIN=$b --setenv=DIR=$d --setenv=FILE=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    bash $E/sync/w094/collect094.sh $n $E/m3/prompts-eval.txt 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt TITAN_PFS=1 \
      TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64 TITAN_TIERED_RESERVE_MIB=1536 "$@" 2>&1 | grep -v " INFO "
  grep -E "titan tiered auto|panicked|CUDA_ERROR" $E/m3/out/$n.server.log | sed 's/.*INFO //' | cut -c1-200 | head -3
  echo "-- $n done ($(el))"
}
cmp() { python3 -c "
import json,sys
a,b=json.load(open(sys.argv[1])),json.load(open(sys.argv[2]))
print('GATE',sys.argv[2].split('/')[-1],'vs',sys.argv[1].split('/')[-1],':',sum(x==y for x,y in zip(a,b)),'/',len(a),'len',len(b))" $1 $2; }
if [[ " $STAGES " == *" reg "* ]]; then
  echo "== [reg] 35B identity gates ($(el))"
  D=$M/Qwen3.6-35B-A3B-MTP; DEP=$E/bin/mistralrs-titan-swap
  coll iq3s-m2 $BIN $D TITAN_MTP=2 && cmp $E/m6/out/h-off.json $E/m3/out/iq3s-m2.json
  # The MTP-off reference (q35-prof) was made from ~/ai/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf, which no longer
  # exists (sync/w094/q35root dangles). Gate MTP off as an A/B against the deployed binary on the MTP file.
  coll iq3s-m0 $BIN $D && cmp $E/m3/out/q35-prof.json $E/m3/out/iq3s-m0.json
  coll dep-m0 $DEP $D && cmp $E/m3/out/dep-m0.json $E/m3/out/iq3s-m0.json && cmp $E/m3/out/q35-prof.json $E/m3/out/dep-m0.json
fi
free -m | head -2
echo "== window3 end $(el)"
