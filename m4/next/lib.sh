# Sourced by queued jobs. Every step has a timeout; servers run as transient units capped at MemoryMax=20G.
set -u
E=$HOME/titan-engine; N=$E/m4/next; M=$HOME/ai/models
F80=$M/qwen3-next-80b/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf
BIN=${BIN:-$E/target-next/release/mistralrs}
cd $N
left() { echo $(( DEADLINE - $(date +%s) )); }
# cap a step's timeout at the time left in the window
cap() { local t=$1 l=$(( DEADLINE - $(date +%s) - 30 )); echo $(( t < l ? t : l )); }
cold80() { python3 -c "import os;fd=os.open('$F80',os.O_RDONLY);os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)"; }

build() { # build WORKTREE
  local L=$N/logs/build-p1.log t=$(date +%s)
  timeout $(cap 1530) systemd-run --user --unit=p1-build --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 \
    -p RuntimeMaxSec=1500 -p WorkingDirectory=$1 \
    --setenv=CARGO_BUILD_JOBS=2 --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 --setenv=CUDA_COMPUTE_CAP=120 \
    --setenv=CUDA_HOME=/usr/local/cuda --setenv=NVCC_CCBIN=/usr/bin/g++-15 --setenv=CUDA_NVCC_FLAGS=-Wno-template-body \
    "--setenv=RUSTFLAGS=-L $E/lib" --setenv=CARGO_TARGET_DIR=$E/target-next \
    --setenv=PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:/usr/bin:/bin \
    -p StandardOutput=truncate:$L -p StandardError=truncate:$L nice -n 10 cargo build --release -p mistralrs-cli --features cuda
  local rc=$?; systemctl --user stop p1-build 2>/dev/null; grep -q "^error" $L && rc=1
  echo "build rc=$rc in $(( $(date +%s) - t ))s"; grep -E "^(warning: unused|error)|Finished" -A9 $L | grep -v candle | head -80
  ls -la $BIN; return $rc
}

# run40 NAME PROMPTFILE MAXTOK GGUF ENV...: chat completions through collect.sh (copy of m3/collect.sh), one server
run40() {
  local name=$1 pf=$2 mt=$3 g=$4; shift 4
  timeout $(cap 1800) systemd-run --user --unit=p1-$name --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 \
    --setenv=BIN=$BIN --setenv=DIR=$(dirname $g) --setenv=FILE=$(basename $g) \
    bash $N/collect.sh $name $pf $mt "$@" 2>&1 | grep -v " INFO "
  summary $name out/$name.server.log
}
summary() { # NAME LOG
  grep -o 'titan tiered auto.*' $2 | head -1
  grep -m1 -o 'titan tiered experts: .*' $2
  grep -o 'decode hit rate.*' $2 | tail -1
  grep -o 'titan tiered timing.*' $2 | tail -1
  grep -o 'titan prof: .*' $2 | tail -1
}
same() { python3 $N/cmp40.py "$@"; }
