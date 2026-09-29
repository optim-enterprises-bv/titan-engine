# Sourced by queued jobs. Every step has a timeout; servers run as transient units capped at MemoryMax=20G.
set -u
E=$HOME/titan-engine; S=$E/m4/skip; N=$E/m4/next; M=$HOME/ai/models
F80=$M/qwen3-next-80b/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf
BIN=${BIN:-$E/target-skip-oxide/release/mistralrs}
cd $S
left() { echo $(( DEADLINE - $(date +%s) )); }
cap() { local t=$1 l=$(( DEADLINE - $(date +%s) - 30 )); echo $(( t < l ? t : l )); }
cold80() { python3 -c "import os;fd=os.open('$F80',os.O_RDONLY);os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)"; }
# the P1 final config, GPU fraction pinned so every run gets the same 143 slots
FIN="TITAN_TIERED=1 TITAN_TIERED_MMAP=1 TITAN_TIERED_GPU_FRACTION=0.28 TITAN_TIERED_TIMING=1 TITAN_TIERED_LOOKAHEAD=2 TITAN_TIERED_LOOKAHEAD_POPULATE=1 TITAN_TIERED_PAGEOUT=1"
PROF="TITAN_TIERED_PROFILE=$N/profile-next.txt"

build() {
  local L=$S/logs/build-skip-oxide.log t=$(date +%s)
  command -v nvcc >/dev/null && echo "note: nvcc on this shell's PATH (the build's PATH has none)"
  timeout $(cap 1230) systemd-run --user --unit=sk-build --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 -p RuntimeMaxSec=1200 \
    -p WorkingDirectory=$E/mr-skip --setenv=CARGO_TARGET_DIR=$E/target-skip-oxide --setenv=TITAN_OXIDE_DIR=$E/oxide-kernels --setenv=CUDA_HOME=$E/nocuda-bin \
    --setenv=CUDA_PATH=$E/nocuda-bin --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDARC_CUDA_VERSION=13030 --setenv=CARGO_BUILD_JOBS=2 \
    --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 "--setenv=RUSTFLAGS=-L $E/lib" --setenv=PATH=$HOME/.cargo/bin:/usr/bin:/bin \
    -p StandardOutput=truncate:$L -p StandardError=truncate:$L nice -n 10 cargo build --release -p mistralrs-cli --features oxide
  local rc=$?; systemctl --user stop sk-build 2>/dev/null; grep -q "^error" $L && rc=1
  echo "build rc=$rc in $(( $(date +%s) - t ))s"; grep -E "^(warning: unused|error)|Finished" -A9 $L | grep -v candle | head -80
  ls -la $BIN; [ $rc = 0 ] && [ $BIN -nt $E/mr-skip/mistralrs-core/src/models/titan_miss_skip.rs ] || return 1
}

# run NAME ENV...: cold page cache, one server: 40 x 64 speed + quality
run() {
  local name=$1; shift
  [ $(left) -gt 330 ] || { echo "skip $name: $(left)s left"; return 1; }
  cold80
  timeout $(cap 900) systemd-run --user --unit=sk-$name --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=880 \
    --setenv=BIN=$BIN --setenv=F80=$F80 --setenv=QUALITY=${QUALITY:-1} bash $S/collect.sh $name "$@" 2>&1 | grep -v " INFO "
}

# hard NAME BUDGET ENV...: cold page cache, one server, hardgen.py for at most BUDGET seconds
hard() {
  local name=$1 b=$2; shift 2
  [ $(left) -gt 240 ] || { echo "skip $name: $(left)s left"; return 1; }
  local l=$(( $(left) - 150 )); b=$(( b < l ? b : l ))
  cold80
  timeout $(cap $(( b + 300 ))) systemd-run --user --unit=sk-$name --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 \
    --setenv=BIN=$BIN --setenv=F80=$F80 bash $S/hard.sh $name $b "$@" 2>&1 | grep -v " INFO "
}
