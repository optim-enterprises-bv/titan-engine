# m4/swap shared helpers. Source from a window script (after setting T0/DEADLINE).
set -u
E=$HOME/titan-engine; S=$E/m4/swap; SRC=$E/mr-swap; M=$HOME/ai/models
PORT=18590
MBIN=$E/target-swap-oxide/release/mistralrs
BIN=${BIN:-$S/mistralrs-swap-w}
cd $S
el() { echo "$(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
killra() { for p in $(ps -eo pid=,ppid=,comm= | awk '$2 == 878468 && $3 ~ /rust-analyzer/ {print $1}'); do echo "killing rust-analyzer $p"; kill $p; done; }
retry_wait() { # NAME SECONDS: wait for $S/retry-NAME (the agent fixes the tree meanwhile)
  echo "!! $1 failed: waiting up to $2 s for $S/retry-$1 ($(el))"
  for i in $(seq 1 $(( $2 / 5 ))); do [ -f $S/retry-$1 ] && { rm -f $S/retry-$1; echo "retrying $1 ($(el))"; return 0; }; [ -f $S/abort ] && { rm -f $S/abort; return 1; }; sleep 5; done
  return 1
}
mbuild() {
  local L=$S/build-oxide.log t=$(date +%s)
  timeout 900 systemd-run --user --unit=swap-build --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 -p RuntimeMaxSec=880 \
    -p WorkingDirectory=$SRC --setenv=CARGO_TARGET_DIR=$E/target-swap-oxide --setenv=TITAN_OXIDE_DIR=$E/oxide-kernels --setenv=CUDA_HOME=$E/nocuda-bin \
    --setenv=CUDA_PATH=$E/nocuda-bin --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDARC_CUDA_VERSION=13030 --setenv=CARGO_BUILD_JOBS=2 \
    --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 "--setenv=RUSTFLAGS=-L $E/lib" --setenv=PATH=$HOME/.cargo/bin:/usr/bin:/bin \
    -p StandardOutput=truncate:$L -p StandardError=truncate:$L nice -n 10 cargo build --release -p mistralrs-cli --features oxide
  local rc=$?; systemctl --user stop swap-build 2>/dev/null; grep -q "^error" $L && rc=1
  echo "oxide build rc=$rc in $(( $(date +%s) - t ))s ($(el))"; grep -E "^error" -A14 $L | head -120; grep "Finished" $L
  [ $rc = 0 ] && [ $MBIN -nt $SRC/mistralrs-core/src/titan_swap.rs ]
}
# srv_start NAME CONFIG: the swap server under MemoryMax=20G, in the background; waits for /v1/models
srv_start() {
  local n=$1 cfg=$2
  SRV=$n; SLOG=$S/out/$n.server.log
  systemctl --user stop swap-srv 2>/dev/null; systemctl --user reset-failed swap-srv 2>/dev/null
  systemd-run --user --unit=swap-srv --collect -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=$(( $(left) + 60 )) \
    -p KillSignal=SIGTERM -p TimeoutStopSec=15 -p WorkingDirectory=$S --setenv=LD_LIBRARY_PATH=$E/lib \
    -p StandardOutput=truncate:$SLOG -p StandardError=truncate:$SLOG $BIN from-config --file $cfg
  local t0=$(date +%s)
  for i in $(seq 1 240); do
    curl -sf -m 2 -o /dev/null localhost:$PORT/v1/models && { echo "-- $n up in $(( $(date +%s) - t0 ))s ($(el))"; return 0; }
    systemctl --user -q is-active swap-srv || { echo "!! $n died"; grep -v " INFO " $SLOG | tail -30; return 1; }
    sleep 1
  done
  echo "!! $n not up"; return 1
}
srv_stop() {
  systemctl --user stop swap-srv 2>/dev/null
  echo "-- $SRV stopped ($(el))"
  grep -E "panicked|ILLEGAL|OUT_OF_MEMORY|out of memory|CUDA_ERROR|ERROR" $SLOG | head -8 | cut -c1-300
}
swaplog() { # the swap / memory lines of the server log
  grep -oE "titan swap.*|titan tiered auto:.*|titan settings for the next model.*|titan doorbell: released.*|titan: CUDA out of memory.*|more references.*" $SLOG | cut -c1-260
}
smi() { nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>&1 | head -1; }
