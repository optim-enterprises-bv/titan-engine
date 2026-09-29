# shared by the step scripts: paths, the nvcc-free build, a model server around a client command
set -u
E=$HOME/titan-engine; P=$E/m4/pc; M=$HOME/ai/models; F=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
BIN=${BIN:-$E/target-pc-oxide/release/mistralrs}
cd $P
left() { echo $(( ${DEADLINE:-$(( $(date +%s) + 3600 ))} - $(date +%s) )); }
build() {
  local L=$P/build.log t=$(date +%s)
  timeout 1230 systemd-run --user --unit=pc-build --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 -p RuntimeMaxSec=1200 \
    -p WorkingDirectory=$E/mr-pc --setenv=CARGO_TARGET_DIR=$E/target-pc-oxide --setenv=TITAN_OXIDE_DIR=$E/oxide-kernels --setenv=CUDA_HOME=$E/nocuda-bin \
    --setenv=CUDA_PATH=$E/nocuda-bin --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDARC_CUDA_VERSION=13030 --setenv=CARGO_BUILD_JOBS=2 \
    --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 "--setenv=RUSTFLAGS=-L $E/lib" --setenv=PATH=$HOME/.cargo/bin:/usr/bin:/bin \
    -p StandardOutput=truncate:$L -p StandardError=truncate:$L nice -n 10 cargo build --release -p mistralrs-cli --features oxide
  local rc=$?; systemctl --user stop pc-build 2>/dev/null; grep -q "^error" $L && rc=1
  echo "oxide build rc=$rc in $(( $(date +%s) - t ))s"
  grep -E "^(error|warning)" -A12 $L | grep -v "^--$" | head -150; grep "Finished" $L
  ls -la $BIN
  return $rc
}
# srv NAME PORT "ENV=.. ENV=.." "client command" [max-seq-len] [model dir]
srv() {
  local n=$1 port=$2 envs=$3 cmd=$4 msl=${5:-65536} dir=${6:-$M/Qwen3.6-35B-A3B-MTP}
  local budget=$(( $(left) - 30 )); [ $budget -gt 1500 ] && budget=1500
  [ $budget -gt 120 ] || { echo "skip $n: $(left)s left"; return 1; }
  echo "-- $n ($envs) start $(date +%T), budget ${budget}s"
  timeout $((budget + 15)) systemd-run --user --unit=pc-$n --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=$budget bash -c "
    cd $P; env LD_LIBRARY_PATH=$E/lib TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt $envs \
      $BIN serve -p $port --no-ui --paged-attn off --max-seq-len $msl --format gguf -m $dir -f $F > out/$n.server.log 2>&1 &
    pid=\$!; t0=\$(date +%s)
    for i in \$(seq 1 200); do curl -sf -m 2 -o /dev/null localhost:$port/v1/models && break; kill -0 \$pid 2>/dev/null || { echo server died; tail -30 out/$n.server.log; exit 1; }; sleep 2; done
    echo \"   loaded in \$(( \$(date +%s) - t0 ))s\"
    $cmd
    kill -TERM \$pid; for i in 1 2 3 4 5 6 7 8; do kill -0 \$pid 2>/dev/null || break; sleep 1; done; kill -KILL \$pid 2>/dev/null; true"
  echo "-- $n end $(date +%T)"; grep -E "panicked|ILLEGAL|out of memory|OOM|Error" out/$n.server.log | head -5
  grep -c "Hybrid prefix cache hit" out/$n.server.log | sed 's/^/   prefix hits: /'
  grep -o "Prefix cache hitrate [0-9.]*%" out/$n.server.log | sort -u | tail -3
}
