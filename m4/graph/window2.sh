trap 'systemctl --user start titan-mistral' EXIT
systemctl --user stop titan-mistral
# CUDA graphs window 2: (1) clean decode measurements of the service binary without nsys, with and without CPU
# misses (TITAN_TIERED_DEBUG_SKIP_MISSES=1, timing only) to bound launch overhead; (2) cuda build of mr-graph;
# (3) TITAN_CUDA_GRAPHS=0/1 A/B, MTP off and 2; (4) nvcc-free build if time allows.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/graph/window2.sh
trap 'exit 130' INT TERM HUP
set -u
E=$HOME/titan-engine; G=$E/m4/graph
exec > >(tee -a $G/window2-$(date +%Y%m%d-%H%M).log) 2>&1
T0=$(date +%s); DEADLINE=$((T0 + 56 * 60))
el() { echo "$(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
echo "== window2 start $(date -Is), titan-mistral: $(systemctl --user is-active titan-mistral)"
for p in $(ps -eo pid=,ppid=,comm= | awk '$2 == 878468 && $3 ~ /rust-analyzer/ {print $1}'); do echo "killing rust-analyzer $p"; kill $p; done
free -m | head -2
P=$E/m4/prompts-eval.txt
run() { # run NAME BIN ENV...
  local n=$1 bin=$2; shift 2
  [ $(left) -gt 240 ] || { echo "skip $n: $(left)s left"; return 1; }
  timeout 420 systemd-run --user --unit=graph-$n --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=400 \
    bash $G/bench.sh $n $bin $P 8 64 "$@"
  echo "-- $n rc=$? ($(el))"
  grep -o "titan tiered timing.*\|titan graphs.*\|titan mtp stats.*" $G/out/$n.server.log | tail -3
}
OLD=$E/target-oss-oxide/release/mistralrs
echo "== (1) measurements, $OLD ($(el))"
run m-off $OLD TITAN_TIERED_TIMING=1
run m-off-skip $OLD TITAN_TIERED_DEBUG_SKIP_MISSES=1
run m-mtp2 $OLD TITAN_MTP=2 TITAN_TIERED_TIMING=1
run m-mtp2-skip $OLD TITAN_MTP=2 TITAN_TIERED_DEBUG_SKIP_MISSES=1

echo "== (2) cuda build of mr-graph ($(el))"
L=$G/build-cuda.log; t=$(date +%s)
timeout 1500 systemd-run --user --unit=graph-build-cuda --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 -p RuntimeMaxSec=1450 \
  -p WorkingDirectory=$E/mr-graph --setenv=CARGO_BUILD_JOBS=2 --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 --setenv=CUDA_COMPUTE_CAP=120 \
  --setenv=CUDA_HOME=/usr/local/cuda --setenv=NVCC_CCBIN=/usr/bin/g++-15 --setenv=CUDA_NVCC_FLAGS=-Wno-template-body \
  "--setenv=RUSTFLAGS=-L $E/lib" --setenv=CARGO_TARGET_DIR=$E/target-graph --setenv=PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:/usr/bin:/bin \
  -p StandardOutput=truncate:$L -p StandardError=truncate:$L nice -n 10 cargo build --release -p mistralrs-cli --features cuda
rc=$?; systemctl --user stop graph-build-cuda 2>/dev/null; grep -q "^error" $L && rc=1
echo "cuda build rc=$rc in $(( $(date +%s) - t ))s ($(el))"; grep -E "^(error|warning: unused)|Finished" -A7 $L | head -60
NEW=$E/target-graph/release/mistralrs
if [ $rc = 0 ]; then
  echo "== (3) graphs A/B ($(el))"
  run g0-off $NEW TITAN_CUDA_GRAPHS=0
  run g1-off $NEW TITAN_CUDA_GRAPHS=1
  run g0-mtp2 $NEW TITAN_CUDA_GRAPHS=0 TITAN_MTP=2
  run g1-mtp2 $NEW TITAN_CUDA_GRAPHS=1 TITAN_MTP=2
  cd $G/out; for p in "m-off g0-off" "g0-off g1-off" "g0-mtp2 g1-mtp2" "g0-off g0-mtp2"; do set -- $p; [ -f $1.json ] && [ -f $2.json ] &&
    python3 -c "import json,sys;a,b=json.load(open(sys.argv[1]+'.json')),json.load(open(sys.argv[2]+'.json'));print('IDENT',sys.argv[1],sys.argv[2],sum(x==y for x,y in zip(a,b)),'/',len(a))" $1 $2; done
  grep -h "titan graphs" g1-off.server.log g1-mtp2.server.log | grep -v INFO | head; grep -hc "capture of" g1-off.server.log g1-mtp2.server.log
fi
if [ $(left) -gt 700 ]; then
  echo "== (4) nvcc-free build ($(el))"
  L=$G/build-oxide.log; t=$(date +%s)
  timeout $(( $(left) - 60 )) systemd-run --user --unit=graph-build-oxide --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 -p RuntimeMaxSec=$(( $(left) - 90 )) \
    -p WorkingDirectory=$E/mr-graph --setenv=CARGO_TARGET_DIR=$E/target-graph-oxide --setenv=TITAN_OXIDE_DIR=$E/oxide-kernels --setenv=CUDA_HOME=$E/nocuda-bin \
    --setenv=CUDA_PATH=$E/nocuda-bin --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDARC_CUDA_VERSION=13030 --setenv=CARGO_BUILD_JOBS=2 \
    --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 "--setenv=RUSTFLAGS=-L $E/lib" --setenv=PATH=$HOME/.cargo/bin:/usr/bin:/bin \
    -p StandardOutput=truncate:$L -p StandardError=truncate:$L nice -n 10 cargo build --release -p mistralrs-cli --features oxide
  rc=$?; systemctl --user stop graph-build-oxide 2>/dev/null; grep -q "^error" $L && rc=1
  echo "oxide build rc=$rc in $(( $(date +%s) - t ))s ($(el))"; grep "titan:" $L | sort -u | head -5; grep -E "^error|Finished" -A7 $L | head -30
  ls -la $E/target-graph-oxide/release/mistralrs
fi
echo "== window2 end $(el)"
