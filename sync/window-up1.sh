trap 'systemctl --user start titan-mistral' EXIT
systemctl --user stop titan-mistral
# upstream mistral.rs v0.9.4 (2370966) bench, window 1: nvcc build, then benches in priority order.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/sync/window-up1.sh
trap 'exit 130' INT TERM HUP
set -u
E=$HOME/titan-engine; S=$E/sync; M=$HOME/ai/models; F=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
exec > >(tee -a $S/window-up1-$(date +%Y%m%d-%H%M).log) 2>&1
T0=$(date +%s); DEADLINE=$((T0 + 57 * 60))
el() { echo "$(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
echo "== window-up1 start $(date -Is), titan-mistral: $(systemctl --user is-active titan-mistral)"
for p in $(ps -eo pid=,ppid=,comm= | awk '$2 == 878468 && $3 ~ /rust-analyzer/ {print $1}'); do echo "killing rust-analyzer $p"; kill $p; done
free -m | head -2
BIN=$E/target-upstream/release/mistralrs
echo "== nvcc build of mr-upstream ($(el))"
L=$S/build-up.log; t=$(date +%s)
timeout 2830 systemd-run --user --unit=up-build --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 -p RuntimeMaxSec=2800 \
  -p WorkingDirectory=$E/mr-upstream --setenv=CARGO_TARGET_DIR=$E/target-upstream --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDA_HOME=/usr/local/cuda \
  --setenv=NVCC_CCBIN=/usr/bin/g++-15 --setenv=CUDA_NVCC_FLAGS=-Wno-template-body --setenv=CARGO_BUILD_JOBS=2 \
  --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 "--setenv=RUSTFLAGS=-L $E/lib" --setenv=PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:/usr/bin:/bin \
  -p StandardOutput=truncate:$L -p StandardError=truncate:$L nice -n 10 cargo build --release -p mistralrs-cli --features cuda
rc=$?; systemctl --user stop up-build 2>/dev/null; grep -q "^error" $L && rc=1
echo "build rc=$rc in $(( $(date +%s) - t ))s ($(el))"; grep -E "^error|Finished" -A12 $L | head -60
ls -la $BIN; [ $rc = 0 ] && [ -x $BIN ] && [ $BIN -nt $E/mr-upstream/Cargo.toml ] || { echo "no binary: ending the window"; tail -30 $L; exit 1; }
$BIN --version

P=$E/m4/prompts-eval.txt
cmp() { python3 -c "import json,sys;a,b=json.load(open(sys.argv[1])),json.load(open(sys.argv[2]));print('IDENT',sys.argv[2].split('/')[-1],'vs',sys.argv[1].split('/')[-1],':',sum(x==y for x,y in zip(a,b)),'/',len(a),'len',len(b))" $1 $2; }
run() { # run NAME NEED_SECS [ENV=..] -- env SFLAGS/NPROMPTS/BIG passed through
  local n=$1 need=$2; shift 2
  [ $(left) -gt $need ] || { echo "skip $n: $(left)s left, need $need"; return 1; }
  local to=$(( $(left) - 30 )); [ $to -gt 1500 ] && to=1500
  echo "== $n ($(el)), timeout ${to}s"
  timeout $to systemd-run --user --unit=up-$n --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=$((to-10)) \
    --setenv=BIN=$BIN --setenv=DIR=$M/Qwen3.6-35B-A3B-MTP --setenv=FILE=$F "--setenv=SFLAGS=${SFLAGS:---max-seq-len 16384 --pa-context-len 16384}" \
    --setenv=NPROMPTS=${NPROMPTS:-40} --setenv=BIG=${BIG:-0} \
    bash $S/collect-up.sh $n $P 256 "$@" 2>&1 | grep -v " INFO "
  systemctl --user stop up-$n 2>/dev/null
  [ -f $S/out/$n.json ] && cmp $E/m3/out/q35-prof.json $S/out/$n.json
  echo "-- $n done ($(el))"
}
# A: defaults (auto device map, paged attn auto=on, CUDA graphs default=on), MTP off, 40x256 + 4k/13k prefill + repeats
BIG=1 run upA 600
# B: upstream MTP from the GGUF (expected: nextn blk.40 is not bound to mtp.*), 10 prompts
NPROMPTS=10 SFLAGS="--max-seq-len 16384 --pa-context-len 16384 --mtp --mtp-n-predict 2" run upB-mtp 400
# C: graphs off
run upC-nograph 600 MISTRALRS_CUDA_GRAPHS=0
# D: collect.sh's own flags (paged attn off => no graphs), unmodified collect.sh
if [ $(left) -gt 600 ]; then
  echo "== upD collect.sh as-is ($(el))"
  timeout $(( $(left) - 30 )) systemd-run --user --unit=up-upD --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 \
    --setenv=BIN=$BIN --setenv=DIR=$M/Qwen3.6-35B-A3B-MTP --setenv=FILE=$F bash $E/m3/collect.sh up-collect $P 256 2>&1 | grep -v " INFO "
  systemctl --user stop up-upD 2>/dev/null; cmp $E/m3/out/q35-prof.json $E/m3/out/up-collect.json
else echo "skip upD: $(left)s left"; fi
free -m | head -2
echo "== window-up1 end $(el)"
