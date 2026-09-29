trap 'systemctl --user start titan-mistral' EXIT
systemctl --user stop titan-mistral
# fmtmmq-window.sh: the one build + GPU window for the fmt-mmq wiring (llama.cpp MMQ prefill for IQ4_NL,
# MXFP4, NVFP4 in mr-mmq). Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/fmtmmq-window.sh
# (1) nvcc build of mr-mmq, (2) dense Qwen3-14B-<fmt> end to end: llama.cpp / old binary / new build
# (+ new build with TITAN_LLAMA_MMQ=0), greedy text and ~2k-token prefill, (3) the nvcc-free
# `--features oxide` build. Every step is capped in memory and time; the service comes back on any exit.
trap 'exit 130' INT TERM HUP
set -u
E=$HOME/titan-engine; D=$E/m4/fmtmmq
OLD=${OLD:-$E/target-oxide/release/mistralrs}; NEW=$E/target-mmq/release/mistralrs
LOG=$E/m4/fmtmmq-window-$(date +%Y%m%d-%H%M).log
exec > >(tee -a $LOG) 2>&1
T0=$(date +%s); DEADLINE=$((T0 + 57 * 60))
el() { echo "$(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
echo "== window start $(date -Is), titan-mistral: $(systemctl --user is-active titan-mistral)"

# cargo under a transient unit: MemoryMax 12G, no swap, RuntimeMaxSec so a timeout also kills the build
build() { # build <unit> <log> <max-seconds> <workdir> <cargo args...>, env from BENV
  local unit=$1 log=$2 max=$3 wd=$4; shift 4
  timeout $((max + 30)) systemd-run --user --unit=$unit --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 \
    -p RuntimeMaxSec=$max -p WorkingDirectory=$wd "${BENV[@]}" \
    -p StandardOutput=truncate:$log -p StandardError=truncate:$log nice -n 10 cargo build --release "$@"
  local rc=$?
  systemctl --user stop $unit 2>/dev/null
  grep -q "^error" $log && rc=1
  return $rc
}
COMMON=(--setenv=CARGO_BUILD_JOBS=2 --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 --setenv=CUDA_COMPUTE_CAP=120
  "--setenv=RUSTFLAGS=-L $E/lib")

# (1) nvcc build
echo "== (1) nvcc build of mr-mmq ($(el))"
BENV=("${COMMON[@]}" --setenv=CUDA_HOME=/usr/local/cuda --setenv=NVCC_CCBIN=/usr/bin/g++-15 --setenv=CUDA_NVCC_FLAGS=-Wno-template-body
  --setenv=CARGO_TARGET_DIR=$E/target-mmq --setenv=PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:/usr/bin:/bin)
t=$(date +%s)
build fmtmmq-build-cuda $E/build-mmq.log $(( $(left) < 1500 ? $(left) : 1500 )) $E/mr-mmq -p mistralrs-cli --features cuda
rc1=$?
echo "build (1) rc=$rc1 in $(( $(date +%s) - t ))s"; grep -E "Finished|^error" -A6 $E/build-mmq.log | head -40
[ $rc1 = 0 ] && [ $NEW -nt $E/mr-mmq/mistralrs-quant/src/gguf/fast_mmq.rs ] || { echo "no new binary: ending the window"; exit 1; }
ls -la $NEW

# (2) end to end
echo "== (2) e2e ($(el))"
for fmt in iq4_nl mxfp4 nvfp4; do
  [ $(left) -gt 900 ] || { echo "skipping $fmt: $(left)s left"; continue; }
  timeout 420 $D/e2e.sh llama_$fmt llama $fmt
  timeout 420 $D/e2e.sh old_$fmt $OLD $fmt
  timeout 420 $D/e2e.sh new_$fmt $NEW $fmt
  [ $(left) -gt 1200 ] && timeout 420 $D/e2e.sh deq_$fmt $NEW $fmt TITAN_LLAMA_MMQ=0
  (cd $D && for p in "llama_$fmt old_$fmt" "llama_$fmt new_$fmt" "old_$fmt new_$fmt" "new_$fmt deq_$fmt"; do
     set -- $p; [ -f out/$1.json ] && [ -f out/$2.json ] && python3 cmp_strip.py $1 $2 | tail -1; done)
  echo "-- $fmt done ($(el))"
done

# (3) nvcc-free build (nonvcc-build.sh: CUDA_HOME is a stub without bin/, no nvcc on PATH)
echo "== (3) nvcc-free --features oxide build ($(el))"
if [ $(left) -gt 300 ]; then
  BENV=("${COMMON[@]}" --setenv=CARGO_TARGET_DIR=$E/target-mmq-oxide --setenv=TITAN_OXIDE_DIR=$E/oxide-kernels
    --setenv=CUDA_HOME=$E/nocuda-bin --setenv=CUDA_PATH=$E/nocuda-bin --setenv=CUDARC_CUDA_VERSION=13030
    --setenv=PATH=$HOME/.cargo/bin:/usr/bin:/bin)
  t=$(date +%s)
  build fmtmmq-build-oxide $E/build-mmq-nonvcc.log $(( $(left) - 60 )) $E/mr-mmq -p mistralrs-cli --features oxide
  rc3=$?
  echo "build (3) rc=$rc3 in $(( $(date +%s) - t ))s"
  grep "titan:" $E/build-mmq-nonvcc.log | sort -u
  grep -iE "nvcc" $E/build-mmq-nonvcc.log | grep -v "nvcc not used\|titan:" | head -5
  grep -E "Finished|^error" -A6 $E/build-mmq-nonvcc.log | head -40
  ls -la $E/target-mmq-oxide/release/mistralrs
else
  echo "skipped: $(left)s left"
fi
echo "== window end $(el)"
