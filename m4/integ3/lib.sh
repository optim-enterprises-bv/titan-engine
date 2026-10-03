# m4/integ3 helpers (integ3-20261003 = integ2-20261003 + deq + g4acc + redcell2 + kv_cache OOM fix): m4/integ2/lib.sh
# retargeted at top-integ3/mr-integ3, oxide-integ3, target-integ3-oxide; units integ3-*.
# Service is titan-spark (stopped by window_begin, restarted on EXIT); titan-mistral is never started.
set -u
E=$HOME/titan-engine; I=$E/top-integ3/m4/integ3; SRC=$E/top-integ3/mr-integ3; OX=$E/oxide-integ3; O=$I/out
TGT=$E/target-integ3-oxide; MBIN=$TGT/release/mistralrs; NEWBIN=$E/bin/mistralrs-titan-integ3; OLDBIN=$E/bin/mistralrs-titan-integ2
M=$HOME/ai/models; PORT=18640; LP=18641
G4=$E/m4/g4s; R=$G4/out; SP=$E/m4/sparkpf; IO=$E/m4/integ/out; C=python3; CL=$I/b2client.py
mkdir -p $O; cd $I
el() { echo "$(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
smi() { nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>&1 | head -1; }
stop_units() { for u in integ3-srv integ3-build integ3-coll integ3-llama integ3-ox integ3-oxcq integ3-meta integ3-tmap; do systemctl --user stop $u 2>/dev/null; done; }
window_begin() { # NAME MINUTES
  WN=$1; T0=$(date +%s); DEADLINE=$((T0 + $2 * 60))
  SPARK=$(systemctl --user is-active titan-spark)
  trap 'stop_units; [ "$SPARK" = active ] && systemctl --user start titan-spark; echo "== $WN exit $(date -Is) ($(el)): titan-spark $(systemctl --user is-active titan-spark), titan-mistral $(systemctl --user is-active titan-mistral)"' EXIT
  trap 'exit 130' INT TERM HUP
  systemctl --user stop titan-spark
  echo "== $WN start $(date -Is), titan-spark was: $SPARK, now: $(systemctl --user is-active titan-spark), titan-mistral: $(systemctl --user is-active titan-mistral)"
  for p in $(ps -eo pid=,ppid=,comm= | awk '$2 == 878468 && $3 ~ /rust-analyzer/ {print $1}'); do echo "killing rust-analyzer $p"; kill $p; done
  free -m | head -2; sleep 3; echo "GPU idle: $(smi)"
}
# cargo oxide build of one oxide-integ crate: CRATE UNIT TIMEOUT
oxbuild() {
  local c=$1 u=$2 L=$O/oxbuild-$1.log
  # each crate states its arch in b.sh (fmt_mmq needs sm_120a for the block-scaled mma); default sm_120
  local arch=sm_120; grep -q "sm_120a" $OX/$c/b.sh 2>/dev/null && arch=sm_120a
  systemctl --user reset-failed $u 2>/dev/null
  timeout $3 systemd-run --user --unit=$u --collect --wait -q -p MemoryMax=4G -p MemorySwapMax=0 -p RuntimeMaxSec=$(( $3 - 10 )) \
    -p WorkingDirectory=$OX/$c --setenv=CUDA_OXIDE_BACKEND=$E/cuda-oxide-fast/librustc_codegen_cuda.so \
    --setenv=CARGO_BUILD_JOBS=2 --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 --setenv=PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:/usr/bin:/bin \
    -p StandardOutput=truncate:$L -p StandardError=truncate:$L nice -n 10 cargo oxide build --arch $arch
  local rc=$?; grep -q "^error" $L && rc=1
  echo "oxide build $c rc=$rc ($(el))"; grep -E "^error" -A12 $L | head -40
  return $rc
}
# mistral.rs nvcc-free build (RULES caps; the m4/b2merge / m4/g4s command), TIMEOUT_S
mbuild() {
  local L=$O/build-$WN.log t=$(date +%s)
  systemctl --user reset-failed integ3-build 2>/dev/null
  timeout $(( $1 + 30 )) systemd-run --user --unit=integ3-build --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 -p RuntimeMaxSec=$1 \
    -p WorkingDirectory=$SRC --setenv=CARGO_TARGET_DIR=$TGT --setenv=TITAN_OXIDE_DIR=$OX --setenv=CUDA_HOME=$E/nocuda-bin \
    --setenv=CUDA_PATH=$E/nocuda-bin --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDARC_CUDA_VERSION=13030 --setenv=CARGO_BUILD_JOBS=2 \
    --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 "--setenv=RUSTFLAGS=-L $E/lib" --setenv=PATH=$HOME/.cargo/bin:/usr/bin:/bin \
    -p StandardOutput=truncate:$L -p StandardError=truncate:$L nice -n 10 cargo build --release -p mistralrs-cli --features oxide
  local rc=$?; systemctl --user stop integ3-build 2>/dev/null; grep -q "^error" $L && rc=1
  echo "mistral.rs build rc=$rc in $(( $(date +%s) - t ))s ($(el))"; grep -E "^error" -A16 $L | head -150; grep "Finished" $L
  [ $rc = 0 ] && [ -x $MBIN ] && [ $MBIN -nt $SRC/Cargo.toml ]
}
# titan server: [BIN=.. MAXSEQ=.. ] srv_start NAME DIR FILE "ENV=V ..." [serve args...]; waits for /v1/models
srv_start() {
  local n=$1 dir=$2 f=$3 envs=$4; shift 4
  SRV=$n; SLOG=$O/$n.server.log
  systemctl --user stop integ3-srv 2>/dev/null; systemctl --user reset-failed integ3-srv 2>/dev/null
  local se=(); for kv in $envs; do se+=("--setenv=$kv"); done
  [ $(left) -gt 120 ] || { echo "!! $n skipped: $(left)s left"; return 1; }
  systemd-run --user --unit=integ3-srv --collect -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=$(( $(left) + 60 )) \
    -p KillSignal=SIGTERM -p TimeoutStopSec=15 -p WorkingDirectory=$I --setenv=LD_LIBRARY_PATH=$E/lib "${se[@]}" \
    -p StandardOutput=truncate:$SLOG -p StandardError=truncate:$SLOG \
    $BIN --seed 0 serve -p $PORT --no-ui --paged-attn off --format gguf -m $dir -f $f "$@"
  local t0=$(date +%s)
  for i in $(seq 1 400); do
    curl -sf -m 2 -o /dev/null localhost:$PORT/v1/models && { echo "-- $n up in $(( $(date +%s) - t0 ))s ($(el)); GPU $(smi); env: $envs"; return 0; }
    systemctl --user -q is-active integ3-srv || { echo "!! $n died"; grep -v " INFO " $SLOG | tail -30; return 1; }
    sleep 1
  done
  echo "!! $n not up"; return 1
}
srv_stop() {
  systemctl --user stop integ3-srv 2>/dev/null
  echo "-- $SRV stopped ($(el))"
  grep -E "panicked|ILLEGAL|OUT_OF_MEMORY|out of memory|CUDA_ERROR|ERROR|titan mtp stats" $SLOG | tail -6 | cut -c1-300
}
# 35B gate run (m4/b2merge/lib.sh coll: the env those references were gated with): coll NAME DIR ENV...; output m3/out/NAME.json
coll() {
  local n=$1 dir=$2; shift 2
  [ $(left) -gt 360 ] || { echo "!! $n skipped: $(left)s left"; return 1; }
  systemctl --user reset-failed integ3-coll 2>/dev/null
  timeout 600 systemd-run --user --unit=integ3-coll --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=580 \
    --setenv=BIN=$BIN --setenv=DIR=$dir --setenv=FILE=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    bash $E/sync/w094/collect094.sh $n $E/m4/prompts-eval.txt 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt \
      TITAN_PFS=1 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64 TITAN_TIERED_RESERVE_MIB=1536 "$@" 2>&1 | grep -v " INFO "
  echo "-- $n ($(el))"
  grep -E "titan tiered|panicked|CUDA_ERROR" $E/m3/out/$n.server.log | sed 's/.*INFO //' | cut -c1-220 | head -3
}
cmp40() { python3 -c "import json,sys;a,b=json.load(open(sys.argv[1])),json.load(open(sys.argv[2]));print('GATE',sys.argv[2].split('/')[-1],'vs',sys.argv[1].split('/')[-1],':',sum(x==y for x,y in zip(a,b)),'/',len(a),'len',len(b))" $1 $2; }
# cargo test of one crate in the release profile (deps reused from mbuild): mtest LOGNAME TIMEOUT_S CARGO-TEST-ARGS...
mtest() {
  local n=$1 to=$2; shift 2; local L=$O/test-$n.log t=$(date +%s)
  systemctl --user reset-failed integ3-build 2>/dev/null
  timeout $(( to + 30 )) systemd-run --user --unit=integ3-build --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 -p RuntimeMaxSec=$to \
    -p WorkingDirectory=$SRC --setenv=CARGO_TARGET_DIR=$TGT --setenv=TITAN_OXIDE_DIR=$OX --setenv=CUDA_HOME=$E/nocuda-bin \
    --setenv=CUDA_PATH=$E/nocuda-bin --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDARC_CUDA_VERSION=13030 --setenv=CARGO_BUILD_JOBS=2 \
    --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 "--setenv=RUSTFLAGS=-L $E/lib" --setenv=LD_LIBRARY_PATH=$E/lib \
    --setenv=PATH=$HOME/.cargo/bin:/usr/bin:/bin -p StandardOutput=truncate:$L -p StandardError=truncate:$L nice -n 10 cargo test --release "$@"
  local rc=$?; systemctl --user stop integ3-build 2>/dev/null
  echo "cargo test $n rc=$rc in $(( $(date +%s) - t ))s ($(el))"; grep -E "^error|^test |test result|panicked" $L | head -40
  return $rc
}
# from-config server: cfg_start NAME TOML; waits for /v1/models (the default model loaded)
cfg_start() {
  local n=$1 toml=$2
  SRV=$n; SLOG=$O/$n.server.log
  systemctl --user stop integ3-srv 2>/dev/null; systemctl --user reset-failed integ3-srv 2>/dev/null
  [ $(left) -gt 120 ] || { echo "!! $n skipped: $(left)s left"; return 1; }
  systemd-run --user --unit=integ3-srv --collect -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=$(( $(left) + 60 )) \
    -p KillSignal=SIGTERM -p TimeoutStopSec=15 -p WorkingDirectory=${CWD:-$I} --setenv=LD_LIBRARY_PATH=$E/lib \
    -p StandardOutput=truncate:$SLOG -p StandardError=truncate:$SLOG $BIN from-config --file $toml
  local t0=$(date +%s)
  for i in $(seq 1 400); do
    curl -sf -m 2 -o /dev/null localhost:$PORT/v1/models && { echo "-- $n up in $(( $(date +%s) - t0 ))s ($(el)); GPU $(smi)"; return 0; }
    systemctl --user -q is-active integ3-srv || { echo "!! $n died"; grep -v " INFO " $SLOG | tail -30; return 1; }
    sleep 1
  done
  echo "!! $n not up"; return 1
}
gate() { # MODE OUT ARGS... (titan on $PORT, gates.py body model "default")
  [ $(left) -gt 240 ] || { echo "skip $1 $2: $(left)s left"; return 1; }
  local mode=$1 out=$2; shift 2
  timeout 900 python3 $SP/gates.py $mode $PORT titan $O/$out.json "$@"
}
cmpg() { python3 $SP/cmp.py "$@" 2>&1; }
same() { python3 $I/same.py "$@"; }
tmap() { timeout 300 systemd-run --user --unit=integ3-tmap --collect --wait --pipe -q -p MemoryMax=4G $HOME/ai/convert-env/bin/python3 $G4/tokmap.py "$@"; }
g5s() { for f in "$@"; do [ -e $f ] && python3 -c "import json,sys;r=json.load(open(sys.argv[1]));print('TPS',sys.argv[1].split('/')[-1],[(x['prompt_tokens'],round(x['prefill_tps'] or 0,1),round(x['decode_tps'] or 0,1)) for x in r])" $f; done; }
