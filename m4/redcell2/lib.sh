# m4/redcell2 helpers: m4/redcell/lib.sh retargeted at top-redcell2/mr-redcell2, oxide-redcell2, target-redcell2-oxide.
# Service is titan-spark (stopped by window_begin, restarted on EXIT); titan-mistral is never started.
set -u
E=$HOME/titan-engine; I=$E/m4/redcell2; SRC=$E/top-redcell2/mr-redcell2; OX=$E/oxide-redcell2; O=$I/out
TGT=$E/target-redcell2-oxide; MBIN=$TGT/release/mistralrs; INTEG=$E/bin/mistralrs-titan-integ2; NEWBIN=$I/mistralrs-redcell2
M=$HOME/ai/models; PORT=18630; LP=18631
mkdir -p $O; cd $I
el() { echo "$(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
smi() { nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>&1 | head -1; }
stop_units() { for u in red-srv red-build red-coll red-llama red-ox red-oxcq red-meta red-tmap; do systemctl --user stop $u 2>/dev/null; done; sudo -n systemctl stop red-ncu 2>/dev/null; sudo -n chown -R dingo:dingo $O 2>/dev/null; }
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
  systemctl --user reset-failed $u 2>/dev/null
  timeout $3 systemd-run --user --unit=$u --collect --wait -q -p MemoryMax=4G -p MemorySwapMax=0 -p RuntimeMaxSec=$(( $3 - 10 )) \
    -p WorkingDirectory=$OX/$c --setenv=CUDA_OXIDE_BACKEND=$E/cuda-oxide-fast/librustc_codegen_cuda.so \
    --setenv=CARGO_BUILD_JOBS=2 --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 --setenv=PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:/usr/bin:/bin \
    -p StandardOutput=truncate:$L -p StandardError=truncate:$L nice -n 10 cargo oxide build --arch sm_120
  local rc=$?; grep -q "^error" $L && rc=1
  echo "oxide build $c rc=$rc ($(el))"; grep -E "^error" -A12 $L | head -40
  return $rc
}
# mistral.rs nvcc-free build (RULES caps; the m4/b2merge / m4/g4s command), TIMEOUT_S
mbuild() {
  local L=$O/build-$WN.log t=$(date +%s)
  systemctl --user reset-failed red-build 2>/dev/null
  timeout $(( $1 + 30 )) systemd-run --user --unit=red-build --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 -p RuntimeMaxSec=$1 \
    -p WorkingDirectory=$SRC --setenv=CARGO_TARGET_DIR=$TGT --setenv=TITAN_OXIDE_DIR=$OX --setenv=CUDA_HOME=$E/nocuda-bin \
    --setenv=CUDA_PATH=$E/nocuda-bin --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDARC_CUDA_VERSION=13030 --setenv=CARGO_BUILD_JOBS=2 \
    --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 "--setenv=RUSTFLAGS=-L $E/lib" --setenv=PATH=$HOME/.cargo/bin:/usr/bin:/bin \
    -p StandardOutput=truncate:$L -p StandardError=truncate:$L nice -n 10 cargo build --release -p mistralrs-cli --features oxide
  local rc=$?; systemctl --user stop red-build 2>/dev/null; grep -q "^error" $L && rc=1
  echo "mistral.rs build rc=$rc in $(( $(date +%s) - t ))s ($(el))"; grep -E "^error" -A16 $L | head -150; grep "Finished" $L
  [ $rc = 0 ] && [ -x $MBIN ] && [ $MBIN -nt $SRC/Cargo.toml ]
}
# titan server: [BIN=.. MAXSEQ=.. ] srv_start NAME DIR FILE "ENV=V ..." [serve args...]; waits for /v1/models
srv_start() {
  local n=$1 dir=$2 f=$3 envs=$4; shift 4
  SRV=$n; SLOG=$O/$n.server.log
  systemctl --user stop red-srv 2>/dev/null; systemctl --user reset-failed red-srv 2>/dev/null
  local se=(); for kv in $envs; do se+=("--setenv=$kv"); done
  [ $(left) -gt 120 ] || { echo "!! $n skipped: $(left)s left"; return 1; }
  systemd-run --user --unit=red-srv --collect -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=$(( $(left) + 60 )) \
    -p KillSignal=SIGTERM -p TimeoutStopSec=15 -p WorkingDirectory=$I --setenv=LD_LIBRARY_PATH=$E/lib "${se[@]}" \
    -p StandardOutput=truncate:$SLOG -p StandardError=truncate:$SLOG \
    $BIN --seed 0 serve -p $PORT --no-ui --paged-attn off --format gguf -m $dir -f $f "$@"
  local t0=$(date +%s)
  for i in $(seq 1 400); do
    curl -sf -m 2 -o /dev/null localhost:$PORT/v1/models && { echo "-- $n up in $(( $(date +%s) - t0 ))s ($(el)); GPU $(smi); env: $envs"; return 0; }
    systemctl --user -q is-active red-srv || { echo "!! $n died"; grep -v " INFO " $SLOG | tail -30; return 1; }
    sleep 1
  done
  echo "!! $n not up"; return 1
}
srv_stop() {
  systemctl --user stop red-srv 2>/dev/null
  echo "-- $SRV stopped ($(el))"
  grep -E "panicked|ILLEGAL|OUT_OF_MEMORY|out of memory|CUDA_ERROR|ERROR|titan mtp stats" $SLOG | tail -6 | cut -c1-300
}
# 35B gate run (m4/b2merge/lib.sh coll: the env those references were gated with): coll NAME DIR ENV...; output m3/out/NAME.json
coll() {
  local n=$1 dir=$2; shift 2
  [ $(left) -gt 360 ] || { echo "!! $n skipped: $(left)s left"; return 1; }
  systemctl --user reset-failed red-coll 2>/dev/null
  timeout 600 systemd-run --user --unit=red-coll --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=580 \
    --setenv=BIN=$BIN --setenv=DIR=$dir --setenv=FILE=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    bash $E/sync/w094/collect094.sh $n $E/m4/prompts-eval.txt 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt \
      TITAN_PFS=1 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64 TITAN_TIERED_RESERVE_MIB=1536 "$@" 2>&1 | grep -v " INFO "
  echo "-- $n ($(el))"
  grep -E "titan tiered|panicked|CUDA_ERROR" $E/m3/out/$n.server.log | sed 's/.*INFO //' | cut -c1-220 | head -3
}
cmp40() { python3 -c "import json,sys;a,b=json.load(open(sys.argv[1])),json.load(open(sys.argv[2]));print('GATE',sys.argv[2].split('/')[-1],'vs',sys.argv[1].split('/')[-1],':',sum(x==y for x,y in zip(a,b)),'/',len(a),'len',len(b))" $1 $2; }
# ---- redcell additions
G4=$E/m4/g4s; R=$G4/out; SP=$E/m4/sparkpf
RED=REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf; RP=$G4/prompts/redcell.json; RA=(--max-model-len 8192 --max-seq-len 8192 -n 0:30 --dtype bf16)
gate() { [ $(left) -gt 240 ] || { echo "skip $1 $2: $(left)s left"; return 1; }; local mode=$1 out=$2; shift 2; timeout 900 python3 $G4/gates.py $mode $PORT titan $O/$out.json "$@"; }
cmpg() { python3 $G4/cmp.py "$@" 2>&1; }
same() { python3 $I/same.py "$@"; }
tmap() { timeout 300 systemd-run --user --unit=red-tmap --collect --wait --pipe -q -p MemoryMax=4G $HOME/ai/convert-env/bin/python3 $G4/tokmap.py "$@"; }
NCU=/usr/local/cuda/bin/ncu
# one ncu-profiled titan run as a root unit: NAME "ENV=V ..." serve-args... ; then call ncu_req CHARS and ncu_end
ncu_start() { local n=$1 envs=$2; shift 2; local se=(); for kv in $envs; do se+=(--setenv=$kv); done
  NCUN=$n
  sudo -n systemd-run --unit=red-ncu --collect -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=1200 \
    --setenv=LD_LIBRARY_PATH=$E/lib "${se[@]}" /usr/bin/bash -c 'log=$1; shift 1; exec >$log 2>&1; exec "$@"' x $O/$n.ncu.log \
    $NCU --target-processes all --metrics gpu__time_duration.sum,launch__grid_size,launch__block_size \
    --cache-control none --clock-control none --launch-count 2000000 -f -o $O/$n \
    $BIN --seed 0 serve -p $PORT --no-ui --paged-attn off --format gguf "$@"
  local up=0; for i in $(seq 1 500); do curl -sf -m 2 -o /dev/null localhost:$PORT/v1/models && { up=1; break; }; sudo -n systemctl is-active -q red-ncu || break; sleep 2; done
  echo "-- ncu $n up=$up ($(el))"; [ $up = 1 ]
}
ncu_end() { # TERM the profiled server (ncu keeps running to write the report), wait, export CSV
  local pids=$(for p in $(cat /sys/fs/cgroup/system.slice/red-ncu.service/cgroup.procs 2>/dev/null); do
    case "$(cat /proc/$p/comm 2>/dev/null)" in mistralrs*) echo $p;; esac; done)
  [ -n "$pids" ] && sudo -n kill -TERM $pids; sleep 8
  for p in $pids; do [ -e /proc/$p ] && sudo -n kill -KILL $p; done
  for i in $(seq 1 600); do sudo -n systemctl is-active -q red-ncu || break; sleep 1; done
  sudo -n systemctl stop red-ncu 2>/dev/null; sudo -n chown -R dingo:dingo $O
  [ -e $O/$NCUN.ncu-rep ] && timeout 600 $NCU --import $O/$NCUN.ncu-rep --csv --page raw --metrics gpu__time_duration.sum,launch__grid_size > $O/$NCUN.csv 2>/dev/null
  echo "-- ncu $NCUN done ($(el)): $(wc -l < $O/$NCUN.csv 2>/dev/null) rows"; tail -3 $O/$NCUN.ncu.log
}
