# m4/orca helpers (from m4/devmap/lib.sh): worktree top-orca/mr-orca, oxide-orca (top-orca/oxide-kernels -> ../oxide-orca;
# oxide-orca-gate = a reflink copy where the kernel gates rebuild PTX, so the embed never sees a regenerated file),
# target-orca-oxide, units orca-*. Live service titan-mistral: window_begin stops it, the EXIT trap restarts it if it was active.
# titan-spark is never started.
set -u
E=$HOME/titan-engine; I=$E/top-orca/m4/orca; SRC=$E/top-orca/mr-orca; OX=$E/oxide-orca; OXG=$E/oxide-orca-gate; O=$I/out
TGT=$E/target-orca-oxide; MBIN=$TGT/release/mistralrs; NEWBIN=$E/bin/mistralrs-titan-orca; IQBIN=$E/bin/mistralrs-titan-iq4xs; OLDBIN=$E/bin/mistralrs-titan-swap
M=$HOME/ai/models; PORT=18660; C=python3; CL=$I/b2client.py; SP=$E/m4/sparkpf
mkdir -p $O; cd $I
el() { echo "$(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
smi() { nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>&1 | head -1; }
stop_units() { for u in orca-srv orca-build orca-coll orca-ox orca-ox2 orca-llama orca-tmap g3spread-llama; do systemctl --user stop $u 2>/dev/null; done; }
window_begin() { # NAME MINUTES
  WN=$1; T0=$(date +%s); DEADLINE=$((T0 + $2 * 60))
  MISTRAL=$(systemctl --user is-active titan-mistral)
  trap 'stop_units; [ "$MISTRAL" = active ] && systemctl --user start titan-mistral; sleep 2; echo "== $WN exit $(date -Is) ($(el)): titan-mistral $(systemctl --user is-active titan-mistral) (was $MISTRAL), titan-spark $(systemctl --user is-active titan-spark)"' EXIT
  trap 'exit 130' INT TERM HUP
  systemctl --user stop titan-mistral
  echo "== $WN start $(date -Is), titan-mistral was: $MISTRAL, now: $(systemctl --user is-active titan-mistral), titan-spark: $(systemctl --user is-active titan-spark)"
  for p in $(ps -eo pid=,ppid=,comm= | awk '$2 == 878468 && $3 ~ /rust-analyzer/ {print $1}'); do echo "killing rust-analyzer $p"; kill $p; done
  free -m | head -2; sleep 3; echo "GPU idle: $(smi)"
}
CARGO_ENV=(--setenv=CARGO_TARGET_DIR=$TGT --setenv=TITAN_OXIDE_DIR=$OX --setenv=CUDA_HOME=$E/nocuda-bin
  --setenv=CUDA_PATH=$E/nocuda-bin --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDARC_CUDA_VERSION=13030 --setenv=CARGO_BUILD_JOBS=2
  --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 "--setenv=RUSTFLAGS=-L $E/lib" --setenv=LD_LIBRARY_PATH=$E/lib
  --setenv=PATH=$HOME/.cargo/bin:/usr/bin:/bin)
mbuild() { # TIMEOUT_S
  local L=$O/build-$WN.log t=$(date +%s)
  systemctl --user reset-failed orca-build 2>/dev/null
  timeout $(( $1 + 30 )) systemd-run --user --unit=orca-build --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 -p RuntimeMaxSec=$1 \
    -p WorkingDirectory=$SRC "${CARGO_ENV[@]}" -p StandardOutput=truncate:$L -p StandardError=truncate:$L \
    nice -n 10 cargo build --release -p mistralrs-cli --features oxide
  local rc=$?; systemctl --user stop orca-build 2>/dev/null; grep -q "^error" $L && rc=1
  echo "mistral.rs build rc=$rc in $(( $(date +%s) - t ))s ($(el))"; grep -E "^error" -A16 $L | head -150; grep "Finished" $L
  [ $rc = 0 ] && [ -x $MBIN ]
}
mtest() { # LOGNAME TIMEOUT_S CARGO-TEST-ARGS...
  local n=$1 to=$2; shift 2; local L=$O/test-$n.log t=$(date +%s)
  systemctl --user reset-failed orca-build 2>/dev/null
  timeout $(( to + 30 )) systemd-run --user --unit=orca-build --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 -p RuntimeMaxSec=$to \
    -p WorkingDirectory=$SRC "${CARGO_ENV[@]}" -p StandardOutput=truncate:$L -p StandardError=truncate:$L nice -n 10 cargo test --release "$@"
  local rc=$?; systemctl --user stop orca-build 2>/dev/null
  echo "cargo test $n rc=$rc in $(( $(date +%s) - t ))s ($(el))"; grep -E "^error|^test |test result|panicked" $L | head -60
  return $rc
}
# srv_start NAME DIR FILE "ENV=V ..." [serve args...]; BIN, MEM (default 20G); waits for /v1/models
srv_start() {
  local n=$1 dir=$2 f=$3 envs=$4; shift 4
  SRV=$n; SLOG=$O/$n.server.log
  systemctl --user stop orca-srv 2>/dev/null; systemctl --user reset-failed orca-srv 2>/dev/null
  local se=(); for kv in $envs; do se+=("--setenv=$kv"); done
  [ $(left) -gt 120 ] || { echo "!! $n skipped: $(left)s left"; return 1; }
  systemd-run --user --unit=orca-srv --collect -q -p MemoryMax=${MEM:-20G} -p MemorySwapMax=0 -p RuntimeMaxSec=$(( $(left) + 60 )) \
    -p KillSignal=SIGTERM -p TimeoutStopSec=15 -p WorkingDirectory=$I --setenv=LD_LIBRARY_PATH=$E/lib "${se[@]}" \
    -p StandardOutput=truncate:$SLOG -p StandardError=truncate:$SLOG \
    $BIN --seed 0 serve -p $PORT --no-ui --paged-attn off --format gguf -m $dir -f $f "$@"
  local t0=$(date +%s)
  for i in $(seq 1 400); do
    curl -sf -m 2 -o /dev/null localhost:$PORT/v1/models && { echo "-- $n up in $(( $(date +%s) - t0 ))s ($(el)); GPU $(smi); env: $envs; args: $*"; maplines $SLOG; return 0; }
    systemctl --user -q is-active orca-srv || { echo "!! $n died"; grep -v " INFO " $SLOG | tail -30; maplines $SLOG; return 1; }
    sleep 1
  done
  echo "!! $n not up"; return 1
}
srv_stop() {
  systemctl --user stop orca-srv 2>/dev/null
  echo "-- $SRV stopped ($(el))"
  sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -E "panicked|ILLEGAL|OUT_OF_MEMORY|out of memory|CUDA_ERROR|ERROR" | tail -6 | cut -c1-300
}
maplines() { sed 's/\x1b\[[0-9;]*m//g' $1 | grep -E "Layers [0-9]|Automatic device map estimate|tensor inventory|DType selected|device mapping parameters|titan tiered" | sed 's/^.*INFO [a-z_:]*: //' | cut -c1-420; }
memlines() { sed 's/\x1b\[[0-9;]*m//g' $1 | grep "titan devmap" | sed 's/^.*INFO [a-z_:]*: //' | cut -c1-260; }
probe() { timeout ${PT:-1200} python3 $I/probe.py $PORT "$@"; }
cfg_start() { # NAME TOML; from-config, the deployed service's ExecStart / env / MemoryMax
  local n=$1 toml=$2
  SRV=$n; SLOG=$O/$n.server.log
  systemctl --user stop orca-srv 2>/dev/null; systemctl --user reset-failed orca-srv 2>/dev/null
  [ $(left) -gt 120 ] || { echo "!! $n skipped: $(left)s left"; return 1; }
  systemd-run --user --unit=orca-srv --collect -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=$(( $(left) + 60 )) \
    -p KillSignal=SIGTERM -p TimeoutStopSec=15 -p WorkingDirectory=$HOME --setenv=LD_LIBRARY_PATH=$E/lib ${CFGENV:-} \
    -p StandardOutput=truncate:$SLOG -p StandardError=truncate:$SLOG $BIN from-config --file $toml
  local t0=$(date +%s)
  for i in $(seq 1 600); do
    curl -sf -m 2 -o /dev/null localhost:$PORT/v1/models && { echo "-- $n up in $(( $(date +%s) - t0 ))s ($(el)); GPU $(smi)"; return 0; }
    systemctl --user -q is-active orca-srv || { echo "!! $n died"; grep -v " INFO " $SLOG | tail -30; return 1; }
    sleep 1
  done
  echo "!! $n not up"; return 1
}
coll() { # NAME DIR ENV... (35B gate run as m4/integ3 coll); output m3/out/NAME.json
  local n=$1 dir=$2; shift 2
  [ $(left) -gt 360 ] || { echo "!! $n skipped: $(left)s left"; return 1; }
  systemctl --user reset-failed orca-coll 2>/dev/null
  timeout 600 systemd-run --user --unit=orca-coll --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=580 \
    --setenv=BIN=$BIN --setenv=DIR=$dir --setenv=FILE=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    bash $E/sync/w094/collect094.sh $n $E/m4/prompts-eval.txt 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt \
      TITAN_PFS=1 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64 TITAN_TIERED_RESERVE_MIB=1536 "$@" 2>&1 | grep -v " INFO "
  echo "-- $n ($(el))"
  sed 's/\x1b\[[0-9;]*m//g' $E/m3/out/$n.server.log | grep -E "titan tiered|panicked|CUDA_ERROR|Layers [0-9]" | sed 's/.*INFO [a-z_:]*: //' | cut -c1-220 | head -6
}
cmp40() { python3 -c "import json,sys;a,b=json.load(open(sys.argv[1])),json.load(open(sys.argv[2]));print('GATE',sys.argv[2].split('/')[-1],'vs',sys.argv[1].split('/')[-1],':',sum(x==y for x,y in zip(a,b)),'/',len(a),'len',len(b))" $1 $2; }
# --- devmap2 additions (from m4/devmap/lib.sh) ---
LLAMA=$HOME/ai/llama.cpp/build/bin/llama-server; LP=18661; G=$E/m4/g4s
oxbuild() { # CRATE UNIT TIMEOUT: cargo oxide build of one crate in oxide-orca-gate (arch from its b.sh)
  local c=$1 u=$2 L=$O/oxbuild-$1.log arch=sm_120
  grep -q "sm_120a" $OXG/$c/b.sh 2>/dev/null && arch=sm_120a
  systemctl --user reset-failed $u 2>/dev/null
  timeout $3 systemd-run --user --unit=$u --collect --wait -q -p MemoryMax=4G -p MemorySwapMax=0 -p RuntimeMaxSec=$(( $3 - 10 )) \
    -p WorkingDirectory=$OXG/$c --setenv=CUDA_OXIDE_BACKEND=$E/cuda-oxide-fast/librustc_codegen_cuda.so \
    --setenv=CARGO_BUILD_JOBS=2 --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 --setenv=PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:/usr/bin:/bin \
    -p StandardOutput=truncate:$L -p StandardError=truncate:$L nice -n 10 cargo oxide build --arch $arch
  local rc=$?; grep -q "^error" $L && rc=1
  echo "oxide build $c ($arch) rc=$rc ($(el))"; grep -E "^error" -A14 $L | head -60
  return $rc
}
lref() { # NAME GGUF OUTPREFIX [extra args]: llama.cpp reference (f16 KV, ngl 99, 12288 ctx), G1 + G3 + G3l
  local n=$1 f=$2 op=$3; shift 3
  systemctl --user stop orca-llama 2>/dev/null; systemctl --user reset-failed orca-llama 2>/dev/null
  systemd-run --user --unit=orca-llama --collect -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=1500 \
    -p StandardOutput=truncate:$O/$n.llama.log -p StandardError=inherit \
    $LLAMA -m $f -c 12288 -ngl 99 -ctk f16 -ctv f16 -np 1 --temp 0 --port $LP --host 127.0.0.1 --no-webui "$@"
  for i in $(seq 1 150); do curl -sf -m 2 -o /dev/null localhost:$LP/health && break; sleep 2; done
  timeout 600 python3 $G/gates.py g1 $LP llama $O/$op-g1.json $I/prompts/qwen3-14b.json | tail -1
  timeout 600 python3 $G/gates.py g3 $LP llama $O/$op-g3.json | tail -1
  timeout 600 python3 $G/gates.py g3 $LP llama $O/$op-g3l.json 28000 | tail -1
  systemctl --user stop orca-llama
}
tmapq() { timeout 300 systemd-run --user --unit=orca-tmap --collect --wait --pipe -q -p MemoryMax=4G $HOME/ai/convert-env/bin/python3 $I/tokmap_qwen3.py "$@"; }
tmapg() { timeout 300 systemd-run --user --unit=orca-tmap --collect --wait --pipe -q -p MemoryMax=4G $HOME/ai/convert-env/bin/python3 $G/tokmap.py "$@"; }
tgate() { timeout 900 python3 $G/gates.py "$1" $PORT titan "${@:2}"; }
cmpg() { python3 $G/cmp.py "$@" 2>&1; }
# --- orca additions ---
OD=$M/orcasaq2-cyber-27b; OF=OrcaSAQ-2-27B-Uncensored.gguf; GP=$I/orca-g1.json
tmo() { timeout 300 systemd-run --user --unit=orca-tmap --collect --wait --pipe -q -p MemoryMax=4G $HOME/ai/convert-env/bin/python3 $I/tokmap_gguf.py $OD/$OF "$@"; }
waitfix() { # TAG MIN_LEFT_S: wait for $I/fix.flag (written after a fix) while time allows
  rm -f $I/fix.flag; date -Is > $O/$1-failed.flag; echo "!! $1 failed: waiting for $I/fix.flag ($(left)s left)"
  while [ ! -e $I/fix.flag ] && [ $(left) -gt $2 ]; do sleep 10; done
  rm -f $O/$1-failed.flag; [ -e $I/fix.flag ] && { rm -f $I/fix.flag; return 0; }; return 1
}
build_bin() { # TIMEOUT_S: mbuild until it succeeds (fix loop), then stage NEWBIN
  until mbuild $1; do waitfix mbuild 1200 || { echo "no binary: ending the window"; exit 1; }; done
  cp $MBIN $NEWBIN; BIN=$NEWBIN; echo "binary: $BIN sha256 $(sha256sum $BIN | cut -d' ' -f1) from $(git -C $SRC log --oneline -1 | cut -c1-60) $(git -C $SRC diff --quiet && echo clean || echo DIRTY)"
}
llama_srv() { # NAME GGUF ARGS...: llama-server on $LP, waits for /health
  local n=$1 f=$2; shift 2
  systemctl --user stop orca-llama 2>/dev/null; systemctl --user reset-failed orca-llama 2>/dev/null
  systemd-run --user --unit=orca-llama --collect -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=1500 \
    -p StandardOutput=truncate:$O/$n.llama.log -p StandardError=inherit $LLAMA -m $f -ngl 99 -np 1 --temp 0 --port $LP --host 127.0.0.1 --no-webui "$@"
  for i in $(seq 1 150); do curl -sf -m 2 -o /dev/null localhost:$LP/health && { echo "-- llama $n up ($(el)); GPU $(smi)"; return 0; }; sleep 2; done
  echo "!! llama $n not up"; return 1
}
gate() { # MODE OUT ARGS... (titan on $PORT; gates.py writes $O/OUT.json)
  [ $(left) -gt 240 ] || { echo "skip $1 $2: $(left)s left"; return 1; }
  local mode=$1 out=$2; shift 2
  timeout 900 python3 $G/gates.py $mode $PORT titan $O/$out.json "$@"
}
same() { python3 $I/same.py "$@"; }
