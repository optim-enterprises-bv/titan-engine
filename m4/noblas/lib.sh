# m4/noblas helpers (from m4/pattn/lib.sh): worktrees top-noblas (candle), top-noblas/mr-noblas, oxide-noblas
# (top-noblas/oxide-kernels -> ../oxide-noblas), target-noblas-oxide (seeded from target-pattn-oxide = the deployed
# pattn3 build), units noblas-*. Live service titan-mistral: window_begin stops it, the EXIT trap restarts it if it was
# active. titan-spark is never started.
set -u
E=$HOME/titan-engine; I=$E/top-noblas/m4/noblas; SRC=$E/top-noblas/mr-noblas; OX=$E/oxide-noblas; O=$I/out
TGT=$E/target-noblas-oxide; MBIN=$TGT/release/mistralrs; OLDBIN=$E/bin/mistralrs-titan-swap; BIN=$OLDBIN
M=$HOME/ai/models; PORT=18690; LP=18691; G=$E/m4/g4s
NSYS=/opt/nvidia/nsight-compute/2026.2.1/host/target-linux-x64/nsys
mkdir -p $O; cd $I
el() { echo "$(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
smi() { nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>&1 | head -1; }
stop_units() { for u in noblas-srv noblas-build noblas-coll noblas-ox noblas-gate noblas-tmap noblas-llama noblas-nsys; do systemctl --user stop $u 2>/dev/null; done; }
restic_wait() { # the nightly restic backup (03:45-04:30) must not overlap a window
  while systemctl --user -q is-active restic-backup; do echo "restic-backup active: waiting"; sleep 60; done
  local h=$(date +%H%M); if [ $h -ge 0340 ] && [ $h -le 0430 ]; then echo "in the restic window ($h): refusing to start"; exit 3; fi
}
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
mbuild() { # TIMEOUT_S [extra cargo args]
  local to=$1; shift
  local L=$O/build-$WN.log t=$(date +%s)
  systemctl --user reset-failed noblas-build 2>/dev/null
  timeout $(( to + 30 )) systemd-run --user --unit=noblas-build --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 -p RuntimeMaxSec=$to \
    -p WorkingDirectory=$SRC "${CARGO_ENV[@]}" -p StandardOutput=truncate:$L -p StandardError=truncate:$L \
    nice -n 10 cargo build --release -p mistralrs-cli --features oxide "$@"
  local rc=$?; systemctl --user stop noblas-build 2>/dev/null; grep -q "^error" $L && rc=1
  echo "mistral.rs build rc=$rc in $(( $(date +%s) - t ))s ($(el))"; grep -E "^error" -A16 $L | head -150; grep "Finished" $L
  [ $rc = 0 ] && [ -x $MBIN ]
}
oxbuild() { # DIR CRATE UNIT TIMEOUT: cargo oxide build of one crate (arch from its b.sh, default sm_120)
  local d=$1 c=$2 u=$3 L=$O/oxbuild-$(basename $1)-$2.log arch=sm_120
  grep -q "sm_120a" $d/$c/b.sh 2>/dev/null && arch=sm_120a
  systemctl --user reset-failed $u 2>/dev/null
  timeout $4 systemd-run --user --unit=$u --collect --wait -q -p MemoryMax=${OXMEM:-4G} -p MemorySwapMax=0 -p RuntimeMaxSec=$(( $4 - 10 )) \
    -p WorkingDirectory=$d/$c --setenv=CUDA_OXIDE_BACKEND=$E/cuda-oxide-fast/librustc_codegen_cuda.so \
    --setenv=CARGO_BUILD_JOBS=2 --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 --setenv=PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:/usr/bin:/bin \
    -p StandardOutput=truncate:$L -p StandardError=truncate:$L nice -n 10 cargo oxide build --arch $arch
  local rc=$?; grep -q "^error" $L && rc=1
  echo "oxide build $(basename $d)/$c ($arch) rc=$rc ($(el))"; grep -E "^error" -A14 $L | head -60; grep -E "Finished" $L | tail -1
  return $rc
}
kgate() { # DIR CRATE LOG TIMEOUT [ENV=V ...]: run a crate's gate binary under systemd (GPU; 8G cap)
  local d=$1 c=$2 L=$3 to=$4; shift 4
  local se=(); for kv in "$@"; do se+=("--setenv=$kv"); done
  systemctl --user reset-failed noblas-gate 2>/dev/null
  timeout $to systemd-run --user --unit=noblas-gate --collect --wait -q -p MemoryMax=8G -p MemorySwapMax=0 -p RuntimeMaxSec=$(( to - 10 )) \
    -p WorkingDirectory=$d/$c --setenv=LD_LIBRARY_PATH=/usr/local/cuda/lib64 "${se[@]}" -p StandardOutput=truncate:$L -p StandardError=truncate:$L $d/$c/target/release/$c
  local rc=$?; echo "gate $(basename $d)/$c rc=$rc ($(el)): $(grep -E -- '-> (PASS|FAIL)' $L | tail -1)"; return $rc
}
waitfix() { # TAG MIN_LEFT_S: wait for $I/fix.flag (written after a fix) while time allows
  rm -f $I/fix.flag; date -Is > $O/$1-failed.flag; echo "!! $1 failed: waiting for $I/fix.flag ($(left)s left)"
  while [ ! -e $I/fix.flag ] && [ $(left) -gt $2 ]; do sleep 10; done
  rm -f $O/$1-failed.flag; [ -e $I/fix.flag ] && { rm -f $I/fix.flag; return 0; }; return 1
}
# cfg_start NAME TOML [WRAP...]: from-config server on $PORT (the deployed service's env / MemoryMax), optional wrapper
# command (nsys) in front of the binary.
cfg_start() {
  local n=$1 toml=$2; shift 2
  SRV=$n; SLOG=$O/$n.server.log
  systemctl --user stop noblas-srv 2>/dev/null; systemctl --user reset-failed noblas-srv 2>/dev/null
  [ $(left) -gt 120 ] || { echo "!! $n skipped: $(left)s left"; return 1; }
  local se=(); for kv in ${SENV:-}; do se+=("--setenv=$kv"); done
  systemd-run --user --unit=noblas-srv --collect -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=$(( $(left) + 60 )) \
    -p KillMode=mixed -p KillSignal=${KSIG:-SIGTERM} -p TimeoutStopSec=${KTO:-15} -p WorkingDirectory=$I \
    --setenv=LD_LIBRARY_PATH=$E/lib:/usr/local/cuda/lib64 "${se[@]}" \
    -p StandardOutput=truncate:$SLOG -p StandardError=truncate:$SLOG "$@" $BIN from-config --file $toml
  local t0=$(date +%s)
  for i in $(seq 1 600); do
    curl -sf -m 2 -o /dev/null localhost:$PORT/v1/models && { echo "-- $n up in $(( $(date +%s) - t0 ))s ($(el)); GPU $(smi)"; return 0; }
    systemctl --user -q is-active noblas-srv || { echo "!! $n died"; sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -v " INFO " | tail -30; return 1; }
    sleep 1
  done
  echo "!! $n not up"; return 1
}
srv_stop() {
  local t=$(date +%s)
  systemctl --user stop noblas-srv 2>/dev/null
  echo "-- $SRV stopped in $(( $(date +%s) - t ))s ($(el))"
  sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -E "panicked|ILLEGAL|OUT_OF_MEMORY|out of memory|CUDA_ERROR|ERROR|nvcc-only|TITAN_NVCC_ONLY|not ported" | tail -8 | cut -c1-300
}
# --- window 2+: the gemm crate and nsys timing ---
GX=$OX/gemm
gemm_build() { # build the gemm crate until it succeeds (fix loop: $I/fix.flag)
  until oxbuild $OX gemm noblas-ox 1200; do waitfix oxbuild 900 || return 1; done
  ls -la $GX/gemm.ptx | cut -c1-200; grep -c "^.visible .entry" $GX/gemm.ptx
  grep -E "not unrolled|warning: unused" $O/oxbuild-oxide-noblas-gemm.log | sort | uniq -c | head
}
# nsys_run NAME TOML CLIENT_ARGS...: nsys (cuda + cublas NVTX, 100 ms flush) around a from-config server, inv.py client
nsys_run() {
  local n=$1 toml=$2; shift 2
  mkdir -p $O/nsys-tmp; rm -f $O/$n.nsys-rep $O/$n.glog
  if SENV="TITAN_GEMM_LOG=$O/$n.glog TMPDIR=$O/nsys-tmp" KSIG=SIGINT KTO=300 cfg_start $n $toml \
       $NSYS profile -t cuda,cublas --sample=none --cpuctxsw=none --cuda-graph-trace=node --cuda-flush-interval=100 -o $O/$n -f true; then
    timeout 900 python3 $I/inv.py $PORT "$@" $O/$n.glog $O/$n.json
    srv_stop
  fi
  ls -la $O/$n.nsys-rep 2>&1 | cut -c1-200
}
# --- phase 2: swap-server gates ---
NEWBIN=$E/bin/mistralrs-titan-noblas
gm() { # MODEL MODE OUT ARGS...: gates_m.py through the swap server on $PORT
  local m=$1 mode=$2 out=$3; shift 3
  [ $(left) -gt 200 ] || { echo "skip $m $mode: $(left)s left"; return 1; }
  MODEL=$m timeout 900 python3 $I/gates_m.py $mode $PORT titan $O/$out.json "$@" 2>&1 | tail -1
}
# roster_gates TAG: G1 / G3 / G5 per roster model through one from-config server ($BIN, deploy/models.toml on $PORT)
roster_gates() {
  local tag=$1
  sed "s/^port = 1234/port = $PORT/" $E/deploy/models.toml > $O/roster-$tag.toml
  cfg_start sw-$tag $O/roster-$tag.toml || return 1
  while read m prompts g3chars; do
    [ -z "$m" ] || [ "${m:0:1}" = "#" ] && continue
    echo "-- $tag $m ($(el), $(left)s left)"
    timeout 600 python3 $I/swapcli.py touch $PORT $m
    gm $m g1 $tag-$m-g1 $prompts
    [ "$g3chars" != 0 ] && gm $m g3 $tag-$m-g3 $g3chars
    gm $m g5 $tag-$m-g5 8000 128
    case "$tag:$m" in new:spark-x2.5|new:qwen3-14b) timeout 900 python3 $I/swapcli.py burst $PORT $m $tag-$m-burst 4 28000 64 | tail -1;; esac
  done < $I/roster.gate
  srv_stop
  sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -E "titan swap: (loaded|unloaded)" | sed -E 's/^.*INFO [a-z_:0-9]*: //' | cut -c1-160
}
coll() { # NAME DIR ENV...: the 35B 40 x 256 greedy gate run (m4/pattn coll), output m3/out/NAME.json
  local n=$1 dir=$2; shift 2
  [ $(left) -gt 360 ] || { echo "!! $n skipped: $(left)s left"; return 1; }
  systemctl --user reset-failed noblas-coll 2>/dev/null
  timeout 600 systemd-run --user --unit=noblas-coll --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=580 \
    --setenv=BIN=$BIN --setenv=DIR=$dir --setenv=FILE=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    bash $E/sync/w094/collect094.sh $n $E/m4/prompts-eval.txt 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt \
      TITAN_PFS=1 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64 TITAN_TIERED_RESERVE_MIB=1536 "$@" 2>&1 | grep -v " INFO "
  echo "-- $n ($(el))"
}
cmp40() { python3 -c "import json,sys;a,b=json.load(open(sys.argv[1])),json.load(open(sys.argv[2]));print('GATE',sys.argv[2].split('/')[-1],'vs',sys.argv[1].split('/')[-1],':',sum(x==y for x,y in zip(a,b)),'/',len(a),'len',len(b))" $1 $2; }
# --- llama.cpp references (GPU, inside a window) ---
LLAMA=$HOME/ai/llama.cpp/build/bin/llama-server
# lref NAME GGUF LLAMABIN G3CHARS [llama args...]: G1 (plain40) + G3 references, f16 KV, temp 0, -c 12288
lref() {
  local n=$1 f=$2 lb=$3 g3=$4; shift 4
  [ $(left) -gt 300 ] || { echo "!! lref $n skipped: $(left)s left"; return 1; }
  systemctl --user stop noblas-llama 2>/dev/null; systemctl --user reset-failed noblas-llama 2>/dev/null
  systemd-run --user --unit=noblas-llama --collect -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=$(( $(left) + 30 )) \
    -p StandardOutput=truncate:$O/$n.llama.log -p StandardError=inherit \
    $lb -m $f -c 12288 -ngl 99 -ctk f16 -ctv f16 -np 1 --temp 0 --port $LP --host 127.0.0.1 --no-webui "$@"
  local ok=0; for i in $(seq 1 300); do curl -sf -m 2 -o /dev/null localhost:$LP/health && { ok=1; break; }; systemctl --user -q is-active noblas-llama || break; sleep 2; done
  [ $ok = 1 ] || { echo "!! llama $n not up"; tail -5 $O/$n.llama.log; systemctl --user stop noblas-llama; return 1; }
  echo "-- llama $n up ($(el)); args $*"
  timeout 1200 python3 $G/gates.py g1 $LP llama $O/$n-g1.json $I/plain40.json | tail -1
  [ "$g3" != 0 ] && timeout 1500 python3 $G/gates.py g3 $LP llama $O/$n-g3.json $g3 | tail -1
  systemctl --user stop noblas-llama
}
# lspread NAME GGUF LLAMABIN G3CHARS [llama args...]: llama.cpp's own G3 spread (-ub 256 / -ub 128 / -b 512)
lspread() {
  local n=$1 f=$2 lb=$3 g3=$4; shift 4
  for v in "ub256 -ub 256" "ub128 -ub 128" "b512 -b 512"; do
    set -- $v "$@"; local vn=$1; shift; local va="$1 $2"; shift 2
    [ $(left) -gt 300 ] || { echo "!! spread $n $vn skipped: $(left)s left"; return 1; }
    systemctl --user stop noblas-llama 2>/dev/null; systemctl --user reset-failed noblas-llama 2>/dev/null
    systemd-run --user --unit=noblas-llama --collect -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=$(( $(left) + 30 )) \
      -p StandardOutput=truncate:$O/$n-$vn.llama.log -p StandardError=inherit \
      $lb -m $f -c 12288 -ngl 99 -ctk f16 -ctv f16 -np 1 --temp 0 --port $LP --host 127.0.0.1 --no-webui $va "$@"
    local ok=0; for i in $(seq 1 300); do curl -sf -m 2 -o /dev/null localhost:$LP/health && { ok=1; break; }; systemctl --user -q is-active noblas-llama || break; sleep 2; done
    [ $ok = 1 ] && timeout 900 python3 $G/gates.py g1 $LP llama $O/$n-$vn-g1.json ${G1P:-$I/plain40.json} | tail -1
    [ $ok = 1 ] && [ "$g3" != 0 ] && timeout 1500 python3 $G/gates.py g3 $LP llama $O/$n-$vn-g3.json $g3 | tail -1
    systemctl --user stop noblas-llama
  done
}
# roster_dec TAG: decode speed per roster model through one from-config server ($BIN)
roster_dec() {
  local tag=$1
  sed "s/^port = 1234/port = $PORT/" $E/deploy/models.toml > $O/roster-$tag.toml
  cfg_start dec-$tag $O/roster-$tag.toml || return 1
  while read m rest; do
    [ -z "$m" ] || [ "${m:0:1}" = "#" ] && continue
    [ $(left) -gt 200 ] || break
    timeout 600 python3 $I/swapcli.py touch $PORT $m > /dev/null
    timeout 900 python3 $I/dec.py $PORT $m $O/dec-$tag-$m.json 2 128
  done < $I/roster.gate
  srv_stop
}
