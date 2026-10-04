# m4/pattn helpers (from m4/orca/lib.sh): worktree top-pattn/mr-pattn, oxide-pattn (top-pattn/oxide-kernels -> ../oxide-pattn),
# target-pattn-oxide (seeded from target-orca-oxide), units pattn-*. Live service titan-mistral: window_begin stops it, the
# EXIT trap restarts it if it was active. titan-spark is never started.
set -u
E=$HOME/titan-engine; I=$E/top-pattn/m4/pattn; SRC=$E/top-pattn/mr-pattn; OX=$E/oxide-pattn; O=$I/out
TGT=$E/target-pattn-oxide; MBIN=$TGT/release/mistralrs; NEWBIN=$E/bin/mistralrs-titan-pattn; OLDBIN=$E/bin/mistralrs-titan-swap
M=$HOME/ai/models; PORT=18670; LP=18671; C=python3; CL=$I/b2client.py; G=$E/m4/g4s
mkdir -p $O; cd $I
el() { echo "$(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
smi() { nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>&1 | head -1; }
stop_units() { for u in pattn-srv pattn-build pattn-coll pattn-ox pattn-gate pattn-tmap pattn-llama; do systemctl --user stop $u 2>/dev/null; done; }
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
  systemctl --user reset-failed pattn-build 2>/dev/null
  timeout $(( $1 + 30 )) systemd-run --user --unit=pattn-build --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 -p RuntimeMaxSec=$1 \
    -p WorkingDirectory=$SRC "${CARGO_ENV[@]}" -p StandardOutput=truncate:$L -p StandardError=truncate:$L \
    nice -n 10 cargo build --release -p mistralrs-cli --features oxide
  local rc=$?; systemctl --user stop pattn-build 2>/dev/null; grep -q "^error" $L && rc=1
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
  systemctl --user reset-failed pattn-gate 2>/dev/null
  timeout $to systemd-run --user --unit=pattn-gate --collect --wait -q -p MemoryMax=8G -p MemorySwapMax=0 -p RuntimeMaxSec=$(( to - 10 )) \
    -p WorkingDirectory=$d/$c "${se[@]}" -p StandardOutput=truncate:$L -p StandardError=truncate:$L $d/$c/target/release/$c
  local rc=$?; echo "gate $(basename $d)/$c rc=$rc ($(el)): $(grep -E 'launcher calls|-> (PASS|FAIL)' $L | tail -1)"; return $rc
}
# srv_start NAME DIR FILE "ENV=V ..." [serve args...]; BIN, MEM (default 20G), PA (paged-attn mode, default off)
srv_start() {
  local n=$1 dir=$2 f=$3 envs=$4; shift 4
  SRV=$n; SLOG=$O/$n.server.log
  systemctl --user stop pattn-srv 2>/dev/null; systemctl --user reset-failed pattn-srv 2>/dev/null
  local se=(); for kv in $envs; do se+=("--setenv=$kv"); done
  [ $(left) -gt 120 ] || { echo "!! $n skipped: $(left)s left"; return 1; }
  systemd-run --user --unit=pattn-srv --collect -q -p MemoryMax=${MEM:-20G} -p MemorySwapMax=0 -p RuntimeMaxSec=$(( $(left) + 60 )) \
    -p KillSignal=SIGTERM -p TimeoutStopSec=15 -p WorkingDirectory=$I --setenv=LD_LIBRARY_PATH=$E/lib "${se[@]}" \
    -p StandardOutput=truncate:$SLOG -p StandardError=truncate:$SLOG \
    $BIN --seed 0 serve -p $PORT --no-ui --paged-attn ${PA:-off} --format gguf -m $dir -f $f "$@"
  local t0=$(date +%s)
  for i in $(seq 1 400); do
    curl -sf -m 2 -o /dev/null localhost:$PORT/v1/models && { echo "-- $n up in $(( $(date +%s) - t0 ))s ($(el)); GPU $(smi); paged-attn ${PA:-off}; env: $envs; args: $*"; return 0; }
    systemctl --user -q is-active pattn-srv || { echo "!! $n died"; sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -v " INFO " | tail -30; return 1; }
    sleep 1
  done
  echo "!! $n not up"; return 1
}
srv_stop() {
  systemctl --user stop pattn-srv 2>/dev/null
  echo "-- $SRV stopped ($(el))"
  sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -E "panicked|ILLEGAL|OUT_OF_MEMORY|out of memory|CUDA_ERROR|ERROR|nvcc-only|TITAN_NVCC_ONLY|not ported" | tail -8 | cut -c1-300
}
palines() { sed 's/\x1b\[[0-9;]*m//g' $1 | grep -iE "paged|PagedAttention|kv cache|KV cache|blocks|flashinfer|without PagedAttention" | sed 's/^.*\(INFO\|WARN\) [a-z_:0-9]*: //' | head -${2:-8} | cut -c1-260; }
coll() { # NAME DIR ENV... (35B gate run as m4/orca coll); output m3/out/NAME.json
  local n=$1 dir=$2; shift 2
  [ $(left) -gt 360 ] || { echo "!! $n skipped: $(left)s left"; return 1; }
  systemctl --user reset-failed pattn-coll 2>/dev/null
  timeout 600 systemd-run --user --unit=pattn-coll --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=580 \
    --setenv=BIN=$BIN --setenv=DIR=$dir --setenv=FILE=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    bash $E/sync/w094/collect094.sh $n $E/m4/prompts-eval.txt 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt \
      TITAN_PFS=1 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64 TITAN_TIERED_RESERVE_MIB=1536 "$@" 2>&1 | grep -v " INFO "
  echo "-- $n ($(el))"
}
cmp40() { python3 -c "import json,sys;a,b=json.load(open(sys.argv[1])),json.load(open(sys.argv[2]));print('GATE',sys.argv[2].split('/')[-1],'vs',sys.argv[1].split('/')[-1],':',sum(x==y for x,y in zip(a,b)),'/',len(a),'len',len(b))" $1 $2; }
gate() { # MODE OUT ARGS... (titan on $PORT; gates.py writes $O/OUT.json)
  [ $(left) -gt 240 ] || { echo "skip $1 $2: $(left)s left"; return 1; }
  local mode=$1 out=$2; shift 2
  timeout 900 python3 $G/gates.py $mode $PORT titan $O/$out.json "$@"
}
tmapq() { timeout 300 systemd-run --user --unit=pattn-tmap --collect --wait --pipe -q -p MemoryMax=4G $HOME/ai/convert-env/bin/python3 $I/tokmap_qwen3.py "$@"; }
cmpg() { python3 $G/cmp.py "$@" 2>&1; }
same() { python3 $I/same.py "$@"; }
conc() { timeout ${CT:-900} python3 $I/conc.py $PORT "$@"; }
tps() { for f in "$@"; do [ -e $f ] && python3 -c "import json,sys;r=json.load(open(sys.argv[1]));print('TPS',sys.argv[1].split('/')[-1],[(x['prompt_tokens'],round(x['prefill_tps'] or 0,1),round(x['decode_tps'] or 0,1)) for x in r])" $f; done; }
probe() { timeout ${PT:-1200} python3 $I/probe.py $PORT "$@"; }
# --- phase A/B additions ---
NEWBIN2=$E/bin/mistralrs-titan-pattn2; SC=$E/scratch-pattn
waitfix() { # TAG MIN_LEFT_S: wait for $I/fix.flag (written after a fix) while time allows
  rm -f $I/fix.flag; date -Is > $O/$1-failed.flag; echo "!! $1 failed: waiting for $I/fix.flag ($(left)s left)"
  while [ ! -e $I/fix.flag ] && [ $(left) -gt $2 ]; do sleep 10; done
  rm -f $O/$1-failed.flag; [ -e $I/fix.flag ] && { rm -f $I/fix.flag; return 0; }; return 1
}
restic_wait() { # the nightly restic backup (03:45-04:30) must not overlap a window
  while systemctl --user -q is-active restic-backup; do echo "restic-backup active: waiting"; sleep 60; done
  local h=$(date +%H%M); if [ $h -ge 0340 ] && [ $h -le 0430 ]; then echo "in the restic window ($h): refusing to start"; exit 3; fi
}
build_bin2() { # TIMEOUT_S: mbuild until it succeeds (fix loop), then stage NEWBIN2
  until mbuild $1; do waitfix mbuild 900 || { echo "no binary: ending the window"; exit 1; }; done
  cp $MBIN $NEWBIN2; BIN=$NEWBIN2; sha256sum $BIN > $NEWBIN2.sha256
  echo "binary: $BIN sha256 $(cut -c1-64 $NEWBIN2.sha256) from $(git -C $SRC log --oneline -1 | cut -c1-60) $(git -C $SRC diff --quiet && echo clean || echo DIRTY)"
}
tmaps() { timeout 300 systemd-run --user --unit=pattn-tmap --collect --wait --pipe -q -p MemoryMax=4G $HOME/ai/convert-env/bin/python3 $G/tokmap.py spark "$@"; }
# --- pattn3 (per-model serving settings in the swap server) ---
NEWBIN3=$E/bin/mistralrs-titan-pattn3; P=$I/swapcli.py
build_bin3() { # TIMEOUT_S: mbuild until it succeeds (fix loop), then stage NEWBIN3
  until mbuild $1; do waitfix mbuild 900 || { echo "no binary: ending the window"; exit 1; }; done
  cp $MBIN $NEWBIN3; BIN=$NEWBIN3; sha256sum $BIN > $NEWBIN3.sha256
  echo "binary: $BIN sha256 $(cut -c1-64 $NEWBIN3.sha256) from $(git -C $SRC log --oneline -1 | cut -c1-60) $(git -C $SRC diff --quiet && echo clean || echo DIRTY)"
}
utest() { # TIMEOUT_S PKG FILTER [FEATURES]: cargo test --release (the build's profile and features) for one package's lib
  local L=$O/utest-$2-$WN.log t=$(date +%s)
  systemctl --user reset-failed pattn-build 2>/dev/null
  timeout $(( $1 + 30 )) systemd-run --user --unit=pattn-build --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 -p RuntimeMaxSec=$1 \
    -p WorkingDirectory=$SRC "${CARGO_ENV[@]}" -p StandardOutput=truncate:$L -p StandardError=truncate:$L \
    nice -n 10 cargo test --release -p $2 --features ${4:-oxide} ${UTARGET:---lib} -- $3
  local rc=$?; systemctl --user stop pattn-build 2>/dev/null
  echo "unit tests $2 [$3] rc=$rc in $(( $(date +%s) - t ))s ($(el)): $(grep -E '^test result' $L | tail -1)"
  grep -E "^test .* (ok|FAILED)$|^error|panicked" $L | head -30
  return $rc
}
cfg_start() { # NAME TOML: from-config, the deployed service's ExecStart / env / MemoryMax
  local n=$1 toml=$2
  SRV=$n; SLOG=$O/$n.server.log
  systemctl --user stop pattn-srv 2>/dev/null; systemctl --user reset-failed pattn-srv 2>/dev/null
  [ $(left) -gt 120 ] || { echo "!! $n skipped: $(left)s left"; return 1; }
  systemd-run --user --unit=pattn-srv --collect -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=$(( $(left) + 60 )) \
    -p KillSignal=SIGTERM -p TimeoutStopSec=15 -p WorkingDirectory=$HOME --setenv=LD_LIBRARY_PATH=$E/lib \
    -p StandardOutput=truncate:$SLOG -p StandardError=truncate:$SLOG $BIN from-config --file $toml
  local t0=$(date +%s)
  for i in $(seq 1 600); do
    curl -sf -m 2 -o /dev/null localhost:$PORT/v1/models && { echo "-- $n up in $(( $(date +%s) - t0 ))s ($(el)); GPU $(smi)"; return 0; }
    systemctl --user -q is-active pattn-srv || { echo "!! $n died"; sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -v " INFO " | tail -30; return 1; }
    sleep 1
  done
  echo "!! $n not up"; return 1
}
swaplog() { # LOG [FROM_LINE]: the swap's load / unload / memory lines since FROM_LINE
  sed 's/\x1b\[[0-9;]*m//g' $1 | tail -n +${2:-1} | grep -E "titan swap( mem)?[: ]|titan tiered auto|Layers [0-9]|PagedAttention KV cache|GPU blocks|max_num_seqs|panicked|OUT_OF_MEMORY|out of memory| ERROR " \
    | grep -v "registered" | sed -E 's/^[0-9T:.Z-]+ +(INFO|WARN|ERROR) [a-z_:0-9]*: /\1 /' | cut -c1-240
}
