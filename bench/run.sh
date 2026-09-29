#!/bin/bash
# titan-engine standing benchmark suite. See bench/README.md.
#
#   bench/run.sh <binary> [label]
#
# Needs the GPU to itself: run it inside a GPU window with titan-mistral stopped (bench/window.sh does both).
# Tiers: kernel (kbench vs llama.cpp, ~2 min), e2e + path (servers on port 18530, ~8 min), hygiene before/after.
# Env: BENCH_SRC   source tree of the binary (default ~/titan-engine/mr-094): its mistralrs-quant PTX feeds the
#                  kernel tier, its git rev is recorded as mistral.rs.
#      BENCH_SKIP  comma list of steps to skip (kernel,service,mtpoff,ident,nsys2,nsys0) - for debugging only;
#                  a run with skipped steps is still recorded, with the missing metrics absent.
set -u
E=$HOME/titan-engine; D=$E/bench; L=$D/lib
BIN=$(readlink -f "${1:?usage: run.sh <binary> [label]}"); LABEL=${2:-$(basename "$BIN")}
SRC=${BENCH_SRC:-$E/mr-094}; SKIP=",${BENCH_SKIP:-},"
NSYS=/opt/nvidia/nsight-compute/2026.2.1/host/target-linux-x64/nsys
PORT=18530
ID=$(date +%Y-%m-%d-%H%M)-$LABEL; R=$D/runs/$ID; mkdir -p $R
exec > >(tee -a $R/run.log) 2>&1
T0=$(date +%s); declare -A STEP
el() { echo "$(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s"; }
skip() { [[ $SKIP == *",$1,"* ]] && { echo "-- $1 skipped (BENCH_SKIP)"; return 0; }; return 1; }
echo "== bench $ID: $BIN (src $SRC) $(date -Is)"
[ -x "$BIN" ] || { echo "no such binary: $BIN"; exit 2; }
if [ "$(systemctl --user is-active titan-mistral)" = active ] && [ -z "${BENCH_FORCE:-}" ]; then
  echo "titan-mistral is running: the bench needs the GPU to itself (use bench/window.sh)"; exit 3
fi
for p in $(ps -eo pid=,ppid=,comm= | awk '$2 == 878468 && $3 ~ /rust-analyzer/ {print $1}'); do echo "killing rust-analyzer $p"; kill $p; done
: > $R/pids
BENCH_SRC=$SRC python3 $L/hygiene.py $R/hyg-before.json $R/pids

gpu_idle() { # wait (max 60 s) until the card is empty
  for i in $(seq 1 ${BENCH_GPU_WAIT:-30}); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 700 ] && return 0; sleep 2; done
  echo "warning: GPU memory still in use: $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ' ')"
}

# ---------------- kernel tier ----------------
if ! skip kernel; then
  t=$(date +%s); echo "== kernel tier ($(el))"
  K=$D/bin/kbench; FFI=$E/oxide-kernels/titan-oxide-ffi/target/release/libtitan_oxide_ffi.a
  if [ ! -x $K ] || [ $E/m4/kbench/kbench.cpp -nt $K ] || [ $FFI -nt $K ]; then
    timeout 300 systemd-run --user --collect --wait --pipe -q -p MemoryMax=4G -p MemorySwapMax=0 bash $D/kbuild.sh || echo "kbench build failed"
  fi
  mapfile -t CASES < <($K --list | grep -E '^q35\.(exp_gate_up\.q4_K|exp_down\.q5_K|exp_down\.q6_K|ffn_fused|qkv|attn_gate|ssm_out|shexp_gate_up|shexp_down|lm_head)\..*/b(1|3|8|512|2048)$')
  echo "${#CASES[@]} cases x 2 reps"
  gpu_idle
  timeout 600 systemd-run --user --collect --wait --pipe -q -p MemoryMax=10G -p MemorySwapMax=0 -p RuntimeMaxSec=590 \
    --setenv=CUDA_CACHE_MAXSIZE=4294967296 --setenv=GGML_CUDA_DISABLE_GRAPHS=1 --setenv=KBENCH_EXACT=1 --setenv=KBENCH_PTX_DIR=$SRC/mistralrs-quant/src/gguf \
    $NSYS profile -t cuda,nvtx --sample=none --cpuctxsw=none -o $R/kbench -f true \
    bash -c "for r in 1 2; do KBENCH_REP=\$r timeout 280 $K ${CASES[*]} > $R/kbench-r\$r.tsv 2> $R/kbench-r\$r.log; echo \"kbench rep \$r rc=\$? \$(grep -c ^RES $R/kbench-r\$r.tsv) rows\"; done" 2>&1 \
    | grep -v "^Collecting\|^Generating\|^Processing\|Generated:\|nsys-rep$\|^\s*$"
  timeout 300 $NSYS export --type sqlite -o $R/kbench.sqlite -f true $R/kbench.nsys-rep > /dev/null 2>&1
  timeout 300 python3 $L/kanalyze.py $R/kbench.sqlite $R $R/kernel.json | tee $R/kernel.txt | head -80
  STEP[kernel]=$(( $(date +%s) - t ))
fi

# ---------------- e2e + path tiers ----------------
mapfile -t SENV < <(python3 $L/svcconf.py env)
mapfile -t SARGS < <(python3 $L/svcconf.py args $PORT)
{ read -r MDIR; read -r MFILE; } < <(python3 $L/svcconf.py model)
OLDDIR=$HOME/ai/models   # the pre-MTP GGUF of the same name: m3/out/q35-prof.json's file
# serve NAME PLAN "EXTRA_ENV" [sed-expression-on-args] : one server under a 20G cap, one client plan, then TERM/KILL
serve() {
  local name=$1 plan=$2 extra=$3 sedx=${4:-} t=$(date +%s)
  skip $name && return
  gpu_idle
  local args; args=$(printf '%s\n' "${SARGS[@]}" | sed -e "$sedx" | tr '\n' ' ')
  local pre="" nsys_args=""
  if [[ $plan == nsys ]]; then pre="$NSYS launch --session-new=bench-$name-$$ -t cuda --cuda-graph-trace=node"; nsys_args="bench-$name-$$ $R/$name"; fi
  echo "== $name: plan $plan, $extra ($(el))"
  timeout 900 systemd-run --user --unit=bench-$name-$$ --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=880 \
    bash -c "
    cd $R; env ${SENV[*]} $extra $pre $BIN --seed 0 $args > $R/$name.server.log 2>&1 &
    pid=\$!; echo \$pid >> $R/pids; t0=\$(date +%s)
    for i in \$(seq 1 300); do curl -sf -m 2 -o /dev/null localhost:$PORT/v1/models && break; kill -0 \$pid 2>/dev/null || break; sleep 1; done
    echo \"$name: server up after \$(( \$(date +%s) - t0 ))s\"
    timeout 700 python3 $L/client.py $plan $PORT $R/e2e-$name.json $nsys_args || echo \"$name: CLIENT FAILED rc=\$?\"
    kill -TERM \$pid; for i in \$(seq 1 20); do kill -0 \$pid 2>/dev/null || break; sleep 0.5; done; kill -KILL \$pid 2>/dev/null; true"
  grep -E "panicked|ILLEGAL|out of memory|OUT_OF_MEMORY|CUDA_ERROR" $R/$name.server.log | head -3 | cut -c1-240
  grep -o "titan tiered auto.*" $R/$name.server.log | head -1 | cut -c1-200
  grep -o "titan mtp stats.*" $R/$name.server.log | tail -1 | cut -c1-200
  STEP[$name]=$(( $(date +%s) - t ))
}
serve service service ""
serve mtpoff  mtpoff  "TITAN_MTP=0"
serve ident   ident   "TITAN_MTP=0" "s#^$MDIR\$#$OLDDIR#;s#^65536\$#4096#"
serve nsys2   nsys    ""
serve nsys0   nsys    "TITAN_MTP=0"
for n in nsys2 nsys0; do
  [ -f $R/$n.nsys-rep ] || continue
  t=$(date +%s)
  timeout 300 $NSYS export --type sqlite -o $R/$n.sqlite -f true $R/$n.nsys-rep > /dev/null 2>&1
  out=$R/nsys-$([ $n = nsys2 ] && echo mtp2 || echo off).json
  timeout 300 python3 $L/nsys_split.py $R/$n.sqlite $out
  STEP[$n]=$(( ${STEP[$n]:-0} + $(date +%s) - t ))
done

BENCH_SRC=$SRC python3 $L/hygiene.py $R/hyg-after.json $R/pids
steps=$(for k in "${!STEP[@]}"; do printf '"%s": %s,' $k ${STEP[$k]}; done)
python3 - <<PY
import json, hashlib
h = hashlib.sha256()
with open("$BIN", "rb") as f:
    for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
json.dump({"id": "$ID", "date": "$(date -Is -d @$T0)", "label": "$LABEL", "binary": "$BIN", "binary_sha256": h.hexdigest(),
           "src": "$SRC", "duration_s": $(( $(date +%s) - T0 )), "steps_s": {${steps%,}}}, open("$R/meta.json", "w"), indent=1)
PY
rm -f $R/*.qdstrm
python3 $L/report.py $R
echo "== bench $ID done in $(el): report bench/runs/$ID.md"
