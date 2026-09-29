trap 'systemctl --user start titan-mistral' EXIT
systemctl --user stop titan-mistral
# CUDA graphs window 4: rebuild da11fc3 (keep-old-scratch fix), gate (c) off/on, re-run (a).
# (was window 3): nvcc-free build of mr-graph (gate d), quick A/B, then gates on that binary:
# (a) 35B 40 x 256 graphs on vs m3/out/q35-prof.json, MTP=2 40 x 256 graphs on vs m6/out/h-off.json;
# (b) tok/s graphs off/on for MTP off and MTP=2 (same 40 x 256 runs); (c) ~13k-token prompt with graphs on.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/graph/window3.sh
trap 'exit 130' INT TERM HUP
set -u
E=$HOME/titan-engine; G=$E/m4/graph; M=$HOME/ai/models; F=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
exec > >(tee -a $G/window3-$(date +%Y%m%d-%H%M).log) 2>&1
T0=$(date +%s); DEADLINE=$((T0 + 57 * 60))
el() { echo "$(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
echo "== window3 start $(date -Is), titan-mistral: $(systemctl --user is-active titan-mistral)"
for p in $(ps -eo pid=,ppid=,comm= | awk '$2 == 878468 && $3 ~ /rust-analyzer/ {print $1}'); do echo "killing rust-analyzer $p"; kill $p; done
free -m | head -2
BIN=$E/target-graph-oxide/release/mistralrs
echo "== (d) nvcc-free build ($(el))"
L=$G/build-oxide3.log; t=$(date +%s)
timeout 1230 systemd-run --user --unit=graph-build-oxide --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 -p RuntimeMaxSec=1200 \
  -p WorkingDirectory=$E/mr-graph --setenv=CARGO_TARGET_DIR=$E/target-graph-oxide --setenv=TITAN_OXIDE_DIR=$E/oxide-kernels --setenv=CUDA_HOME=$E/nocuda-bin \
  --setenv=CUDA_PATH=$E/nocuda-bin --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDARC_CUDA_VERSION=13030 --setenv=CARGO_BUILD_JOBS=2 \
  --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 "--setenv=RUSTFLAGS=-L $E/lib" --setenv=PATH=$HOME/.cargo/bin:/usr/bin:/bin \
  -p StandardOutput=truncate:$L -p StandardError=truncate:$L nice -n 10 cargo build --release -p mistralrs-cli --features oxide
rc=$?; systemctl --user stop graph-build-oxide 2>/dev/null; grep -q "^error" $L && rc=1
echo "oxide build rc=$rc in $(( $(date +%s) - t ))s ($(el))"; grep "titan:" $L | sort -u | head -5; grep -E "^(error|warning: unused)|Finished" -A7 $L | grep -v half | head -40
ls -la $BIN; [ $rc = 0 ] && [ $BIN -nt $E/mr-graph/mistralrs-core/src/models/titan_graph.rs ] || { echo "no new binary: ending the window"; exit 1; }
command -v nvcc >/dev/null && echo "note: nvcc on this shell's PATH (the build's PATH had none)"


P=$E/m4/prompts-eval.txt
big() { # big NAME GRAPHS  -- service config, 3 back-to-back ~13k prompts
  local n=$1 g=$2
  [ $(left) -gt 420 ] || { echo "skip $n: $(left)s left"; return 1; }
  timeout 660 systemd-run --user --unit=graph-$n --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=640 bash -c "
    cd $G; env LD_LIBRARY_PATH=$E/lib TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt TITAN_MTP=2 TITAN_CUDA_GRAPHS=$g \
      $BIN serve -p 18491 --no-ui --paged-attn off --max-seq-len 65536 --format gguf -m $M/Qwen3.6-35B-A3B-MTP -f $F > out/$n.server.log 2>&1 &
    pid=\$!; for i in \$(seq 1 200); do curl -sf -m 2 -o /dev/null localhost:18491/v1/models && break; sleep 2; done
    for k in 1 2 3; do python3 $E/m4/bigprompt.py 18491 13000 || echo FAIL; done
    kill -TERM \$pid; sleep 5; kill -KILL \$pid 2>/dev/null; true"
  echo "-- $n ($(el))"; grep -o "titan graphs:.*\|ILLEGAL.*\|panicked.*" $G/out/$n.server.log | tail -3
}
echo "== (c) 3x ~13k back-to-back, graphs off then on ($(el))"
big big4-off 0
big big4-on 1
coll() {
  local n=$1 dir=$2; shift 2
  [ $(left) -gt 330 ] || { echo "skip $n: $(left)s left"; return 1; }
  timeout 600 systemd-run --user --unit=graph-$n --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=580 \
    --setenv=BIN=$BIN --setenv=DIR=$dir --setenv=FILE=$F \
    bash $E/m3/collect.sh $n $P 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt "$@" 2>&1 | grep -v " INFO "
  echo "-- $n ($(el))"
}
cmp() { python3 -c "import json,sys;a,b=json.load(open(sys.argv[1])),json.load(open(sys.argv[2]));print('GATE',sys.argv[2].split('/')[-1],'vs',sys.argv[1].split('/')[-1],':',sum(x==y for x,y in zip(a,b)),'/',len(a),'len',len(b))" $1 $2; }
echo "== (a) 35B 40 x 256, graphs on ($(el))"
coll g4-m1 $M/Qwen3.6-35B-A3B-MTP TITAN_CUDA_GRAPHS=1 TITAN_MTP=2; cmp $E/m6/out/h-off.json $E/m3/out/g4-m1.json
coll g4-a1 $M TITAN_CUDA_GRAPHS=1; cmp $E/m3/out/q35-prof.json $E/m3/out/g4-a1.json
free -m | head -2
echo "== window4 end $(el)"
