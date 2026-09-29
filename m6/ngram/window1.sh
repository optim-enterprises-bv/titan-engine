# 30-minute service gap (RULES-agents.md, 2026-09-28 15:10): after the flock, before stopping the service
while :; do t=$(systemctl --user show titan-mistral -p ActiveEnterTimestampMonotonic --value)
  now=$(awk '{print int($1*1000000)}' /proc/uptime); [ $(( (now - t) / 1000000 )) -ge 1800 ] && break; sleep 30; done
trap 'systemctl --user start titan-mistral' EXIT
systemctl --user stop titan-mistral
echo "== service stopped $(date -Is)"
# n-gram (prompt-lookup) drafting on titan-094, window 1: nvcc-free build of mr-ngram94 (gate 4), then
# gate 1 (byte identity): MTP=2 + TITAN_NGRAM=1 40 x 256 vs m6/out/h-off.json; no MTP + TITAN_NGRAM=1 vs m3/out/q35-prof.json;
# gate 2 (speed, 40 prompts): MTP=2 ngram off vs on, same binary; gate 3: code-editing bench off / on (policies).
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m6/ngram/window1.sh
trap 'exit 130' INT TERM HUP
set -u
E=$HOME/titan-engine; N=$E/m6/ngram; M=$HOME/ai/models; F=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
mkdir -p $N/out
exec > >(tee -a $N/window1-$(date +%Y%m%d-%H%M).log) 2>&1
T0=$(date +%s); DEADLINE=$((T0 + 57 * 60))
el() { echo "$(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
echo "== ngram window1 start $(date -Is), titan-mistral: $(systemctl --user is-active titan-mistral)"
for p in $(ps -eo pid=,ppid=,comm= | awk '$2 == 878468 && $3 ~ /rust-analyzer/ {print $1}'); do echo "killing rust-analyzer $p"; kill $p; done
free -m | head -2
BIN=$E/target-ngram94-oxide/release/mistralrs
echo "== gate 4: nvcc-free build ($(el))"
L=$N/build-oxide1.log; t=$(date +%s)
timeout 1530 systemd-run --user --unit=ngram-build-oxide --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 -p RuntimeMaxSec=1500 \
  -p WorkingDirectory=$E/mr-ngram94 --setenv=CARGO_TARGET_DIR=$E/target-ngram94-oxide --setenv=TITAN_OXIDE_DIR=$E/oxide-kernels --setenv=CUDA_HOME=$E/nocuda-bin \
  --setenv=CUDA_PATH=$E/nocuda-bin --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDARC_CUDA_VERSION=13030 --setenv=CARGO_BUILD_JOBS=2 \
  --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 "--setenv=RUSTFLAGS=-L $E/lib" --setenv=PATH=$HOME/.cargo/bin:/usr/bin:/bin \
  -p StandardOutput=truncate:$L -p StandardError=truncate:$L nice -n 10 cargo build --release -p mistralrs-cli --features oxide
rc=$?; systemctl --user stop ngram-build-oxide 2>/dev/null; grep -q "^error" $L && rc=1
echo "oxide build rc=$rc in $(( $(date +%s) - t ))s ($(el))"; grep -E "^(error|warning)|Finished" -A7 $L | grep -v half | head -60
ls -la $BIN; [ $rc = 0 ] && [ $BIN -nt $E/mr-ngram94/mistralrs-core/src/speculative/ngram.rs ] || { echo "no new binary: ending the window"; exit 1; }
ldd $BIN | grep -i "not found\|cuda\|nvrtc\|cublas" | head

P=$E/m4/prompts-eval.txt
coll() {
  local n=$1 dir=$2; shift 2
  [ $(left) -gt 330 ] || { echo "skip $n: $(left)s left"; return 1; }
  timeout 600 systemd-run --user --unit=ngram-$n --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=580 \
    --setenv=BIN=$BIN --setenv=DIR=$dir --setenv=FILE=$F \
    bash $E/sync/w094/collect094.sh $n $P 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt "$@" 2>&1 | grep -v " INFO "
  echo "-- $n ($(el))"; grep -o "titan ngram stats.*\|titan mtp stats.*\|panicked.*" $E/m3/out/$n.server.log | tail -2
  cp $E/m3/out/$n.json $E/m3/out/$n.server.log $N/out/ 2>/dev/null
}
cmp() { python3 -c "import json,sys;a,b=json.load(open(sys.argv[1])),json.load(open(sys.argv[2]));print('GATE',sys.argv[2].split('/')[-1],'vs',sys.argv[1].split('/')[-1],':',sum(x==y for x,y in zip(a,b)),'/',len(a),'len',len(b))" $1 $2; }
code() { # code NAME ENV...  -- service config (MTP=2, tiered, 64k, paged off), the 20-prompt code-editing bench
  local n=$1; shift
  [ $(left) -gt 780 ] || { echo "skip $n: $(left)s left"; return 1; }
  timeout 900 systemd-run --user --unit=ngram-$n --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=880 bash -c "
    cd $N; env LD_LIBRARY_PATH=$E/lib TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt TITAN_MTP=2 $* \
      $BIN --seed 0 serve -p 18493 --no-ui --paged-attn off --max-seq-len 65536 --format gguf -m $M/Qwen3.6-35B-A3B-MTP -f $F > out/$n.server.log 2>&1 &
    pid=\$!; for i in \$(seq 1 200); do curl -sf -m 2 -o /dev/null localhost:18493/v1/models && break; kill -0 \$pid 2>/dev/null || break; sleep 2; done
    python3 $N/run_code_bench.py 18493 $N/code-bench.json out/$n || echo FAIL
    kill -TERM \$pid; sleep 5; kill -KILL \$pid 2>/dev/null; true"
  echo "-- $n ($(el))"; grep -o "titan ngram stats.*\|titan mtp stats.*\|titan ngram: .*\|panicked.*\|out of memory.*" $N/out/$n.server.log | tail -4
}
echo "== gate 1a: MTP=2 + ngram, 40 x 256 vs h-off ($(el))"
coll ng1-m2on $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2 TITAN_NGRAM=1; cmp $E/m6/out/h-off.json $E/m3/out/ng1-m2on.json
echo "== gate 2: MTP=2, ngram off, same binary ($(el))"
coll ng1-m2off $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2; cmp $E/m6/out/h-off.json $E/m3/out/ng1-m2off.json
echo "== gate 3: code bench off / on ($(el))"
code cb1-off
code cb1-first TITAN_NGRAM=1
python3 -c "import json;a,b=json.load(open('$N/out/cb1-off.json')),json.load(open('$N/out/cb1-first.json'));print('CODE identity first vs off:',sum(x==y for x,y in zip(a,b)),'/',len(a))"
echo "== gate 1b: no MTP + ngram, 40 x 256 vs q35-prof ($(el))"
coll ng1-a1on $M TITAN_NGRAM=1; cmp $E/m3/out/q35-prof.json $E/m3/out/ng1-a1on.json
code cb1-agree TITAN_NGRAM=1 TITAN_NGRAM_POLICY=agree
python3 -c "import json;a,b=json.load(open('$N/out/cb1-off.json')),json.load(open('$N/out/cb1-agree.json'));print('CODE identity agree vs off:',sum(x==y for x,y in zip(a,b)),'/',len(a))"
free -m | head -2
echo "== ngram window1 end $(el)"
