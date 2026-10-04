#!/bin/bash
# Build-test the public release export (release/public-update, commit 5b84d1c) from a clean scratch copy, then the
# 35B gate pair and an OrcaSAQ (dense IQ4_XS) smoke. Live service titan-mistral: stopped here, restarted on EXIT.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/pubtest/window.sh
set -u
E=$HOME/titan-engine; P=$E/m4/pubtest; O=$P/out; mkdir -p $O
S=$E/pubtest-src; TGT=$E/target-pubtest-oxide; BIN=$TGT/release/mistralrs; M=$HOME/ai/models; PORT=18650
exec > >(tee -a $P/window-$(date +%Y%m%d-%H%M).log) 2>&1
T0=$(date +%s); el() { echo "$(( ($(date +%s)-T0)/60 ))m$(( ($(date +%s)-T0)%60 ))s"; }
MIS=$(systemctl --user is-active titan-mistral)
trap 'systemctl --user stop pubtest-build pubtest-srv pubtest-coll 2>/dev/null; [ "$MIS" = active ] && systemctl --user start titan-mistral; echo "== exit $(date -Is) ($(el)): titan-mistral $(systemctl --user is-active titan-mistral)"' EXIT
trap 'exit 130' INT TERM HUP
systemctl --user stop titan-mistral
echo "== start $(date -Is), export commit $(git -C $E/release/public-update log --oneline -1)"
rm -rf $S; mkdir -p $S; ( cd $E/release/public-update && git ls-files -z | xargs -0 cp -a --reflink=auto --parents -t $S )
echo "copied $(find $S -type f | wc -l) files; LFS ptx size $(stat -c %s $S/oxide-kernels/mistralrs-quant-c/mistralrs_quant_c.ptx)"
[ -d $TGT ] || cp -a --reflink=auto $E/target-orca-oxide $TGT
L=$O/build.log; t=$(date +%s)
timeout 2430 systemd-run --user --unit=pubtest-build --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 -p RuntimeMaxSec=2400 \
  -p WorkingDirectory=$S/mistral.rs --setenv=CARGO_TARGET_DIR=$TGT --setenv=TITAN_OXIDE_DIR=$S/oxide-kernels --setenv=CUDA_HOME=$E/nocuda-bin \
  --setenv=CUDA_PATH=$E/nocuda-bin --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDARC_CUDA_VERSION=13030 --setenv=CARGO_BUILD_JOBS=2 \
  --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 "--setenv=RUSTFLAGS=-L $E/lib" --setenv=PATH=$HOME/.cargo/bin:/usr/bin:/bin \
  -p StandardOutput=truncate:$L -p StandardError=truncate:$L nice -n 10 cargo build --release -p mistralrs-cli --features oxide
rc=$?; grep -q "^error" $L && rc=1; echo "BUILD rc=$rc in $(( $(date +%s)-t ))s ($(el))"; grep -E "^error" -A12 $L | head -60; grep Finished $L
[ $rc = 0 ] && [ -x $BIN ] && [ $BIN -nt $S/mistral.rs/Cargo.toml ] || { echo "BUILD FAILED"; exit 1; }
coll() { local n=$1; shift
  timeout 600 systemd-run --user --unit=pubtest-coll --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=580 \
    --setenv=BIN=$BIN --setenv=DIR=$M/Qwen3.6-35B-A3B-MTP --setenv=FILE=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    bash $E/sync/w094/collect094.sh $n $E/m4/prompts-eval.txt 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt \
      TITAN_PFS=1 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64 TITAN_TIERED_RESERVE_MIB=1536 "$@" 2>&1 | grep -v " INFO " | tail -3; }
cmp40() { python3 -c "import json,sys;a,b=json.load(open(sys.argv[1])),json.load(open(sys.argv[2]));print('GATE',sys.argv[2].split('/')[-1],'vs',sys.argv[1].split('/')[-1],':',sum(x==y for x,y in zip(a,b)),'/',len(a))" $1 $2; }
echo "== 35B gate pair ($(el))"
coll pubtest-m2 TITAN_MTP=2 && cmp40 $E/m6/out/h-off-noblas.json $E/m3/out/pubtest-m2.json
coll pubtest-mf0 TITAN_MTP=0 && cmp40 $E/m6/out/mtpfile-off-noblas.json $E/m3/out/pubtest-mf0.json
echo "== OrcaSAQ dense IQ4_XS smoke ($(el))"
systemd-run --user --unit=pubtest-srv --collect -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=600 --setenv=LD_LIBRARY_PATH=$E/lib \
  --setenv=TITAN_MTP=0 --setenv=TITAN_PREFILL_BIG_CHUNK=0 -p StandardOutput=truncate:$O/orca.server.log -p StandardError=truncate:$O/orca.server.log \
  $BIN --seed 0 serve -p $PORT --no-ui --paged-attn off --format gguf -m $M/orcasaq2-cyber-27b -f OrcaSAQ-2-27B-Uncensored.gguf --max-seq-len 8192
for i in $(seq 1 300); do curl -sf -m 2 -o /dev/null localhost:$PORT/v1/models && break; sleep 1; done
curl -s -m 300 localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"default","max_tokens":60,"temperature":0,"messages":[{"role":"user","content":"In one sentence: what is the capital of France?"}]}' | python3 -c "import sys,json; r=json.load(sys.stdin); m=r['choices'][0]['message']; print('ORCA reply:', ((m.get('content') or '')+' '+(m.get('reasoning_content') or ''))[:120].replace(chr(10),' '))" || echo "ORCA smoke FAILED"
systemctl --user stop pubtest-srv; grep -cE "panicked|CUDA_ERROR" $O/orca.server.log | sed 's/^/orca panics\/cuda errors: /'
echo "== Spark paged ON, 4 concurrent ($(el))"
systemd-run --user --unit=pubtest-srv --collect -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=600 --setenv=LD_LIBRARY_PATH=$E/lib \
  -p StandardOutput=truncate:$O/spark.server.log -p StandardError=truncate:$O/spark.server.log \
  $BIN --seed 0 serve -p $PORT --no-ui --paged-attn on --max-seqs 4 --format gguf -m $M/spark-x2.5 -f Spark-X2.5-4B-Q4_K_M.gguf --max-seq-len 16384
for i in $(seq 1 300); do curl -sf -m 2 -o /dev/null localhost:$PORT/v1/models && break; sleep 1; done
python3 - <<PY
import json,urllib.request,concurrent.futures as cf
def ask(k):
    b=json.dumps({"model":"default","max_tokens":64,"temperature":0,"messages":[{"role":"user","content":f"In one sentence: what is {k+2} times 3?"}]}).encode()
    r=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:$PORT/v1/chat/completions",b,{"Content-Type":"application/json"}),timeout=300))
    return r["usage"]["completion_tokens"]
with cf.ThreadPoolExecutor(4) as ex: print("SPARK paged x4 ok, tokens:", list(ex.map(ask,range(4))))
PY
systemctl --user stop pubtest-srv; grep -cE "panicked|CUDA_ERROR" $O/spark.server.log | sed 's/^/spark panics\/cuda errors: /'
echo "== done ($(el))"
