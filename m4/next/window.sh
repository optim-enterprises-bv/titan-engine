trap 'systemctl --user start titan-mistral' EXIT
systemctl --user stop titan-mistral
# M4 qwen3next window. Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/next/window.sh [steps]
# steps (default "build l4 q35 oxide next"): build = nvcc build of mr-next; l4 = 4-layer cut of the 80B,
# llama.cpp vs mistral.rs (stock / tiered owned / tiered mmap); q35 = 40-prompt 35B baseline (gate 2);
# oxide = nvcc-free build (gate 3); next = the 80B, llama.cpp vs mistral.rs tiered + mmap (gate 1).
trap 'exit 130' INT TERM HUP
set -u
E=$HOME/titan-engine; N=$E/m4/next; M=$HOME/ai/models
STEPS=${1:-build l4 q35 oxide next}
LOG=$N/logs/window-$(date +%Y%m%d-%H%M).log
exec > >(tee -a $LOG) 2>&1
T0=$(date +%s); DEADLINE=$((T0 + 57 * 60))
el() { echo "$(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
has() { [[ " $STEPS " == *" $1 "* ]]; }
echo "== window start $(date -Is), titan-mistral: $(systemctl --user is-active titan-mistral), steps: $STEPS"
free -m | head -2
for p in $(ps -eo pid=,ppid=,comm= | awk '$2 == 878468 && $3 ~ /rust-analyzer/ {print $1}'); do echo "killing rust-analyzer $p"; kill $p; done

if has build; then
  echo "== build ($(el))"
  L=$N/logs/build-next.log; t=$(date +%s)
  timeout 1530 systemd-run --user --unit=next-build-cuda --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 \
    -p RuntimeMaxSec=1500 -p WorkingDirectory=$E/mr-next \
    --setenv=CARGO_BUILD_JOBS=2 --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 --setenv=CUDA_COMPUTE_CAP=120 \
    --setenv=CUDA_HOME=/usr/local/cuda --setenv=NVCC_CCBIN=/usr/bin/g++-15 --setenv=CUDA_NVCC_FLAGS=-Wno-template-body \
    "--setenv=RUSTFLAGS=-L $E/lib" --setenv=CARGO_TARGET_DIR=$E/target-next \
    --setenv=PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:/usr/bin:/bin \
    -p StandardOutput=truncate:$L -p StandardError=truncate:$L nice -n 10 cargo build --release -p mistralrs-cli --features cuda
  rc=$?; systemctl --user stop next-build-cuda 2>/dev/null; grep -q "^error" $L && rc=1
  echo "build rc=$rc in $(( $(date +%s) - t ))s"; grep -E "^(warning: unused|error)|Finished" -A7 $L | head -60
  [ $rc = 0 ] || { echo "build failed: ending the window"; exit 1; }
  ls -la $E/target-next/release/mistralrs
fi

if has l4 && [ $(left) -gt 900 ]; then
  echo "== l4: 4-layer qwen3next cut ($(el))"
  timeout 1200 $N/pair.sh $M/qwen3next-L4.gguf 2 llama-l4 "l4-stock" \
    "l4-owned TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=0.25" \
    "l4-mmap TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=0.25 TITAN_TIERED_MMAP=1"
  cd $N; python3 cmp.py llama-l4 l4-stock | tail -20; python3 cmp.py l4-stock l4-owned | tail -1; python3 cmp.py l4-stock l4-mmap | tail -1
  python3 cmp.py l4-stock l4-mmap | grep argmax
fi

if has q35 && [ $(left) -gt 700 ]; then
  echo "== q35: gate 2, 40 prompts x 256 on the 35B ($(el))"
  for spec in "q35-next" "q35-next-mmap TITAN_TIERED_MMAP=1"; do
    [ $(left) -gt 600 ] || { echo "skip $spec: $(left)s left"; continue; }
    set -- $spec; n=$1; shift
    timeout 900 systemd-run --user --unit=next-q35 --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 \
      --setenv=BIN=$E/target-next/release/mistralrs --setenv=DIR=$M --setenv=FILE=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
      bash $E/m3/collect.sh $n $E/m4/prompts-eval.txt 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto \
      TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt "$@" 2>&1 | grep -v " INFO "
    (cd $E/m3 && [ -f out/$n.json ] && python3 -c "import json;a,b=json.load(open('out/q35-prof.json')),json.load(open('out/$n.json'));print('$n identical to q35-prof:',sum(x==y for x,y in zip(a,b)),'/',len(a))")
    grep -o 'titan tiered auto.*' $E/m3/out/$n.server.log | head -1; grep -o 'decode hit rate.*' $E/m3/out/$n.server.log | tail -1
    grep -m1 -o 'titan tiered experts: .*' $E/m3/out/$n.server.log
  done
fi

if has oxide && [ $(left) -gt 600 ]; then
  echo "== oxide: gate 3, nvcc-free build ($(el))"
  PATH=$HOME/.cargo/bin:/usr/bin:/bin bash $N/nonvcc-build.sh $(( $(left) - 120 < 1500 ? $(left) - 120 : 1500 ))
fi

F80=$M/qwen3-next-80b/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf
if has next && [ -f $F80 ] && [ $(left) -gt 1200 ]; then
  echo "== next: gate 1, the 80B ($(el)), $(ls -la $F80)"
  NCM=${NCM:-37}
  timeout $(( $(left) - 60 )) $N/pair.sh $F80 $NCM llama-next \
    "next-mmap TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_MMAP=1"
  cd $N; python3 cmp.py llama-next next-mmap
fi
echo "== window end $(el)"
