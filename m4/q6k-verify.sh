#!/bin/bash
cd "$(dirname "$0")"; E=$HOME/titan-engine
wait_ram() { while [ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -lt $1 ]; do sleep 30; done; }
wait_ram 11
systemd-run --user --unit=titan-build --collect --wait -q -p MemoryMax=9G -p MemorySwapMax=0 -p WorkingDirectory=$E/mr-m3 --setenv=CARGO_TARGET_DIR=$E/mistral.rs/target \
  --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDA_HOME=/usr/local/cuda --setenv=NVCC_CCBIN=/usr/bin/g++-15 --setenv=CUDA_NVCC_FLAGS=-Wno-template-body \
  --setenv=CUDAFORGE_THREADS=4 --setenv=CARGO_BUILD_JOBS=4 "--setenv=RUSTFLAGS=-L $E/lib" --setenv=PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:/usr/bin:/bin \
  -p StandardOutput=truncate:$E/build-titan.log -p StandardError=truncate:$E/build-titan.log nice -n 15 ionice -c3 cargo build --release -p mistralrs-cli --features cuda || { echo build failed; grep -E '^error' -A8 $E/build-titan.log | head -40; exit 1; }
echo build ok
cd $E/m3
wait_ram 13
systemd-run --user --unit=titan-v24 --collect --wait -q -p MemoryMax=10500M -p MemorySwapMax=0 -p WorkingDirectory=$E/m3 -p StandardOutput=truncate:$E/m4/v24.log -p StandardError=truncate:$E/m4/v24.log \
  ./collect.sh prof50-v3 prompts-eval.txt 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=0.5 TITAN_TIERED_PROFILE=$E/m3/profile-calib.txt
grep -v INFO $E/m4/v24.log | grep tok/s
python3 -c "import json;a,b=json.load(open('out/eval.json')),json.load(open('out/prof50-v3.json'));print('24L eval vs prof50-v3:',sum(x==y for x,y in zip(a,b)),'/',len(a))"
for fr in 0.62 0.68 0.74 0.80; do
  wait_ram 13
  n=q35-v3-$fr
  systemd-run --user --unit=titan-v35 --collect --wait -q -p MemoryMax=14G -p MemorySwapMax=0 -p WorkingDirectory=$E/m3 -p StandardOutput=truncate:$E/m4/$n.log -p StandardError=truncate:$E/m4/$n.log \
    env DIR=$HOME/ai/models FILE=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf ./collect.sh $n $E/m4/prompts-eval.txt 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=$fr TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt
  echo "fraction $fr:"; grep -v INFO $E/m4/$n.log | grep -E "tok/s|died" | head -2
  grep -o 'decode hit rate.*' out/$n.server.log | tail -1
  [ -f out/$n.json ] && python3 -c "import json;a,b=json.load(open('out/q35-prof.json')),json.load(open('out/$n.json'));print('  identical to q35-prof:',sum(x==y for x,y in zip(a,b)),'/',len(a))" || { grep -v " INFO " out/$n.server.log | grep -iE "panic|error" | head -3; break; }
done
