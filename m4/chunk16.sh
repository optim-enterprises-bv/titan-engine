#!/bin/bash
cd "$(dirname "$0")"; E=$HOME/titan-engine
systemd-run --user --unit=titan-build --collect --wait -q -p MemoryMax=9G -p MemorySwapMax=0 -p WorkingDirectory=$E/mr-m3 --setenv=CARGO_TARGET_DIR=$E/mistral.rs/target \
  --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDA_HOME=/usr/local/cuda --setenv=NVCC_CCBIN=/usr/bin/g++-15 --setenv=CUDA_NVCC_FLAGS=-Wno-template-body \
  --setenv=CUDAFORGE_THREADS=4 --setenv=CARGO_BUILD_JOBS=4 "--setenv=RUSTFLAGS=-L $E/lib" --setenv=PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:/usr/bin:/bin \
  -p StandardOutput=truncate:$E/build-titan.log -p StandardError=truncate:$E/build-titan.log nice -n 15 ionice -c3 cargo build --release -p mistralrs-cli --features cuda || { echo build failed; grep -E '^(error|warning: unused)' -A8 $E/build-titan.log | head -40; exit 1; }
echo build ok
cd $E/m3
n=q35-chunk16
systemd-run --user --unit=titan-auto --collect --wait -q -p MemoryMax=14G -p MemorySwapMax=0 -p WorkingDirectory=$E/m3 -p StandardOutput=truncate:$E/m4/$n.log -p StandardError=truncate:$E/m4/$n.log \
  env DIR=$HOME/ai/models FILE=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf ./collect.sh $n $E/m4/prompts-eval.txt 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt
grep -o 'titan tiered auto.*' out/$n.server.log | head -1
grep -v INFO $E/m4/$n.log | grep -E "tok/s|died" | head -2
grep -o 'decode hit rate.*' out/$n.server.log | tail -1
[ -f out/$n.json ] && python3 -c "import json;a,b=json.load(open('out/q35-prof.json')),json.load(open('out/$n.json'));print('identical to q35-prof:',sum(x==y for x,y in zip(a,b)),'/',len(a))" || grep -v " INFO " out/$n.server.log | grep -iE "panic|error" | head -3
