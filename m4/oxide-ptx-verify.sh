#!/bin/bash
cd "$(dirname "$0")"; E=$HOME/titan-engine
systemd-run --user --unit=titan-build --collect --wait -q -p MemoryMax=9G -p MemorySwapMax=0 -p WorkingDirectory=$E/mr-m3 --setenv=CARGO_TARGET_DIR=$E/mistral.rs/target \
  --setenv=TITAN_OXIDE_DIR=$E/oxide-kernels \
  --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDA_HOME=/usr/local/cuda --setenv=NVCC_CCBIN=/usr/bin/g++-15 --setenv=CUDA_NVCC_FLAGS=-Wno-template-body \
  --setenv=CUDAFORGE_THREADS=4 --setenv=CARGO_BUILD_JOBS=4 "--setenv=RUSTFLAGS=-L $E/lib" --setenv=PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:/usr/bin:/bin \
  -p StandardOutput=truncate:$E/build-titan.log -p StandardError=truncate:$E/build-titan.log nice -n 15 ionice -c3 cargo build --release -p mistralrs-cli --features cuda || { echo build failed; grep -E '^error' -A8 $E/build-titan.log | head -40; exit 1; }
grep -m1 "titan: all" $E/build-titan.log; echo build ok
cd $E/m3
systemd-run --user --unit=titan-ox24 --collect --wait -q -p MemoryMax=10500M -p MemorySwapMax=0 -p WorkingDirectory=$E/m3 -p StandardOutput=truncate:$E/m4/ox24.log -p StandardError=truncate:$E/m4/ox24.log \
  ./collect.sh ox-eval prompts-eval.txt 256 TITAN_TIERED=1 TITAN_TIERED_TRACE=/dev/null
grep -v INFO $E/m4/ox24.log | grep -E "tok/s|died"
python3 -c "import json;a,b=json.load(open('out/eval.json')),json.load(open('out/ox-eval.json'));print('24L all-GPU: nvcc vs oxide candle PTX:',sum(x==y for x,y in zip(a,b)),'/',len(a))"
systemd-run --user --unit=titan-ox35 --collect --wait -q -p MemoryMax=16G -p MemorySwapMax=0 -p WorkingDirectory=$E/m3 -p StandardOutput=truncate:$E/m4/ox35.log -p StandardError=truncate:$E/m4/ox35.log \
  env DIR=$HOME/ai/models FILE=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf ./collect.sh ox-q35 $E/m4/prompts-eval.txt 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt
grep -v INFO $E/m4/ox35.log | grep -E "tok/s|died"
python3 -c "import json;a,b=json.load(open('out/q35-prof.json')),json.load(open('out/ox-q35.json'));print('35B: nvcc vs oxide candle PTX:',sum(x==y for x,y in zip(a,b)),'/',len(a))"
