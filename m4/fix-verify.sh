#!/bin/bash
cd "$(dirname "$0")"; E=$HOME/titan-engine
wait_ram() { while [ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -lt $1 ]; do sleep 30; done; }
wait_ram 11
systemd-run --user --unit=titan-build --collect --wait -q -p MemoryMax=9G -p MemorySwapMax=0 -p WorkingDirectory=$E/mr-m3 --setenv=CARGO_TARGET_DIR=$E/mistral.rs/target \
  --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDA_HOME=/usr/local/cuda --setenv=NVCC_CCBIN=/usr/bin/g++-15 --setenv=CUDA_NVCC_FLAGS=-Wno-template-body \
  --setenv=CUDAFORGE_THREADS=4 --setenv=CARGO_BUILD_JOBS=4 "--setenv=RUSTFLAGS=-L $E/lib" --setenv=PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:/usr/bin:/bin \
  -p StandardOutput=truncate:$E/build-titan.log -p StandardError=truncate:$E/build-titan.log nice -n 15 ionice -c3 cargo build --release -p mistralrs-cli --features cuda || { echo build failed; grep -E '^error' -A6 $E/build-titan.log | head -30; exit 1; }
echo build ok
wait_ram 13
systemd-run --user --unit=titan-m4-det --collect --wait -q -p MemoryMax=14G -p MemorySwapMax=0 -p WorkingDirectory=$E/m4 -p StandardOutput=truncate:$E/m4/det.out -p StandardError=truncate:$E/m4/det.out ./determinism.sh
grep -v INFO det.out | grep -E "vs|tok/s"
python3 compare.py llama det1 | tail -1; python3 - <<'PY'
import json
a,b=json.load(open("out/llama.json")),json.load(open("out/det1.json"))
for i,(x,y) in enumerate(zip(a,b)):
    x,y=x.lstrip(),y.lstrip(); k=next((j for j in range(min(len(x),len(y))) if x[j]!=y[j]),min(len(x),len(y)))
    print(i,"identical" if x==y else f"same for {k}/{len(x)} chars")
PY
