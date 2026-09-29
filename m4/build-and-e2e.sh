#!/bin/bash
wait_ram() { while [ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -lt $1 ]; do sleep 30; done; }
E=$HOME/titan-engine
wait_ram 11
systemd-run --user --unit=titan-build --collect --wait -q -p MemoryMax=9G -p MemorySwapMax=0 -p WorkingDirectory=$E/mr-m3 --setenv=CARGO_TARGET_DIR=$E/mistral.rs/target \
  --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDA_HOME=/usr/local/cuda --setenv=NVCC_CCBIN=/usr/bin/g++-15 \
  --setenv=CUDA_NVCC_FLAGS=-Wno-template-body --setenv=CUDAFORGE_THREADS=4 --setenv=CARGO_BUILD_JOBS=4 \
  "--setenv=RUSTFLAGS=-L $E/lib" --setenv=PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:/usr/bin:/bin \
  -p StandardOutput=truncate:$E/build-titan.log -p StandardError=truncate:$E/build-titan.log \
  nice -n 15 ionice -c3 cargo build --release -p mistralrs-cli --features cuda
rc=$?; echo "build rc=$rc $(date)"; grep -E "^error" -A6 $E/build-titan.log | head -30; [ $rc = 0 ] || exit 1
wait_ram 14
systemd-run --user --unit=titan-m4 --collect --wait -q -p MemoryMax=14G -p MemorySwapMax=0 -p WorkingDirectory=$E/m4 \
  -p StandardOutput=truncate:$E/m4/e2e.log -p StandardError=truncate:$E/m4/e2e.log ./e2e.sh "tiered55 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=0.55"
echo "e2e rc=$? $(date)"; grep -v INFO $E/m4/e2e.log | tail -30
