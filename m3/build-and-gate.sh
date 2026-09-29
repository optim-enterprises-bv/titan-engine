#!/bin/bash
# wait for RAM, build mistral.rs (9G cap), wait again, run the M3 gate (10.5G cap)
wait_ram() { while [ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -lt $1 ]; do sleep 30; done; }
E=$HOME/titan-engine
wait_ram 13
echo "build start $(date)"
systemd-run --user --unit=titan-build --collect --wait -q -p MemoryMax=9G -p MemorySwapMax=0 -p WorkingDirectory=$E/mr-m3 --setenv=CARGO_TARGET_DIR=$E/mistral.rs/target \
  --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDA_HOME=/usr/local/cuda --setenv=NVCC_CCBIN=/usr/bin/g++-15 \
  --setenv=CUDA_NVCC_FLAGS=-Wno-template-body --setenv=CUDAFORGE_THREADS=4 --setenv=CARGO_BUILD_JOBS=4 \
  "--setenv=RUSTFLAGS=-L $E/lib" --setenv=PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:/usr/bin:/bin \
  -p StandardOutput=truncate:$E/build-titan.log -p StandardError=truncate:$E/build-titan.log \
  nice -n 15 ionice -c3 cargo build --release -p mistralrs-cli --features cuda
rc=$?; echo "build rc=$rc $(date)"; tail -2 $E/build-titan.log
[ $rc = 0 ] || exit 1
wait_ram 12
systemd-run --user --unit=titan-m3 --collect --wait -q -p MemoryMax=10500M -p MemorySwapMax=0 -p WorkingDirectory=$E/m3 \
  -p StandardOutput=truncate:$E/m3/gate-m3.log -p StandardError=truncate:$E/m3/gate-m3.log ./gate-m3.sh
echo "gate rc=$? $(date)"; grep -v INFO $E/m3/gate-m3.log
