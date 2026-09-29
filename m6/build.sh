#!/bin/bash
# M6: build mistralrs-cli (cuda) into target-mtp
E=$HOME/titan-engine
systemd-run --user --unit=mtp-build --collect --wait -q -p MemoryMax=9G -p MemorySwapMax=0 -p WorkingDirectory=$E/${SRC:-mr-m6} \
  --setenv=CARGO_TARGET_DIR=$E/target-mtp --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDA_HOME=/usr/local/cuda \
  --setenv=NVCC_CCBIN=/usr/bin/g++-15 --setenv=CUDA_NVCC_FLAGS=-Wno-template-body --setenv=CUDAFORGE_THREADS=4 \
  --setenv=CARGO_BUILD_JOBS=4 "--setenv=RUSTFLAGS=-L $E/lib" --setenv=PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:/usr/bin:/bin \
  -p StandardOutput=truncate:$E/m6/build.log -p StandardError=truncate:$E/m6/build.log \
  nice -n 10 cargo build --release -p mistralrs-cli --features cuda
rc=$?
grep -E '^(error|warning: unused)' -A12 $E/m6/build.log | head -120
echo "build rc=$rc"
