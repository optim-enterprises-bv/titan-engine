#!/bin/bash
# mistral.rs (cuda) into target-fmt, memory-capped; log in ~/titan-engine/build-fmt.log
systemd-run --user --unit=fmt-build --collect --wait -q -p MemoryMax=9G -p MemorySwapMax=0 -p WorkingDirectory=$HOME/titan-engine/mistral.rs \
  --setenv=CARGO_TARGET_DIR=$HOME/titan-engine/target-fmt --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDA_HOME=/usr/local/cuda \
  --setenv=NVCC_CCBIN=/usr/bin/g++-15 --setenv=CUDA_NVCC_FLAGS=-Wno-template-body --setenv=CUDAFORGE_THREADS=4 --setenv=CARGO_BUILD_JOBS=4 \
  "--setenv=RUSTFLAGS=-L $HOME/titan-engine/lib" --setenv=PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:/usr/bin:/bin \
  /bin/sh -c "cargo build --release -p mistralrs-cli --features cuda > $HOME/titan-engine/build-fmt.log 2>&1; echo rc=\$? >> $HOME/titan-engine/build-fmt.log"
tail -3 $HOME/titan-engine/build-fmt.log
