#!/bin/bash
# ctest.sh <test filter...>: candle-core lib tests (release, cuda) in target-fmt-candle, memory-capped.
cd $HOME/titan-engine/candle
systemd-run --user --unit=fmt-ctest --collect --wait -q -p MemoryMax=9G -p MemorySwapMax=0 -p WorkingDirectory=$PWD \
  --setenv=CARGO_TARGET_DIR=$HOME/titan-engine/target-fmt-candle --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDA_HOME=/usr/local/cuda \
  --setenv=NVCC_CCBIN=/usr/bin/g++-15 --setenv=CUDA_NVCC_FLAGS=-Wno-template-body --setenv=CARGO_BUILD_JOBS=4 \
  --setenv=PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:/usr/bin:/bin --setenv=LD_LIBRARY_PATH=$HOME/titan-engine/lib:/usr/local/cuda/lib64 "--setenv=RUSTFLAGS=-C target-cpu=native -L $HOME/titan-engine/lib" \
  /bin/sh -c "cargo test --release -p candle-core --features cuda --lib -- $* > $HOME/titan-engine/oxide-kernels/iq4_nl/ctest.log 2>&1; echo rc=\$? >> $HOME/titan-engine/oxide-kernels/iq4_nl/ctest.log"
grep -E "^test |test result|error|panicked|rc=" $HOME/titan-engine/oxide-kernels/iq4_nl/ctest.log | head -40
