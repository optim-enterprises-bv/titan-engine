#!/bin/bash
E=$HOME/titan-engine
systemd-run --user --unit=titan-build --collect --wait -q -p MemoryMax=9G -p MemorySwapMax=0 -p WorkingDirectory=$E/mr-m3 --setenv=CARGO_TARGET_DIR=$E/mistral.rs/target \
  --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDA_HOME=/usr/local/cuda --setenv=NVCC_CCBIN=/usr/bin/g++-15 --setenv=CUDA_NVCC_FLAGS=-Wno-template-body \
  --setenv=CUDAFORGE_THREADS=4 --setenv=CARGO_BUILD_JOBS=4 "--setenv=RUSTFLAGS=-L $E/lib" --setenv=PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:/usr/bin:/bin \
  -p StandardOutput=truncate:$E/build-titan.log -p StandardError=truncate:$E/build-titan.log nice -n 15 ionice -c3 cargo build --release -p mistralrs-cli --features cuda || { echo nvcc build failed; exit 1; }
grep -c "served from cuda-oxide" $E/build-titan.log | sed 's/^/oxide PTX lines in nvcc build log (want 0): /'
cd $E/m4/prof
./prof.sh nvcc $E/mistral.rs/target/release/mistralrs
./prof.sh oxide $E/target-oxide/release/mistralrs
