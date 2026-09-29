#!/bin/bash
# Build mistral.rs with every CUDA kernel from cuda-oxide and nvcc unreachable:
# CUDA_HOME is a stub with only include/ and lib64/ (no bin/), /usr/local/cuda/bin is not on PATH.
E=$HOME/titan-engine
command -v nvcc && { echo "nvcc still reachable on PATH"; exit 1; }
t0=$(date +%s)
systemd-run --user --unit=titan-nonvcc --collect --wait -q -p MemoryMax=9G -p MemorySwapMax=0 -p WorkingDirectory=$E/mr-m3 \
  --setenv=CARGO_TARGET_DIR=$E/target-oxide --setenv=TITAN_OXIDE_DIR=$E/oxide-kernels --setenv=CUDA_HOME=$E/nocuda-bin --setenv=CUDA_PATH=$E/nocuda-bin \
  --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDARC_CUDA_VERSION=13030 --setenv=CARGO_BUILD_JOBS=4 "--setenv=RUSTFLAGS=-L $E/lib" --setenv=PATH=$HOME/.cargo/bin:/usr/bin:/bin \
  -p StandardOutput=truncate:$E/build-nonvcc.log -p StandardError=truncate:$E/build-nonvcc.log nice -n 15 ionice -c3 cargo build --release -p mistralrs-cli --features oxide
rc=$?; echo "build rc=$rc in $(( $(date +%s) - t0 ))s"
grep "titan:" $E/build-nonvcc.log | sort -u
grep -iE "nvcc" $E/build-nonvcc.log | grep -v "nvcc not used\|titan:" | head -5
[ $rc = 0 ] || { grep -E "^error" -A10 $E/build-nonvcc.log | head -40; exit 1; }
ls -la $E/target-oxide/release/mistralrs
