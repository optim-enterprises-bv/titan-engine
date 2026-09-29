#!/bin/bash
# nvcc-free build of mr-oss (--features oxide): CUDA_HOME is a stub without bin/, nvcc is not on PATH.
# usage: nonvcc-build.sh MAX_SECONDS
E=$HOME/titan-engine; L=$E/m5/oss/logs/build-oss-oxide.log
command -v nvcc && { echo "nvcc still reachable on PATH"; exit 1; }
t0=$(date +%s)
timeout $(( ${1:-1500} + 30 )) systemd-run --user --unit=oss-build-oxide --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 \
  -p RuntimeMaxSec=${1:-1500} -p WorkingDirectory=$E/mr-oss \
  --setenv=CARGO_TARGET_DIR=$E/target-oss-oxide --setenv=TITAN_OXIDE_DIR=$E/oxide-kernels --setenv=CUDA_HOME=$E/nocuda-bin \
  --setenv=CUDA_PATH=$E/nocuda-bin --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDARC_CUDA_VERSION=13030 --setenv=CARGO_BUILD_JOBS=2 \
  --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 "--setenv=RUSTFLAGS=-L $E/lib" --setenv=PATH=$HOME/.cargo/bin:/usr/bin:/bin \
  -p StandardOutput=truncate:$L -p StandardError=truncate:$L nice -n 10 cargo build --release -p mistralrs-cli --features oxide
rc=$?; systemctl --user stop oss-build-oxide 2>/dev/null; grep -q "^error" $L && rc=1
echo "oxide build rc=$rc in $(( $(date +%s) - t0 ))s"
grep "titan:" $L | sort -u
grep -iE "nvcc" $L | grep -v "nvcc not used\|titan:" | head -5
grep -E "Finished|^error" -A8 $L | head -40
ls -la $E/target-oss-oxide/release/mistralrs
exit $rc
