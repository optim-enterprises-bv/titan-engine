# nvcc (--features cuda) build of titan-094 into target-094-cuda (seeded from target-oss). Arg: log name, time cap (s)
E=$HOME/titan-engine; L=$E/sync/w094/logs/${1:-build-cuda}.log; CAP=${2:-1800}; t=$(date +%s)
mkdir -p $E/cuda-build-094
timeout $CAP systemd-run --user --unit=t094-build-cuda --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 -p RuntimeMaxSec=$((CAP-20)) \
  -p WorkingDirectory=$E/mr-094 --setenv=CARGO_TARGET_DIR=$E/target-094-cuda --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDA_HOME=/usr/local/cuda \
  --setenv=NVCC_CCBIN=/usr/bin/g++-15 --setenv=CUDA_NVCC_FLAGS=-Wno-template-body --setenv=CARGO_BUILD_JOBS=2 \
  --setenv=MISTRALRS_CUDA_BUILD_ROOT=$E/cuda-build-094 \
  --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 "--setenv=RUSTFLAGS=-L $E/lib" --setenv=PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:/usr/bin:/bin \
  -p StandardOutput=truncate:$L -p StandardError=truncate:$L nice -n 10 cargo build --release -p mistralrs-cli --features cuda
rc=$?; systemctl --user stop t094-build-cuda 2>/dev/null; grep -q "^error" $L && rc=1
echo "cuda build rc=$rc in $(( $(date +%s) - t ))s"; grep -E "^error" -A12 $L | head -80; grep Finished $L
ls -la $E/target-094-cuda/release/mistralrs
exit $rc
