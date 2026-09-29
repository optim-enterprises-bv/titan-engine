# Sourced by queued jobs. Every step has a timeout; servers run as transient units capped at MemoryMax=20G.
set -u
E=$HOME/titan-engine; O=$E/m5/oss; M=$HOME/ai/models
G20=$M/gpt-oss-20b-F16.gguf; G120=$M/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
BIN=${BIN:-$E/target-oss/release/mistralrs}
cd $O
left() { echo $(( DEADLINE - $(date +%s) )); }
cap() { local t=$1 l=$(( DEADLINE - $(date +%s) - 30 )); echo $(( t < l ? t : l )); }

build() { # cuda build of mr-oss into target-oss
  local L=$O/logs/build-oss.log t=$(date +%s)
  timeout $(cap 1530) systemd-run --user --unit=oss-build --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 \
    -p RuntimeMaxSec=1500 -p WorkingDirectory=$E/mr-oss \
    --setenv=CARGO_BUILD_JOBS=2 --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 --setenv=CUDA_COMPUTE_CAP=120 \
    --setenv=CUDA_HOME=/usr/local/cuda --setenv=NVCC_CCBIN=/usr/bin/g++-15 --setenv=CUDA_NVCC_FLAGS=-Wno-template-body \
    "--setenv=RUSTFLAGS=-L $E/lib" --setenv=CARGO_TARGET_DIR=$E/target-oss \
    --setenv=PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:/usr/bin:/bin \
    -p StandardOutput=truncate:$L -p StandardError=truncate:$L nice -n 10 cargo build --release -p mistralrs-cli --features cuda
  local rc=$?; systemctl --user stop oss-build 2>/dev/null; grep -q "^error" $L && rc=1
  echo "build rc=$rc in $(( $(date +%s) - t ))s"; grep -E "^(warning|error)" -A9 $L | grep -v candle | head -80; grep Finished $L
  ls -la $BIN; return $rc
}
