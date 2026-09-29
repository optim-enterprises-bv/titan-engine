#!/bin/bash
# nvcc-free build of the titan-094 integration candidate (flash-decode ad653651a+b7f5c6e98 fix, mmvq-moe
# f74ee6668, HEAD f5458a45c) into the warm target-kv-oxide, in place inside mr-094 (already at f5458a45c).
# PTX for both kernels is already embedded in the tree (verified identical to oxide-kernels ce2b189/ec96e5b
# modulo trailing whitespace); no export_ptx.sh step needed.
set -u
E=$HOME/titan-engine; W=$E/m4/fm; L=$W/logs/${1:-build}.log; t=$(date +%s)
mkdir -p $W/logs $W/ctl
rm -f $W/ctl/build.ok
timeout 1800 systemd-run --user --unit=fm-build --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 -p RuntimeMaxSec=1780 \
  -p WorkingDirectory=$E/mr-094 --setenv=CARGO_TARGET_DIR=$E/target-kv-oxide --setenv=TITAN_OXIDE_DIR=$E/oxide-kernels --setenv=CUDA_HOME=$E/nocuda-bin \
  --setenv=CUDA_PATH=$E/nocuda-bin --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDARC_CUDA_VERSION=13030 --setenv=CARGO_BUILD_JOBS=2 \
  --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 "--setenv=RUSTFLAGS=-L $E/lib" --setenv=PATH=$HOME/.cargo/bin:/usr/bin:/bin \
  -p StandardOutput=truncate:$L -p StandardError=truncate:$L nice -n 10 cargo build --release -p mistralrs-cli --features oxide
rc=$?; systemctl --user stop fm-build 2>/dev/null; grep -q "^error" $L && rc=1
echo "oxide build rc=$rc in $(( $(date +%s) - t ))s"
grep -c "^error" $L; grep -E "^(error|warning: unused)" -A14 $L | head -150; grep Finished $L
B=$E/target-kv-oxide/release/mistralrs
ls -la $B
[ $rc = 0 ] && [ $B -nt $E/mr-094/mistralrs-quant/src/gguf/titan_tiered.rs ] && sha256sum $B > $W/ctl/build.ok
command -v nvcc >/dev/null && echo "note: nvcc on this shell's PATH (the build's PATH had none)"
exit $rc
