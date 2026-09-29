#!/bin/bash
# M6: nvcc-free build (--features oxide, nvcc unreachable: stub CUDA_HOME, no cuda bin on PATH) into target-mtp-oxide; proves the oxide launchers link.
E=$HOME/titan-engine
systemd-run --user --unit=mtp-build-oxide --collect --wait -q -p MemoryMax=9G -p MemorySwapMax=0 -p WorkingDirectory=$E/mr-m6 \
  --setenv=CARGO_TARGET_DIR=$E/target-mtp-oxide --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDA_HOME=$E/nocuda-bin --setenv=CUDA_PATH=$E/nocuda-bin \
  --setenv=TITAN_OXIDE_DIR=$E/oxide-kernels --setenv=CUDARC_CUDA_VERSION=13030 \
  --setenv=CARGO_BUILD_JOBS=4 "--setenv=RUSTFLAGS=-L $E/lib" --setenv=PATH=$HOME/.cargo/bin:/usr/bin:/bin \
  -p StandardOutput=truncate:$E/m6/build-oxide.log -p StandardError=truncate:$E/m6/build-oxide.log \
  nice -n 10 cargo build --release -p mistralrs-cli --features oxide
rc=$?
grep -E '^error' -A12 $E/m6/build-oxide.log | head -80
echo "oxide build rc=$rc"
