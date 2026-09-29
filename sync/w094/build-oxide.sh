# nvcc-free build of titan-094 (mr-094) into target-094-oxide. Arg: log name.
E=$HOME/titan-engine; L=$E/sync/w094/logs/${1:-build-oxide}.log; t=$(date +%s)
timeout 1500 systemd-run --user --unit=t094-build-oxide --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 -p RuntimeMaxSec=1480 \
  -p WorkingDirectory=$E/mr-094 --setenv=CARGO_TARGET_DIR=$E/target-094-oxide --setenv=TITAN_OXIDE_DIR=$E/oxide-kernels --setenv=CUDA_HOME=$E/nocuda-bin \
  --setenv=CUDA_PATH=$E/nocuda-bin --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDARC_CUDA_VERSION=13030 --setenv=CARGO_BUILD_JOBS=2 \
  --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 "--setenv=RUSTFLAGS=-L $E/lib" --setenv=PATH=$HOME/.cargo/bin:/usr/bin:/bin \
  -p StandardOutput=truncate:$L -p StandardError=truncate:$L nice -n 10 cargo build --release -p mistralrs-cli --features oxide
rc=$?; systemctl --user stop t094-build-oxide 2>/dev/null; grep -q "^error" $L && rc=1
echo "oxide build rc=$rc in $(( $(date +%s) - t ))s"
grep -c "^error" $L; grep "titan:" $L | sort -u
ls -la $E/target-094-oxide/release/mistralrs
exit $rc
