# nvcc-free build of mmvq-moe (mr-mmvqmoe) into target-mmvqmoe-oxide (gate d: it links without nvcc).
E=$HOME/titan-engine; W=$E/m4/mmvqmoe; L=$W/logs/${1:-build}.log; t=$(date +%s); rm -f $W/ctl/build.ok
bash $E/oxide-kernels/mmvq-moe/export_ptx.sh || exit 1
timeout 1800 systemd-run --user --unit=mmvqmoe-build --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 -p RuntimeMaxSec=1780 \
  -p WorkingDirectory=$E/mr-mmvqmoe --setenv=CARGO_TARGET_DIR=$E/target-mmvqmoe-oxide --setenv=TITAN_OXIDE_DIR=$E/oxide-kernels --setenv=CUDA_HOME=$E/nocuda-bin \
  --setenv=CUDA_PATH=$E/nocuda-bin --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDARC_CUDA_VERSION=13030 --setenv=CARGO_BUILD_JOBS=2 \
  --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 "--setenv=RUSTFLAGS=-L $E/lib" --setenv=PATH=$HOME/.cargo/bin:/usr/bin:/bin \
  -p StandardOutput=truncate:$L -p StandardError=truncate:$L nice -n 10 cargo build --release -p mistralrs-cli --features oxide
rc=$?; systemctl --user stop mmvqmoe-build 2>/dev/null; grep -q "^error" $L && rc=1
echo "oxide build rc=$rc in $(( $(date +%s) - t ))s"
grep -c "^error" $L; grep -E "^(error|warning: unused)" -A14 $L | head -150; grep Finished $L
B=$E/target-mmvqmoe-oxide/release/mistralrs
ls -la $B
[ $rc = 0 ] && [ $B -nt $E/mr-mmvqmoe/mistralrs-quant/src/gguf/titan_tiered.rs ] && sha256sum $B > $W/ctl/build.ok
command -v nvcc >/dev/null && echo "note: nvcc on this shell's PATH (the build's PATH had none)"
ldd $B | grep -i "not found\|cuda\|nvrtc" | head
exit $rc
