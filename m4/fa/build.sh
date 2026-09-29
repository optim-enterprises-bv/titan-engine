# nvcc-free build of flash-decode (mr-fa) into target-fa-oxide.
E=$HOME/titan-engine; W=$E/m4/fa; L=$W/logs/${1:-build}.log; t=$(date +%s); rm -f $W/ctl/build.ok
timeout 1800 systemd-run --user --unit=fa-build --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 -p RuntimeMaxSec=1780 \
  -p WorkingDirectory=$E/mr-fa --setenv=CARGO_TARGET_DIR=$E/target-fa-oxide --setenv=TITAN_OXIDE_DIR=$E/oxide-kernels --setenv=CUDA_HOME=$E/nocuda-bin \
  --setenv=CUDA_PATH=$E/nocuda-bin --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDARC_CUDA_VERSION=13030 --setenv=CARGO_BUILD_JOBS=2 \
  --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 "--setenv=RUSTFLAGS=-L $E/lib" --setenv=PATH=$HOME/.cargo/bin:/usr/bin:/bin \
  -p StandardOutput=truncate:$L -p StandardError=truncate:$L nice -n 10 cargo build --release -p mistralrs-cli --features oxide
rc=$?; systemctl --user stop fa-build 2>/dev/null; grep -q "^error" $L && rc=1
echo "oxide build rc=$rc in $(( $(date +%s) - t ))s"
grep -c "^error" $L; grep -E "^(error|warning: unused)" -A14 $L | head -150; grep Finished $L
ls -la $E/target-fa-oxide/release/mistralrs
[ $rc = 0 ] && [ $E/target-fa-oxide/release/mistralrs -nt $E/mr-fa/mistralrs-core/src/attention/flash_decode.rs ] && sha256sum $E/target-fa-oxide/release/mistralrs > $W/ctl/build.ok
command -v nvcc >/dev/null && echo "note: nvcc on this shell's PATH (the build's PATH had none)"
exit $rc
