# nvcc-free build of prefill-stream (mr-pfs) into target-pfs-oxide. Arg: log name.
E=$HOME/titan-engine; L=$E/m4/pfs/logs/${1:-build}.log; t=$(date +%s); rm -f $E/m4/pfs/ctl/build.ok
timeout 1500 systemd-run --user --unit=pfs-build --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 -p RuntimeMaxSec=1480 \
  -p WorkingDirectory=$E/mr-pfs --setenv=CARGO_TARGET_DIR=$E/target-pfs-oxide --setenv=TITAN_OXIDE_DIR=$E/oxide-kernels --setenv=CUDA_HOME=$E/nocuda-bin \
  --setenv=CUDA_PATH=$E/nocuda-bin --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDARC_CUDA_VERSION=13030 --setenv=CARGO_BUILD_JOBS=2 \
  --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 "--setenv=RUSTFLAGS=-L $E/lib" --setenv=PATH=$HOME/.cargo/bin:/usr/bin:/bin \
  -p StandardOutput=truncate:$L -p StandardError=truncate:$L nice -n 10 cargo build --release -p mistralrs-cli --features oxide
rc=$?; systemctl --user stop pfs-build 2>/dev/null; grep -q "^error" $L && rc=1
echo "oxide build rc=$rc in $(( $(date +%s) - t ))s"
grep -c "^error" $L; grep "titan:" $L | sort -u
grep -E "^(error|warning: unused)" -A12 $L | head -120
ls -la $E/target-pfs-oxide/release/mistralrs
[ $rc = 0 ] && [ $E/target-pfs-oxide/release/mistralrs -nt $E/mr-pfs/mistralrs-quant/src/gguf/titan_pfs.rs ] && sha256sum $E/target-pfs-oxide/release/mistralrs > $E/m4/pfs/ctl/build.ok
command -v nvcc >/dev/null && echo "note: nvcc on this shell's PATH (the build's PATH had none)"
exit $rc
