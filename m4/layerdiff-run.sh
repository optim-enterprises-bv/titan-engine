#!/bin/bash
E=$HOME/titan-engine
systemd-run --user --unit=titan-ldbuild --collect --wait -q -p MemoryMax=9G -p MemorySwapMax=0 -p WorkingDirectory=$E/snap/mr \
  --setenv=CARGO_TARGET_DIR=$E/target-oxide --setenv=TITAN_OXIDE_DIR=$E/oxide-kernels --setenv=CUDA_HOME=$E/nocuda-bin --setenv=CUDA_PATH=$E/nocuda-bin \
  --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDARC_CUDA_VERSION=13030 --setenv=CARGO_BUILD_JOBS=4 "--setenv=RUSTFLAGS=-L $E/lib" --setenv=PATH=$HOME/.cargo/bin:/usr/bin:/bin \
  -p StandardOutput=truncate:$E/build-nonvcc.log -p StandardError=truncate:$E/build-nonvcc.log nice -n 15 ionice -c3 cargo build --release -p mistralrs-cli --features oxide || { echo build failed; grep -E '^error' -A8 $E/build-nonvcc.log | head -30; exit 1; }
echo build ok
cd $E/m4 && rm -f out/layers-35b.txt
export LD_LIBRARY_PATH=$E/lib TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt TITAN_LAYER_DUMP=$E/m4/out/layers-35b.txt
$E/target-oxide/release/mistralrs --seed 0 serve -p 18490 --no-ui --paged-attn off --max-seq-len 4096 --format gguf -m $HOME/ai/models -f Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf > out/layerdump.server.log 2>&1 & pid=$!
for i in $(seq 1 300); do curl -sf -m 2 -o /dev/null localhost:18490/v1/models && break; sleep 2; done
curl -s localhost:18490/v1/completions -H 'Content-Type: application/json' -d '{"model":"default","prompt":"The capital of France is","max_tokens":1,"temperature":0}' | head -c 300; echo
kill $pid; wait $pid 2>/dev/null
python3 layerdiff.py ${TMPDIR:-/tmp}/evalcb.txt out/layers-35b.txt
