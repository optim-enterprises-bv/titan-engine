#!/bin/bash
# Build the kernel-tier harness (m4/kbench/kbench.cpp) into bench/bin/kbench. Host C++ only: links
# ~/ai/llama.cpp's libggml* (acecd56), cudart and titan-oxide-ffi's staticlib. No GPU use.
set -eu
E=$HOME/titan-engine; K=$E/m4/kbench; B=$HOME/ai/llama.cpp/build/bin; O=$E/bench/bin/kbench
NV=/opt/nvidia/nsight-compute/2026.2.1/host/target-linux-x64/nvtx/include
FFI=$E/oxide-kernels/titan-oxide-ffi/target/release/libtitan_oxide_ffi.a
mkdir -p $E/bench/bin
g++-15 -O2 -std=c++17 $K/kbench.cpp -o $O.tmp \
  -I$HOME/ai/llama.cpp/ggml/include -I/usr/local/cuda/include -I$NV \
  -L$B -lggml -lggml-base -lggml-cuda -Wl,-rpath,$B \
  -L/usr/local/cuda/lib64 -lcudart -Wl,-rpath,/usr/local/cuda/lib64 \
  $FFI -lcuda -lpthread -ldl -lm -lrt -lutil -lgcc_s
mv $O.tmp $O
