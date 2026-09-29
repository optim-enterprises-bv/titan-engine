#!/bin/bash
# Compile kbench (host C++ only: links ~/ai/llama.cpp's libggml*, cudart, and titan-oxide-ffi's staticlib).
set -eu
K=$HOME/titan-engine/m4/kbench; B=$HOME/ai/llama.cpp/build/bin
NV=/opt/nvidia/nsight-compute/2026.2.1/host/target-linux-x64/nvtx/include
FFI=$HOME/titan-engine/oxide-kernels/titan-oxide-ffi/target/release/libtitan_oxide_ffi.a
g++-15 -O2 -std=c++17 $K/kbench.cpp -o $K/kbench \
  -I$HOME/ai/llama.cpp/ggml/include -I/usr/local/cuda/include -I$NV \
  -L$B -lggml -lggml-base -lggml-cuda -Wl,-rpath,$B \
  -L/usr/local/cuda/lib64 -lcudart -Wl,-rpath,/usr/local/cuda/lib64 \
  $FFI -lcuda -lpthread -ldl -lm -lrt -lutil -lgcc_s
