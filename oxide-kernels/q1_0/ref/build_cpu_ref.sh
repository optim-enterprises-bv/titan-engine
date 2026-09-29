#!/bin/sh
# Same compiler and flags as llama.cpp's ggml-cpu (compile_commands.json): gcc-15 -O3 -march=native -std=gnu11.
L=$HOME/ai/llama.cpp
cd "$(dirname "$0")"
gcc-15 -O3 -DNDEBUG -std=gnu11 -march=native -I$L/ggml/include cpu_ref.c -o cpu_ref \
  -L$L/build/bin -lggml-cpu -lggml-base -Wl,-rpath,$L/build/bin
