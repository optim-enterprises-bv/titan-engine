#!/bin/bash
# The exact llama.cpp command line this box was running before the titan test,
# captured from the live process on 2026-10-23. Restore = run this.
exec /home/dingo/.unsloth/llama.cpp/build/bin/llama-server \
  -m /home/dingo/models/optim-Q3_K_M.gguf --alias optim --host 0.0.0.0 --port 8000 \
  -ngl 99 -ot token_embd=CPU -c 128000 -np 1 -ub 64 -b 128 \
  -ctk q4_0 -ctv q4_0 -fa on --jinja
