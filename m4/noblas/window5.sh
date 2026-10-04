#!/bin/bash
# noblas window 5 (phase 2): clean llama.cpp references (never --lora; f16 KV, temp 0) where none exist in gates.py
# format: G1 on plain40.json (the prompts the roster gates sent these models) and G3 (4 x ~4k) where the noblas G3
# output differs from the deployed binary's. MoE models that do not fit the card keep their experts on the CPU.
source $HOME/titan-engine/top-noblas/m4/noblas/lib.sh
restic_wait
exec > >(tee -a $I/window5-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin noblas-w5 58
Q=$M/Qwen3.6-35B-A3B-MTP
lref lref-q35 $Q/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf $LLAMA 13500 --n-cpu-moe 20
lref lref-iq2m $M/q35-lowbit/Qwen3.6-35B-A3B-UD-IQ2_M.gguf $LLAMA 13500
lref lref-mx $Q/Qwen3.6-35B-A3B-MXFP4_MOE.gguf $LLAMA 13500 --n-cpu-moe 20
lref lref-oss20 $M/gpt-oss-20b-F16.gguf $LLAMA 6000 --n-cpu-moe 6
lref lref-bonsai $M/bonsai/Bonsai-27B-gguf/Bonsai-27B-Q1_0.gguf $LLAMA 0
lref lref-bonsai2 $M/bonsai2-27b/Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf $E/ref/sudoingx-bonsai2/build/bin/llama-server 13500
lref lref-next $M/qwen3-next-80b/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf $LLAMA 0 --n-cpu-moe 44
lref lref-oss120 $M/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf $LLAMA 0 --n-cpu-moe 34
echo "== noblas-w5 end $(el)"
