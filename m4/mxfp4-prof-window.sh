#!/bin/bash
# Same speed run as mxfp4-real.sh, with the Q4_K model's expert-hotness profile (same routing).
trap 'systemctl --user start titan-mistral' EXIT
systemctl --user stop titan-mistral; sleep 3
E=$HOME/titan-engine; M=$HOME/ai/models/Qwen3.6-35B-A3B-MTP
export LD_LIBRARY_PATH=$E/lib:/usr/local/cuda/lib64
cd $E/m3 && BIN=$E/target-cpu-prod/release/mistralrs DIR=$M FILE=Qwen3.6-35B-A3B-MXFP4_MOE.gguf ./collect.sh mx-q35-prof $E/m4/prompts-eval.txt 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt 2>&1 | grep -v INFO | grep tok/s
grep -o "decode hit rate.*" out/mx-q35-prof.server.log | tail -1
