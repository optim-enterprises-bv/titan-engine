#!/bin/bash
# GPU window: stop titan-mistral, run the real-MXFP4 comparison with the AVX2 CPU twin build, always restart.
trap 'systemctl --user start titan-mistral' EXIT
systemctl --user stop titan-mistral; sleep 3
BIN=$HOME/titan-engine/target-cpu-prod/release/mistralrs ./mxfp4-real.sh
