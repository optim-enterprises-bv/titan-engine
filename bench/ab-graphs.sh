#!/bin/bash
# One window: bench the deployed kv build with CUDA graphs off, then on (A/B of TITAN_CUDA_GRAPHS).
trap 'systemctl --user start titan-mistral' EXIT
E=$HOME/titan-engine; BIN=$E/bin/mistralrs-titan-kv
for p in $(ps -eo pid=,ppid=,comm= | awk '$2 == 878468 && $3 ~ /rust-analyzer/ {print $1}'); do kill $p; done
systemctl --user stop titan-mistral; sleep 3
BENCH_SRC=$E/mr-094 bash $E/bench/run.sh $BIN kv-graphs-off
BENCH_SRC=$E/mr-094 BENCH_EXTRA_ENV="TITAN_CUDA_GRAPHS=1" bash $E/bench/run.sh $BIN kv-graphs-on
