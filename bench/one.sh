#!/bin/bash
# One window: bench the deployed unit (its binary and env) as-is.
trap 'systemctl --user start titan-mistral' EXIT
E=$HOME/titan-engine
for p in $(ps -eo pid=,ppid=,comm= | awk '$2 == 878468 && $3 ~ /rust-analyzer/ {print $1}'); do kill $p; done
systemctl --user stop titan-mistral; sleep 3
BIN=$(python3 - <<PY
import shlex,os
for l in open(os.path.expanduser("~/titan-engine/deploy/titan-mistral.service")):
    if l.startswith("ExecStart="): print(shlex.split(l[10:])[0].replace("%h",os.path.expanduser("~")))
PY
)
echo "bench binary: $BIN"
BENCH_SRC=$E/mr-094 bash $E/bench/run.sh "$BIN" "$1"
