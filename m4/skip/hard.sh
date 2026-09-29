#!/bin/bash
# One 80B server: hardgen.py (100 HumanEval + 60 GSM8K, resumable) under a time budget. usage: hard.sh NAME BUDGET ENV...
set -u
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=$HOME/titan-engine/lib:${LD_LIBRARY_PATH:-}
PORT=18473
name=$1 budget=$2; shift 2
env "$@" "$BIN" --seed 0 serve -p $PORT --no-ui --paged-attn off --max-seq-len 4096 \
    --format gguf -m "$(dirname $F80)" -f "$(basename $F80)" > out/$name.hard.server.log 2>&1 &
pid=$!
for i in $(seq 1 240); do
    curl -s -m 2 -o /dev/null localhost:$PORT/v1/models && break
    kill -0 $pid 2>/dev/null || { echo "server died"; tail -20 out/$name.hard.server.log; exit 1; }
    sleep 2
done
python3 hardgen.py $PORT $name $budget; rc=$?
grep -o 'titan miss-skip: .*' out/$name.hard.server.log | tail -1
kill -TERM $pid; for i in 1 2 3 4 5; do kill -0 $pid 2>/dev/null || break; sleep 1; done; kill -KILL $pid 2>/dev/null
wait $pid 2>/dev/null
exit $rc
