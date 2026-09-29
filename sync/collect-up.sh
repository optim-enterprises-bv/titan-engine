#!/bin/bash
# m3/collect.sh adapted for upstream mistral.rs: same prompts/sampling, but server flags come from SFLAGS
# (collect.sh hard-codes --paged-attn off --max-seq-len 4096, which disables upstream's CUDA graphs).
# usage: collect-up.sh NAME PROMPTFILE MAX_TOKENS ENV...   env: BIN DIR FILE SFLAGS NPROMPTS BIG=1 (4k/13k prefill + repeat)
set -u
cd "$(dirname "$0")"
DIR=${DIR:-$HOME/ai/models/Qwen3.6-35B-A3B-MTP}
FILE=${FILE:-Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
SFLAGS=${SFLAGS:---max-seq-len 16384 --pa-context-len 16384}
PORT=18463
name=$1 prompts=$2 max_tokens=$3; shift 3
mkdir -p out
echo "server: $BIN --seed 0 serve -p $PORT --no-ui $SFLAGS --format gguf -m $DIR -f $FILE  env: $*"
t0=$(date +%s)
env LD_LIBRARY_PATH=/usr/local/cuda/lib64 "$@" "$BIN" --seed 0 serve -p $PORT --no-ui $SFLAGS \
    --format gguf -m "$DIR" -f "$FILE" > out/$name.server.log 2>&1 &
pid=$!
for i in $(seq 1 300); do
    curl -s -m 2 -o /dev/null localhost:$PORT/v1/models && break
    kill -0 $pid 2>/dev/null || { echo "server died after $(( $(date +%s) - t0 ))s"; grep -v " INFO " out/$name.server.log | tail -25; exit 1; }
    sleep 2
done
rss() { awk '/VmRSS/ {printf "%d MiB", $2/1024}' /proc/$1/status 2>/dev/null; }
echo "load: $(( $(date +%s) - t0 ))s, RSS $(rss $pid)"
( sleep 60; echo "mem@60s: RSS $(rss $pid)"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>&1 | head -1 ) &
mon=$!
python3 ubench.py "$name" "$PORT" "$max_tokens" "$prompts" "${NPROMPTS:-40}" "${BIG:-0}"
rc=$?
kill $mon 2>/dev/null
grep -iE "device map|layers? .* (on|to) |mapping|cpu|cuda graph|mtp|paged|warn|error" out/$name.server.log | grep -v "^\s*$" | cut -c1-240 | sort | uniq -c | sort -rn | head -25
kill -TERM $pid; for i in 1 2 3 4 5; do kill -0 $pid 2>/dev/null || break; sleep 1; done; kill -KILL $pid 2>/dev/null; wait $pid 2>/dev/null
exit $rc
