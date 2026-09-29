#!/bin/bash
# llama.cpp vs mistral.rs on one GGUF: 8 prompts x 64 greedy tokens (client.py) + first-token top-10 (toplogp.py).
# usage: pair.sh GGUF NCPUMOE [NAME ENV...]...  each server runs as a transient unit capped at MemoryMax=20G;
# a sampler records peak MemoryCurrent (cgroup, page cache included) and peak RssAnon / RssFile.
# NAME=llama* runs llama.cpp (--n-cpu-moe NCPUMOE, mmap on, extra args in $LLAMA_ARGS); any other NAME runs $BIN with ENV.
set -u; cd "$(dirname "$0")"; mkdir -p out
G=$1; NCM=$2; shift 2
BIN=${BIN:-$HOME/titan-engine/target-next/release/mistralrs}
LL=$HOME/ai/llama.cpp/build/bin/llama-server
LDP=$HOME/titan-engine/lib:/usr/local/cuda/lib64
wait_up() { for i in $(seq 1 450); do curl -sf -m 2 -o /dev/null localhost:$1$2 && return 0; systemctl --user -q is-active $3 || return 1; sleep 2; done; return 1; }
sampler() { # sampler UNIT OUT: peak cgroup memory and RSS split until the unit stops
  local u=$1 o=$2 mc=0 ra=0 rf=0
  while systemctl --user -q is-active $u; do
    local pid=$(systemctl --user show -p MainPID --value $u)
    local c=$(systemctl --user show -p MemoryCurrent --value $u)
    [[ $c =~ ^[0-9]+$ ]] && (( c > mc )) && mc=$c
    if [ -r /proc/$pid/status ]; then
      local a=$(awk '/RssAnon/{print $2}' /proc/$pid/status) f=$(awk '/RssFile/{print $2}' /proc/$pid/status)
      [ -n "$a" ] && (( a > ra )) && ra=$a; [ -n "$f" ] && (( f > rf )) && rf=$f
    fi
    echo "peak MemoryCurrent $((mc >> 20)) MiB, RssAnon $((ra >> 10)) MiB, RssFile $((rf >> 10)) MiB" > $o
    sleep 1
  done
}
unit=
trap '[ -n "$unit" ] && systemctl --user stop $unit 2>/dev/null' EXIT
for spec in "$@"; do
  set -- $spec; name=$1; shift
  unit=next-$name-$$
  if [[ $name == llama* ]]; then
    port=18481; up=/health
    cmd=($LL -m $G -ngl 99 --n-cpu-moe $NCM -c 4096 --port $port --temp 0 --top-k 1 -np 1 ${LLAMA_ARGS:-})
  else
    port=18482; up=/v1/models
    cmd=($BIN --seed 0 serve -p $port --no-ui --paged-attn off --max-seq-len 4096 --format gguf -m $(dirname $G) -f $(basename $G))
  fi
  envs=(--setenv=LD_LIBRARY_PATH=$LDP); for e in "$@"; do envs+=(--setenv=$e); done
  t0=$(date +%s)
  systemd-run --user --unit=$unit --collect -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=1500 "${envs[@]}" \
    -p StandardOutput=truncate:$PWD/out/$name.log -p StandardError=truncate:$PWD/out/$name.log "${cmd[@]}"
  sampler $unit out/$name.mem & sp=$!
  if wait_up $port $up $unit; then
    echo "$name: up in $(( $(date +%s) - t0 ))s"
    timeout 900 python3 ../client.py $port $name 64
    kind=$([[ $name == llama* ]] && echo llama || echo mistral)
    timeout 600 python3 toplogp.py $port $name $kind 2>&1 | tail -1
  else
    echo "$name: failed to come up"; grep -v " INFO " out/$name.log | tail -15
  fi
  systemctl --user stop $unit 2>/dev/null; wait $sp 2>/dev/null
  cat out/$name.mem; grep -o 'titan tiered auto.*\|decode hit rate.*' out/$name.log | tail -2
done
