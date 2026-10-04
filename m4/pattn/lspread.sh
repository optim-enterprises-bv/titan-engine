#!/bin/bash
# llama.cpp GPU reference + its batching self-variation (m4/g4acc/g3spread.sh procedure: f16 KV, ngl 99, -np 1) for the
# gates.py G3 long prompts at the given lengths. Inside a window only (GPU). usage: lspread.sh GGUF OUTDIR CTX CHARS...
# -> OUTDIR/{base,lub256,lub128,lb512}-c<CHARS>.json
set -u
F=$1; OUT=$2; CTX=$3; shift 3; LLAMA=$HOME/ai/llama.cpp/build/bin/llama-server; G=$HOME/titan-engine/m4/g4s; P=18672
mkdir -p $OUT
for v in "base" "lub256 -ub 256" "lub128 -ub 128" "lb512 -b 512"; do
  set -- $v "${@}"; n=$1; shift; args=(); while [ $# -gt 0 ] && [[ $1 == -* ]]; do args+=("$1" "$2"); shift 2; done
  systemd-run --user --unit=pattn-llama --collect -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=1200 \
    -p StandardOutput=truncate:$OUT/$n.server.log -p StandardError=inherit \
    $LLAMA -m $F -c $CTX -ngl 99 -ctk f16 -ctv f16 -np 1 --temp 0 --port $P --host 127.0.0.1 --no-webui "${args[@]}"
  for i in $(seq 1 150); do curl -sf -m 2 -o /dev/null localhost:$P/health && break; sleep 2; done
  for c in "$@"; do timeout 1200 python3 $G/gates.py g3 $P llama $OUT/$n-c$c.json $c | tail -1; done
  systemctl --user stop pattn-llama
done
