#!/bin/bash
# Regenerate llama.cpp's G3 self-variation (GPU; run inside a window, sourcing a window lib.sh first):
#   source m4/g4acc/lib.sh; window_begin ...; bash m4/g4acc/g3spread.sh GGUF OUTDIR
# -> OUTDIR/{lub256,lub128,lb512}-g3{,l}.json: the reference's llama-server (f16 KV, FA on, ngl 99) with only the
# batching changed. Then: g3spread.py REF.json OUTDIR/lub256-g3.json,OUTDIR/lub128-g3.json,OUTDIR/lb512-g3.json -- TITAN.json
set -u
F=$1; OUT=$2; LLAMA=$HOME/ai/llama.cpp/build/bin/llama-server; G=$HOME/titan-engine/m4/g4s; P=18651
for v in "lub256 -ub 256" "lub128 -ub 128" "lb512 -b 512"; do
  set -- $v; n=$1; shift
  systemd-run --user --unit=g3spread-llama --collect -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=900 \
    -p StandardOutput=truncate:$OUT/$n.server.log -p StandardError=inherit \
    $LLAMA -m $F -c 12288 -ngl 99 -ctk f16 -ctv f16 -np 1 --temp 0 --port $P --host 127.0.0.1 --no-webui "$@"
  for i in $(seq 1 150); do curl -sf -m 2 -o /dev/null localhost:$P/health && break; sleep 2; done
  timeout 900 python3 $G/gates.py g3 $P llama $OUT/$n-g3.json
  timeout 900 python3 $G/gates.py g3 $P llama $OUT/$n-g3l.json 28000
  systemctl --user stop g3spread-llama
done
