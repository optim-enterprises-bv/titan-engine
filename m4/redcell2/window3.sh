#!/bin/bash
# redcell2 window 3 (of 3), no build: the window-2 binary (mr 39719cbdc, sha e992ce46) is the final code.
#  [S] llama.cpp's own REDCELL G3 spread (m4/g4acc/g3spread.sh: GPU llama-server, -ub 256 / -ub 128 / -b 512) at ~4k (g3)
#      and ~7k (g3l, 28000 chars)
#  [T] titan G3 / G3l, prefix cache 0: grouped route (default) and per-token route (TITAN_MOE_UNALIGNED_DECODE=1)
#  [L] llama.cpp GPU G5 (reference server: f16 KV, -ngl 99) for the prefill / decode comparison
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/redcell2/window3.sh
source $HOME/titan-engine/m4/redcell2/lib.sh
exec > >(tee -a $I/window3-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin redcell2-window3 50
BIN=$NEWBIN; echo "binary sha256 $(sha256sum $BIN | cut -c1-16)"
F=$M/redcell-26b/$RED
echo "== [S] llama.cpp G3 spread ($(el))"
timeout 1500 bash $E/m4/g4acc/g3spread.sh $F $O/spread
ls $O/spread
echo "== [T] titan G3 / G3l ($(el))"
if srv_start tg $M/redcell-26b $RED "" "${RA[@]}" --prefix-cache-n 0; then gate g3 tg-g3 2>&1 | tail -1; gate g3 tg-g3l 28000 2>&1 | tail -1; gate g5 tg-g5 13500 128 2>&1 | tail -1; srv_stop; fi
if srv_start tq $M/redcell-26b $RED "TITAN_MOE_UNALIGNED_DECODE=1" "${RA[@]}" --prefix-cache-n 0; then gate g3 tq-g3 2>&1 | tail -1; gate g3 tq-g3l 28000 2>&1 | tail -1; srv_stop; fi
echo "== [L] llama.cpp GPU G5 ($(el))"
LLAMA=$HOME/ai/llama.cpp/build/bin/llama-server
systemd-run --user --unit=red-llama --collect -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=600 \
  -p StandardOutput=truncate:$O/l-g5.server.log -p StandardError=inherit \
  $LLAMA -m $F -c 12288 -ngl 99 -ctk f16 -ctv f16 -np 1 --temp 0 --port $LP --host 127.0.0.1 --no-webui
for i in $(seq 1 150); do curl -sf -m 2 -o /dev/null localhost:$LP/health && break; sleep 2; done
timeout 600 python3 $G4/gates.py g3 $LP llama $O/l-g3.json 2>&1 | tail -1
timeout 600 python3 $G4/gates.py g3 $LP llama $O/l-g3l.json 28000 2>&1 | tail -1
timeout 600 python3 $G4/gates.py g5 $LP llama $O/l-g5.json 13500 128 2>&1 | tail -1
timeout 600 python3 $G4/gates.py g5 $LP llama $O/l-g5l.json 26000 64 2>&1 | tail -1
systemctl --user stop red-llama
echo "== window3 done ($(el))"
