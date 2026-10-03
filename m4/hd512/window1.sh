#!/bin/bash
# hd512 window 1 (diagnostics, no build): where does gemma4-12b's ~4k G3 drift come from?
#   T1 integ binary, bf16, TITAN_G4_ATTN_F32=1 (prompt attention on F32 copies)
#   T2 integ binary, --dtype f32 (whole activation path + KV cache in F32)
#   L1 llama.cpp GPU with bf16 K/V cache (what titan's bf16 KV storage alone costs llama)
#   L2 llama.cpp GPU with -fa off (eager attention in llama.cpp)
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/hd512/window1.sh
set -u
source $HOME/titan-engine/m4/hd512/lib.sh
exec > >(tee -a $I/window1-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin window1 25
BIN=$INTEG; G12D=$M/gemma4-12b-qat; G12=gemma-4-12b-it-qat-q4_0.gguf
G12F=$G12D/$G12
echo "binary $(sha256sum $BIN | cut -c1-16)"
CTX=(--max-model-len 12288 --max-seq-len 12288 -n 0:48 --prefix-cache-n 0)

for v in "t1 bf16 TITAN_G4_ATTN_F32=1" "t2 f32 X=1" "t0 bf16 X=1"; do
  set -- $v
  if srv_start g12-$1 $G12D $G12 "$3" "${CTX[@]}" --dtype $2; then
    gate g3 $1-g3; gate g3 $1-g3l 28000
    srv_stop
    tmap gemma4 $O/$1-g3.json $O/$1-g3l.json
    cmpg g3 $O/$1-g3.json $R/ref-g12-g3.json; cmpg g3 $O/$1-g3l.json $R/ref-g12-g3l.json
  else srv_stop; fi
done

KIND=llama
for v in "l1 -ctk bf16 -ctv bf16" "l2 -fa off -ctk f16 -ctv f16" "l3 -ctk f16 -ctv f16"; do
  set -- $v; n=$1; shift
  if lsrv $n $G12F -c 12288 -ngl 99 "$@"; then
    gate g3 $n-g3; gate g3 $n-g3l 28000
    lstop
    cmpg g3 $O/$n-g3.json $R/ref-g12-g3.json; cmpg g3 $O/$n-g3l.json $R/ref-g12-g3l.json
  else lstop; fi
done
free -m | head -2
