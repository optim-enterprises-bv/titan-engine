#!/bin/bash
# noblas window 6 (phase 2): [D] decode tok/s per roster model, noblas vs deployed (one swap server each, deployed
# roster); [N] nsys (cuda + cublas trace) of one noblas swap server serving every roster model: zero cuBLAS ranges and
# kernels; [S] llama.cpp G1 / G3 spread (-ub 256 / -ub 128 / -b 512) for the models whose output changed.
source $HOME/titan-engine/top-noblas/m4/noblas/lib.sh
restic_wait
exec > >(tee -a $I/window6-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin noblas-w6 58
BIN=$NEWBIN roster_dec new
BIN=$OLDBIN roster_dec old
echo "== [N] ($(el))"
BIN=$NEWBIN; sed "s/^port = 1234/port = $PORT/" $E/deploy/models.toml > $O/roster-nsys.toml
mkdir -p $O/nsys-tmp; rm -f $O/proof-noblas.nsys-rep
if SENV="TMPDIR=$O/nsys-tmp" KSIG=SIGINT KTO=300 cfg_start proof-noblas $O/roster-nsys.toml \
     $NSYS profile -t cuda,cublas --sample=none --cpuctxsw=none --cuda-graph-trace=node --cuda-flush-interval=100 -o $O/proof-noblas -f true; then
  while read m rest; do [ -z "$m" ] || [ "${m:0:1}" = "#" ] && continue; [ $(left) -gt 1200 ] || break
    timeout 600 python3 $I/dec.py $PORT $m $O/proof-$m.json 1 32; done < $I/roster.gate
  srv_stop
fi
ls -la $O/proof-noblas.nsys-rep
echo "== [S] ($(el))"
Q=$M/Qwen3.6-35B-A3B-MTP
lspread sp-q35 $Q/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf $LLAMA 13500 --n-cpu-moe 20
G1P=$E/m4/g4s/prompts/redcell.json lspread sp-red $M/redcell-26b/REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf $LLAMA 0
lspread sp-iq2m $M/q35-lowbit/Qwen3.6-35B-A3B-UD-IQ2_M.gguf $LLAMA 13500
lspread sp-mx $Q/Qwen3.6-35B-A3B-MXFP4_MOE.gguf $LLAMA 13500 --n-cpu-moe 20
lspread sp-oss20 $M/gpt-oss-20b-F16.gguf $LLAMA 6000 --n-cpu-moe 6
lspread sp-bonsai2 $M/bonsai2-27b/Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf $E/ref/sudoingx-bonsai2/build/bin/llama-server 13500
echo "== noblas-w6 end $(el)"
