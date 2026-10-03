#!/bin/bash
# hd512 window 3 (last): gates on the window-2 binary (bin/mistralrs-w2 = mr-hd512 4aa8b028a + oxide be03263; defaults:
# gemma4 prompt attention on flash-prefill, inputs model dtype, kv bf16). [G] gemma4-12b G1 (integ config) identity,
# G3 ~4k / ~8k repeat, G2 new vs integ binary + llama tie check; [S] Spark full integ procedure incl. G2 identity;
# [RC] REDCELL G3 x2 flash on / off + G1; [R] regression core.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/hd512/window3.sh
set -u
source $HOME/titan-engine/m4/hd512/lib.sh
exec > >(tee -a $I/window3-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin window3 58
NEW=$I/bin/mistralrs-w2; BIN=$NEW; echo "binary sha256 $(sha256sum $BIN | cut -c1-16)"
G12D=$M/gemma4-12b-qat; G12=gemma-4-12b-it-qat-q4_0.gguf; GP=$G4/prompts/gemma4-12b.json
same() { python3 $I/same.py "$@"; }
tps() { for f in "$@"; do python3 -c "import json,sys;r=json.load(open(sys.argv[1]));print('TPS',sys.argv[1].split('/')[-1],[(x['prompt_tokens'],round(x['prefill_tps'] or 0,1),round(x['decode_tps'] or 0,1)) for x in r])" $f; done; }
C=python3; CL=$I/b2client.py

echo "== [G1] gemma4-12b G1 + G2, integ config (8192, default cache) ($(el))"
if srv_start g12s $G12D $G12 "" --max-model-len 8192 --max-seq-len 8192 -n 0:48 --dtype bf16; then
  gate g1 f-g1 $GP; gate g2 f-g2 $GP 256; srv_stop
fi
BIN=$INTEG
if srv_start g12i $G12D $G12 "" --max-model-len 8192 --max-seq-len 8192 -n 0:48 --dtype bf16; then
  gate g1 i-g1 $GP; gate g2 i-g2 $GP 256; srv_stop
fi
BIN=$NEW
tmap gemma4 $O/f-g1.json $O/i-g1.json > /dev/null
same $O/f-g1.json $E/m4/integ/out/g12-g1.json; same $O/f-g1.json $O/i-g1.json; same $O/f-g2.json $O/i-g2.json
cmpg g1 $O/f-g1.json $R/ref-g12-g1.json | head -1; cmpg g2 $O/f-g2.json $R/ref-g12-g2.json; cmpg g2 $O/i-g2.json $R/ref-g12-g2.json
echo "== [G3] gemma4-12b G3 repeat, prefix-cache-n 0 ($(el))"
if srv_start g12p $G12D $G12 "" --max-model-len 12288 --max-seq-len 12288 -n 0:48 --dtype bf16 --prefix-cache-n 0; then
  gate g3 f-g3; gate g3 f-g3l 28000; echo "VRAM: $(smi); OOM lines $(grep -c OUT_OF_MEMORY $SLOG)"; srv_stop
fi
tmap gemma4 $O/f-g3.json $O/f-g3l.json > /dev/null
cmpg g3 $O/f-g3.json $R/ref-g12-g3.json | head -1; cmpg g3 $O/f-g3l.json $R/ref-g12-g3l.json | head -1
same $O/f-g3.json $O/e1-g3.json; same $O/f-g3l.json $O/e1-g3l.json; tps $O/f-g3.json $O/f-g3l.json
echo "== [T] llama tie check on gemma4 G2 divergences ($(el))"
if lsrv l-tie $G12D/$G12 -c 8192 -ngl 99 -ctk f16 -ctv f16; then
  KIND=llama gate tie tie-g12 $GP $O/f-g2.json $R/ref-g12-g2.json; lstop
fi

echo "== [S] Spark integ procedure ($(el))"
if srv_start sp $M/spark-x2.5 Spark-X2.5-4B-Q4_K_M.gguf "" --max-model-len 16384 --max-seq-len 16384; then
  for c in 2000 15000 33000; do gate g5 sp3-g5-$c $c 128; done
  gate g3 sp3-g3; gate g1 sp3-g1 $G4/prompts/spark.json; gate g2 sp3-g2 $G4/prompts/spark.json 256; srv_stop
  tmap spark $O/sp3-g1.json $O/sp3-g3.json > /dev/null
  same $O/sp3-g3.json $E/m4/integ/out/sp-g3.json; same $O/sp3-g1.json $E/m4/integ/out/sp-g1.json; same $O/sp3-g2.json $E/m4/integ/out/sp-g2.json
  cmpg g3 $O/sp3-g3.json $R/refc-spark-g3.json | head -1
fi

echo "== [R] regression core ($(el))"
coll hd512-m2 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2 && cmp40 $E/m6/out/h-off.json $E/m3/out/hd512-m2.json
coll hd512-mf0 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=0 && cmp40 $E/m6/out/mtpfile-off.json $E/m3/out/hd512-mf0.json
if srv_start b-off $M/bonsai2-27b Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf "TITAN_MTP=0 TITAN_REASONING_EFFORT=medium" --max-seq-len 65536; then
  $C $CL greedy titan $PORT b-off 8 300; srv_stop; $C $CL cmp o-off b-off
fi
if srv_start i-new $M/q35-lowbit Qwen3.6-35B-A3B-UD-IQ2_M.gguf "TITAN_MTP=2 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64" --max-seq-len 65536; then
  $C $CL greedy titan $PORT i-new 8 300; srv_stop; $C $CL cmp i-integ i-new
fi
if srv_start q-new $M/bonsai/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf "TITAN_ATTN_FLASH_PREFILL=1" --max-seq-len 32768; then
  $C $CL greedy titan $PORT q-new 8 300; srv_stop; $C $CL cmp q-integ q-new
fi

echo "== [RC] REDCELL G3 x2 flash on / off + G1 ($(el))"
RD=$M/redcell-26b; RF=REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf
for v in "ra TITAN_G4_ATTN_FLASH=1" "rb TITAN_G4_ATTN_FLASH=0" "rc TITAN_G4_ATTN_FLASH=1" "rd TITAN_G4_ATTN_FLASH=0"; do
  set -- $v; n=$1; shift
  if [ $(left) -gt 240 ] && srv_start red-$n $RD $RF "$*" --max-model-len 8192 --max-seq-len 8192 -n 0:30 --dtype bf16 --prefix-cache-n 0; then
    gate g3 $n-g3; [ $n = ra ] && gate g1 $n-g1 $G4/prompts/redcell.json; srv_stop
    tmap gemma4 $O/$n-g3.json > /dev/null; cmpg g3 $O/$n-g3.json $R/ref-red-g3.json | head -1; tps $O/$n-g3.json
  else srv_stop; fi
done
[ -e $O/ra-g1.json ] && tmap gemma4 $O/ra-g1.json > /dev/null && cmpg g1 $O/ra-g1.json $R/ref-red-g1.json | head -1
free -m | head -2
