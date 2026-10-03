#!/bin/bash
# g4acc window 3: regression core on the window-2 binary (mr-g4acc 492a555bc, oxide gate PASS) vs bin/mistralrs-titan-integ2;
# gemma4 bf16 G1 / G2 identity vs integ2 (integ2 procedure); Spark identity; llama.cpp self-perturbation G3 (ubatch 256 /
# 128, flash on, f16 KV) to measure the reference's own noise floor; P2 G3 repeat (determinism).
set -u
source $HOME/titan-engine/m4/g4acc/lib.sh
exec > >(tee -a $I/window3-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin window3 58
NEW=$I/bin/mistralrs-w2; BIN=$NEW; echo "binary sha256 $(sha256sum $BIN | cut -c1-16)"
G12D=$M/gemma4-12b-qat; G12=gemma-4-12b-it-qat-q4_0.gguf; GP=$G4/prompts/gemma4-12b.json
same() { python3 $I/same.py "$@"; }
C=python3; CL=$I/b2client.py

echo "== [R] regression core ($(el))"
coll g4acc-m2 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2 && cmp40 $E/m6/out/h-off.json $E/m3/out/g4acc-m2.json
coll g4acc-mf0 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=0 && cmp40 $E/m6/out/mtpfile-off.json $E/m3/out/g4acc-mf0.json
if srv_start b-off $M/bonsai2-27b Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf "TITAN_MTP=0 TITAN_REASONING_EFFORT=medium" --max-seq-len 65536; then
  $C $CL greedy titan $PORT b-off 8 300; srv_stop; $C $CL cmp o-off b-off
fi
if srv_start i-new $M/q35-lowbit Qwen3.6-35B-A3B-UD-IQ2_M.gguf "TITAN_MTP=2 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64" --max-seq-len 65536; then
  $C $CL greedy titan $PORT i-new 8 300; srv_stop; $C $CL cmp i-integ2 i-new
fi
if srv_start q-new $M/bonsai/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf "TITAN_ATTN_FLASH_PREFILL=1" --max-seq-len 32768; then
  $C $CL greedy titan $PORT q-new 8 300; srv_stop; $C $CL cmp q-integ2 q-new
fi

echo "== [G] gemma4-12b bf16 (integ2 procedure) G1 + G2 ($(el))"
if srv_start g12s $G12D $G12 "" --max-model-len 8192 --max-seq-len 8192 -n 0:48 --dtype bf16; then
  gate g1 b-g1 $GP; gate g2 b-g2 $GP 256; srv_stop
fi
tmap gemma4 $O/b-g1.json > /dev/null; same $O/b-g1.json $O/g12-g1-integ2.json; cmpg g1 $O/b-g1.json $R/ref-g12-g1.json | head -1
BIN=$INTEG
if srv_start g12i $G12D $G12 "" --max-model-len 8192 --max-seq-len 8192 -n 0:48 --dtype bf16; then
  gate g2 bi-g2 $GP 256; srv_stop
fi
BIN=$NEW; same $O/b-g2.json $O/bi-g2.json

echo "== [S] Spark identity ($(el))"
if srv_start sp $M/spark-x2.5 Spark-X2.5-4B-Q4_K_M.gguf "" --max-model-len 16384 --max-seq-len 16384; then
  for c in 2000 15000 33000; do gate g5 sp3-g5-$c $c 128; done
  gate g3 sp3-g3; gate g1 sp3-g1 $G4/prompts/spark.json; gate g2 sp3-g2 $G4/prompts/spark.json 256; srv_stop
  tmap spark $O/sp3-g1.json $O/sp3-g3.json > /dev/null
  same $O/sp3-g3.json $O/sp-g3-integ2.json; same $O/sp3-g1.json $O/sp-g1-integ2.json; same $O/sp3-g2.json $O/sp-g2-integ2.json
fi

echo "== [L] llama.cpp self-perturbation ($(el))"
KIND=llama
for v in "lub256 -ub 256" "lub128 -ub 128" "lb512 -b 512"; do
  set -- $v; n=$1; shift
  if lsrv $n $G12D/$G12 -c 12288 -ngl 99 -ctk f16 -ctv f16 "$@"; then
    gate g3 $n-g3; gate g3 $n-g3l 28000; lstop
    cmpg g3 $O/$n-g3.json $R/ref-g12-g3.json | head -1; cmpg g3 $O/$n-g3l.json $R/ref-g12-g3l.json | head -1
  else lstop; fi
done
KIND=titan

echo "== [P] P2 G3 repeat ($(el))"
if [ $(left) -gt 200 ] && srv_start g12-q2r $G12D $G12 "" --max-model-len 12288 --max-seq-len 12288 -n 0:48 --prefix-cache-n 0 --dtype f32; then
  gate g3 q2r-g3; srv_stop; tmap gemma4 $O/q2r-g3.json > /dev/null; same $O/q2r-g3.json $O/q2-g3.json
fi
free -m | head -2
