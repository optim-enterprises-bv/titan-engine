#!/bin/bash
# g4acc window 4 (follow-up 1 of 3): [K] gate re-run (decode-row cases for d512 bf16); [B] build (bf16 decode on
# flash-prefill); [F] --dtype f32 + TITAN_G4_KV16=0 diagnosis: fresh-server G1, then G3 / G3l (8k OOM), then G1 again;
# [D] decode tok/s short / 3.5k / 8k: new bf16, flash decode off, integ2, llama.cpp; [G] bf16 G1 (integ2 procedure) +
# G2 + tie; P2 G1; [RC] REDCELL G1 + decode; [S] Spark identity; [R] regression core.
set -u
source $HOME/titan-engine/m4/g4acc/lib.sh
exec > >(tee -a $I/window4-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin window4 58
K=$OX/flash-prefill; G12D=$M/gemma4-12b-qat; G12=gemma-4-12b-it-qat-q4_0.gguf; GP=$G4/prompts/gemma4-12b.json
same() { python3 $I/same.py "$@"; }
tps() { for f in "$@"; do python3 -c "import json,sys;r=json.load(open(sys.argv[1]));print('TPS',sys.argv[1].split('/')[-1],[(x['prompt_tokens'],round(x['prefill_tps'] or 0,1),round(x['decode_tps'] or 0,1)) for x in r])" $f; done; }
C=python3; CL=$I/b2client.py
PC0=(--max-model-len 12288 --max-seq-len 12288 -n 0:48 --prefix-cache-n 0)
echo "src: $(git -C $SRC log --oneline -1 | cut -c1-60) $(git -C $SRC diff --quiet && echo clean || echo DIRTY); oxide $(git -C $OX diff --quiet && echo clean || echo DIRTY)"

echo "== [K] gate ($(el))"
cp $K/flash_prefill.ptx $O/fp-w3.ptx
if oxbuild flash-prefill g4acc-ox 600; then
  cmp -s $K/flash_prefill.ptx $O/fp-w3.ptx && echo "PTX unchanged (gate-only change)" || echo "!! PTX changed"
  (cd $K && timeout 900 ./target/release/flash-prefill > $O/gate4-fp.log 2>&1); echo "gate rc=$?"
  grep -E "accuracy|mutations|differ in any bit|gate:|WEAK|FAIL" $O/gate4-fp.log | grep -v informational | tail -20
fi

echo "== [B] build ($(el))"
rm -f $I/rebuild.flag
until mbuild 1500; do
  echo "build failed: waiting for $I/rebuild.flag ($(left)s left)"; date -Is > $O/build-failed.flag
  while [ ! -e $I/rebuild.flag ] && [ $(left) -gt 1500 ]; do sleep 10; done
  [ -e $I/rebuild.flag ] || { echo "no binary"; exit 1; }
  rm -f $I/rebuild.flag $O/build-failed.flag
done
cp $MBIN $I/bin/mistralrs-w4; BIN=$I/bin/mistralrs-w4; NEW=$BIN; echo "binary sha256 $(sha256sum $BIN | cut -c1-16)"

echo "== [F] f32 + KV32 diagnosis ($(el))"
mkdir -p $O/dump-f1a $O/dump-f1b
if srv_start g12-f1 $G12D $G12 "TITAN_G4_KV16=0" "${PC0[@]}" --dtype f32; then
  gate g1 f1a-g1 $GP
  gate g3 f1-g3; gate g3 f1-g3l 28000; echo "OOM lines: $(grep -c OUT_OF_MEMORY $SLOG)"
  gate g1 f1b-g1 $GP
  srv_stop
  tmap gemma4 $O/f1a-g1.json $O/f1b-g1.json $O/f1-g3.json > /dev/null
  cmpg g1 $O/f1a-g1.json $R/ref-g12-g1.json | head -1; cmpg g1 $O/f1b-g1.json $R/ref-g12-g1.json | head -1
  cmpg g3 $O/f1-g3.json $R/ref-g12-g3.json | head -1
fi
# per-layer dumps of G1 prompt 0, fresh vs after the 8k request
if srv_start g12-f1d $G12D $G12 "TITAN_G4_KV16=0 TITAN_G4_DUMP=$O/dump-f1a" "${PC0[@]}" --dtype f32; then
  timeout 120 python3 $I/dump/req1.py $PORT $I/dump/g1p0.txt
  srv_stop
fi
if srv_start g12-f1e $G12D $G12 "TITAN_G4_KV16=0 TITAN_G4_DUMP=$O/dump-f1b" "${PC0[@]}" --dtype f32; then
  timeout 300 python3 $I/dump/req1.py $PORT $I/dump/g3p0.txt > /dev/null; gate g3 f1x-g3l 28000 > /dev/null 2>&1
  timeout 120 python3 $I/dump/req1.py $PORT $I/dump/g1p0.txt
  srv_stop
fi
python3 $I/dump/cmpdump.py $O/dump-f1b $O/dump-f1a | head -12

echo "== [D] decode tok/s ($(el))"
dec() { # NAME BIN ENV DTYPE
  local n=$1; BIN=$2
  if srv_start g12-$n $G12D $G12 "$3" "${PC0[@]}" --dtype $4; then
    gate g5 $n-d0 2000 128; gate g5 $n-d1 13500 128; gate g5 $n-d2 28000 128; srv_stop; tps $O/$n-d0.json $O/$n-d1.json $O/$n-d2.json
  fi
}
dec nb $NEW "X=1" bf16
dec nbo $NEW "TITAN_G4_FLASH_DECODE=0" bf16
dec i2 $INTEG "X=1" bf16
dec nf $NEW "X=1" f32
BIN=$NEW
KIND=llama
if lsrv l-dec $G12D/$G12 -c 12288 -ngl 99 -ctk f16 -ctv f16; then
  gate g5 l-d0 2000 128; gate g5 l-d1 13500 128; gate g5 l-d2 28000 128; lstop; tps $O/l-d0.json $O/l-d1.json $O/l-d2.json
fi
KIND=titan

echo "== [G] gemma4 bf16 G1 (integ2 procedure) + G2 + tie; P2 G1 ($(el))"
if srv_start g12s $G12D $G12 "" --max-model-len 8192 --max-seq-len 8192 -n 0:48 --dtype bf16; then
  gate g1 nb-g1 $GP; gate g2 nb-g2 $GP 256; srv_stop
fi
tmap gemma4 $O/nb-g1.json > /dev/null; same $O/nb-g1.json $O/g12-g1-integ2.json; cmpg g1 $O/nb-g1.json $R/ref-g12-g1.json | head -1
same $O/nb-g2.json $O/b-g2.json; cmpg g2 $O/nb-g2.json $R/ref-g12-g2.json
if srv_start g12f $G12D $G12 "" --max-model-len 8192 --max-seq-len 8192 -n 0:48 --dtype f32; then
  gate g1 nf-g1 $GP; srv_stop; tmap gemma4 $O/nf-g1.json > /dev/null; cmpg g1 $O/nf-g1.json $R/ref-g12-g1.json | head -1
fi
KIND=llama
if lsrv l-tie $G12D/$G12 -c 8192 -ngl 99 -ctk f16 -ctv f16; then
  gate tie tie-nb $GP $O/nb-g2.json $R/ref-g12-g2.json; lstop
fi
KIND=titan

echo "== [RC] REDCELL G1 + decode ($(el))"
RD=$M/redcell-26b; RF=REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf
if srv_start red-nb $RD $RF "" --max-model-len 8192 --max-seq-len 8192 -n 0:30 --dtype bf16 --prefix-cache-n 0; then
  gate g1 rnb-g1 $G4/prompts/redcell.json; gate g5 rnb-d1 13500 128; srv_stop
  tmap gemma4 $O/rnb-g1.json > /dev/null; cmpg g1 $O/rnb-g1.json $R/ref-red-g1.json | head -1; tps $O/rnb-d1.json
fi

echo "== [S] Spark identity ($(el))"
if srv_start sp $M/spark-x2.5 Spark-X2.5-4B-Q4_K_M.gguf "" --max-model-len 16384 --max-seq-len 16384; then
  for c in 2000 15000 33000; do gate g5 sp4-g5-$c $c 128; done
  gate g3 sp4-g3; gate g1 sp4-g1 $G4/prompts/spark.json; gate g2 sp4-g2 $G4/prompts/spark.json 256; srv_stop
  tmap spark $O/sp4-g1.json $O/sp4-g3.json > /dev/null
  same $O/sp4-g3.json $O/sp-g3-integ2.json; same $O/sp4-g1.json $O/sp-g1-integ2.json; same $O/sp4-g2.json $O/sp-g2-integ2.json
  cmpg g1 $O/sp4-g1.json $R/refc-spark-g1.json | head -1
fi

echo "== [R] regression core ($(el))"
coll g4acc4-m2 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2 && cmp40 $E/m6/out/h-off.json $E/m3/out/g4acc4-m2.json
coll g4acc4-mf0 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=0 && cmp40 $E/m6/out/mtpfile-off.json $E/m3/out/g4acc4-mf0.json
if srv_start b-off $M/bonsai2-27b Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf "TITAN_MTP=0 TITAN_REASONING_EFFORT=medium" --max-seq-len 65536; then
  $C $CL greedy titan $PORT b4-off 8 300; srv_stop; $C $CL cmp o-off b4-off
fi
if srv_start i-new $M/q35-lowbit Qwen3.6-35B-A3B-UD-IQ2_M.gguf "TITAN_MTP=2 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64" --max-seq-len 65536; then
  $C $CL greedy titan $PORT i4-new 8 300; srv_stop; $C $CL cmp i-integ2 i4-new
fi
if srv_start q-new $M/bonsai/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf "TITAN_ATTN_FLASH_PREFILL=1" --max-seq-len 32768; then
  $C $CL greedy titan $PORT q4-new 8 300; srv_stop; $C $CL cmp q-integ2 q4-new
fi
free -m | head -2
