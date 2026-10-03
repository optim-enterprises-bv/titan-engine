#!/bin/bash
# g4acc window 1: [K] flash-prefill r2 / f16-o32 kernels build + gate; [B] build; [E] gemma4-12b G3 ~4k/~8k + G1 per
# precision piece: P0 bf16 (flash, r2 sliding, no host masks), P1 --dtype f32 with F32 cache + eager F32 attention
# (TITAN_G4_KV16=0), P2 --dtype f32 + F16 cache + f16 flash (llama.cpp layout); KTRACE of P2's kernels (Q4_0 dequant?);
# VRAM, prefill, decode; [D] P2 last-token dump vs llama.cpp; [RC] REDCELL G3 P0 / P2; [S] Spark integ2 procedure.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/g4acc/window1.sh
set -u
source $HOME/titan-engine/m4/g4acc/lib.sh
exec > >(tee -a $I/window1-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin window1 58
K=$OX/flash-prefill; G12D=$M/gemma4-12b-qat; G12=gemma-4-12b-it-qat-q4_0.gguf
same() { python3 $I/same.py "$@"; }
tps() { for f in "$@"; do python3 -c "import json,sys;r=json.load(open(sys.argv[1]));print('TPS',sys.argv[1].split('/')[-1],[(x['prompt_tokens'],round(x['prefill_tps'] or 0,1),round(x['decode_tps'] or 0,1)) for x in r])" $f; done; }
CTX=(--max-model-len 12288 --max-seq-len 12288 -n 0:48 --prefix-cache-n 0)
echo "src: $(git -C $SRC log --oneline -1 | cut -c1-60) $(git -C $SRC diff --quiet && echo clean || echo DIRTY); oxide $(git -C $OX diff --quiet && echo clean || echo DIRTY)"

echo "== [K] kernel build + gate ($(el))"
KOK=0
for try in 1 2 3; do
  if oxbuild flash-prefill g4acc-ox 600 && [ $K/flash_prefill.ptx -nt $K/src/main.rs ]; then
    (cd $K && FP_TIME=1 timeout 900 ./target/release/flash-prefill > $O/gate-fp.log 2>&1); echo "gate rc=$?"
    grep -vE "^  (flash|time)" $O/gate-fp.log | tail -45; grep -E "^  time .*(d512|r2|s 4096 kvh 2)" $O/gate-fp.log | head -40
    grep -q "^flash-prefill gate: PASS" $O/gate-fp.log && KOK=1
    break
  fi
  [ $try = 3 ] && break
  echo "kernel build failed: waiting for $I/kbuild.flag ($(left)s left)"; date -Is > $O/kbuild-failed.flag
  while [ ! -e $I/kbuild.flag ] && [ $(left) -gt 2400 ]; do sleep 10; done
  [ -e $I/kbuild.flag ] || break
  rm -f $I/kbuild.flag $O/kbuild-failed.flag
done
echo "kernel gate ok=$KOK"
grep -q "flash_prefill_r2_w8_f16o32" $K/flash_prefill.ptx && bash $K/export_ptx.sh $SRC || echo "!! no new PTX"

echo "== [B] mistral.rs build ($(el))"
rm -f $I/rebuild.flag
until mbuild 1500; do
  echo "build failed: waiting for $I/rebuild.flag ($(left)s left)"; date -Is > $O/build-failed.flag
  while [ ! -e $I/rebuild.flag ] && [ $(left) -gt 1500 ]; do sleep 10; done
  [ -e $I/rebuild.flag ] || { echo "no binary: ending the window"; exit 1; }
  rm -f $I/rebuild.flag $O/build-failed.flag; echo "rebuilding ($(el))"
done
cp $MBIN $I/bin/mistralrs-w1; BIN=$I/bin/mistralrs-w1; echo "binary sha256 $(sha256sum $BIN | cut -c1-16)"

echo "== [E] gemma4-12b precision pieces ($(el))"
run_mode() { # NAME DTYPE "ENV"
  local n=$1 dt=$2 env=$3
  if srv_start g12-$n $G12D $G12 "$env" "${CTX[@]}" --dtype $dt; then
    gate g3 $n-g3; gate g3 $n-g3l 28000; echo "VRAM after G3 8k: $(smi); OOM $(grep -c OUT_OF_MEMORY $SLOG)"
    gate g1 $n-g1 $G4/prompts/gemma4-12b.json; gate g5 $n-g5 13500 128
    srv_stop
    grep -E "titan gemma4 attention" $SLOG | sed 's/.*INFO //' | head -1; grep "^KTRACE" $SLOG | sort > $O/$n.ktrace; grep -ci "dequant" $O/$n.ktrace
    tmap gemma4 $O/$n-g3.json $O/$n-g3l.json $O/$n-g1.json > /dev/null
    cmpg g3 $O/$n-g3.json $R/ref-g12-g3.json | head -1; cmpg g3 $O/$n-g3l.json $R/ref-g12-g3l.json | head -1
    cmpg g1 $O/$n-g1.json $R/ref-g12-g1.json | head -1
    tps $O/$n-g3.json $O/$n-g3l.json $O/$n-g5.json
  else srv_stop; fi
}
run_mode p2 f32 "TITAN_KTRACE=1"
run_mode p0 bf16 "X=1"
run_mode p1 f32 "TITAN_G4_KV16=0"
grep -i "dequant\|q4_0\|Q4_0" $O/p2.ktrace | head

echo "== [D] P2 dump vs llama.cpp f16-FA (hd512's llama dumps) ($(el))"
mkdir -p $O/dump-p2
if srv_start g12-dp2 $G12D $G12 "TITAN_G4_DUMP=$O/dump-p2" "${CTX[@]}" --dtype f32; then
  timeout 300 python3 $I/dump/req1.py $PORT $I/dump/g3p0.txt; srv_stop
fi
python3 $I/dump/cmpdump.py $O/dump-p2 $E/m4/hd512/out/dump-lf16 $E/m4/hd512/out/dump-lbf16 | awk 'NR==1 || $1 % 6 == 5 || $1 < 3 || $1 > 44'

echo "== [RC] REDCELL G3 ($(el))"
RD=$M/redcell-26b; RF=REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf
for v in "rp2 f32" "rp0 bf16"; do
  set -- $v
  if [ $(left) -gt 300 ] && srv_start red-$1 $RD $RF "" --max-model-len 8192 --max-seq-len 8192 -n 0:30 --dtype $2 --prefix-cache-n 0; then
    gate g3 $1-g3; [ $1 = rp2 ] && gate g1 $1-g1 $G4/prompts/redcell.json; echo "VRAM: $(smi)"; srv_stop
    tmap gemma4 $O/$1-g3.json > /dev/null; cmpg g3 $O/$1-g3.json $R/ref-red-g3.json | head -1; tps $O/$1-g3.json
  else srv_stop; fi
done
[ -e $O/rp2-g1.json ] && tmap gemma4 $O/rp2-g1.json > /dev/null && cmpg g1 $O/rp2-g1.json $R/ref-red-g1.json | head -1

echo "== [S] Spark integ2 procedure ($(el))"
if [ $(left) -gt 300 ] && srv_start sp $M/spark-x2.5 Spark-X2.5-4B-Q4_K_M.gguf "" --max-model-len 16384 --max-seq-len 16384; then
  for c in 2000 15000 33000; do gate g5 sp-g5-$c $c 128; done
  gate g3 sp-g3; gate g1 sp-g1 $G4/prompts/spark.json; gate g2 sp-g2 $G4/prompts/spark.json 256; srv_stop
  tmap spark $O/sp-g1.json $O/sp-g3.json > /dev/null
  same $O/sp-g3.json $O/sp-g3-integ2.json; same $O/sp-g1.json $O/sp-g1-integ2.json; same $O/sp-g2.json $O/sp-g2-integ2.json
fi
free -m | head -2
