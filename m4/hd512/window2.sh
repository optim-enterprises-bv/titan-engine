#!/bin/bash
# hd512 window 2: [K] flash-prefill kernels (d512 bf16 / f16, d256 f16) build + gate (+ FP_TIME); export PTX;
# [B] mistral.rs build (gemma4 prompt attention routing + precision switches + TITAN_G4_DUMP);
# [E] gemma4-12b G3 ~4k / ~8k (+ G1) per mode: E0 flash off (= integ), E1 flash bf16, E2 F32 inputs + F32 eager,
#     E3 F32 inputs + f16 flash, E4 F32 inputs + bf16 flash; [D] last-token dumps titan E0 / E3 vs llama.cpp f16-FA
#     and bf16-KV on G3 prompt 0; [S] Spark G1 / G3 identity vs integ; [RC] REDCELL G3 E1 / E3.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/hd512/window2.sh
set -u
source $HOME/titan-engine/m4/hd512/lib.sh
exec > >(tee -a $I/window2-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin window2 58
K=$OX/flash-prefill; G12D=$M/gemma4-12b-qat; G12=gemma-4-12b-it-qat-q4_0.gguf; G12F=$G12D/$G12
same() { python3 $I/same.py "$@"; }
tps() { for f in "$@"; do python3 -c "import json,sys;r=json.load(open(sys.argv[1]));print('TPS',sys.argv[1].split('/')[-1],[(x['prompt_tokens'],round(x['prefill_tps'] or 0,1)) for x in r])" $f; done; }
CTX=(--max-model-len 12288 --max-seq-len 12288 -n 0:48 --dtype bf16 --prefix-cache-n 0)
echo "src: $(git -C $SRC log --oneline -1 | cut -c1-70) $(git -C $SRC diff --quiet && echo clean || echo DIRTY); oxide $(git -C $OX log --oneline -1 | cut -c1-50) $(git -C $OX diff --quiet && echo clean || echo DIRTY)"

echo "== [K] kernel build + gate ($(el))"
KOK=0
for try in 1 2 3; do
  if oxbuild flash-prefill hd512-ox 600 && [ $K/flash_prefill.ptx -nt $K/src/main.rs ]; then
    grep -E "not unrolled|warning: unused" $O/oxbuild-flash-prefill.log | sort | uniq -c | head -5
    (cd $K && FP_TIME=1 timeout 600 ./target/release/flash-prefill > $O/gate-fp.log 2>&1); echo "gate rc=$?"
    grep -vE "^  (flash|time)" $O/gate-fp.log | tail -30; grep -E "^  time" $O/gate-fp.log | head -40
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
if [ -f $K/flash_prefill.ptx ] && grep -q "flash_prefill_d512_w8_f16" $K/flash_prefill.ptx; then
  bash $K/export_ptx.sh $SRC
else
  echo "!! no d512 PTX: flash modes will fail"
fi

echo "== [B] mistral.rs build ($(el))"
rm -f $I/rebuild.flag
until mbuild 1500; do
  echo "build failed: waiting for $I/rebuild.flag ($(left)s left)"; date -Is > $O/build-failed.flag
  while [ ! -e $I/rebuild.flag ] && [ $(left) -gt 1500 ]; do sleep 10; done
  [ -e $I/rebuild.flag ] || { echo "no binary: ending the window"; exit 1; }
  rm -f $I/rebuild.flag $O/build-failed.flag; echo "rebuilding ($(el))"
done
cp $MBIN $I/bin/mistralrs-w2; BIN=$I/bin/mistralrs-w2; echo "binary sha256 $(sha256sum $BIN | cut -c1-16)"

echo "== [E] gemma4-12b modes ($(el))"
run_mode() { # NAME "ENV" G1?
  local n=$1 env=$2
  if srv_start g12-$n $G12D $G12 "$env" "${CTX[@]}"; then
    gate g3 $n-g3; gate g3 $n-g3l 28000
    [ "${3:-}" = g1 ] && gate g1 $n-g1 $G4/prompts/gemma4-12b.json
    srv_stop
    grep -E "titan gemma4 prompt attention|flash-prefill" $SLOG | sed 's/.*INFO //' | head -2
    tmap gemma4 $O/$n-g3.json $O/$n-g3l.json $( [ -e $O/$n-g1.json ] && echo $O/$n-g1.json ) > /dev/null
    cmpg g3 $O/$n-g3.json $R/ref-g12-g3.json | head -1; cmpg g3 $O/$n-g3l.json $R/ref-g12-g3l.json | head -1
    [ -e $O/$n-g1.json ] && cmpg g1 $O/$n-g1.json $R/ref-g12-g1.json | head -1
    tps $O/$n-g3.json $O/$n-g3l.json
  else srv_stop; fi
}
run_mode e0 "TITAN_G4_ATTN_FLASH=0" g1
same $O/e0-g3.json $O/t0-g3.json; same $O/e0-g3l.json $O/t0-g3l.json; same $O/e0-g1.json $E/m4/integ/out/g12-g1.json
run_mode e1 "TITAN_G4_ATTN_FLASH=1" g1
same $O/e1-g1.json $E/m4/integ/out/g12-g1.json
run_mode e3 "TITAN_G4_ATTN_IN=f32 TITAN_G4_ATTN_KV=f16" g1
run_mode e2 "TITAN_G4_ATTN_IN=f32 TITAN_G4_ATTN_KV=f32"
run_mode e4 "TITAN_G4_ATTN_IN=f32 TITAN_G4_ATTN_KV=bf16"

echo "== [D] last-token dumps, G3 prompt 0 ($(el))"
for v in "e0 TITAN_G4_ATTN_FLASH=0" "e3 TITAN_G4_ATTN_IN=f32 TITAN_G4_ATTN_KV=f16" "e1 TITAN_G4_ATTN_FLASH=1"; do
  set -- $v; n=$1; shift; mkdir -p $O/dump-$n
  if [ $(left) -gt 400 ] && srv_start g12-d$n $G12D $G12 "$* TITAN_G4_DUMP=$O/dump-$n" "${CTX[@]}"; then
    timeout 300 python3 $I/dump/req1.py $PORT $I/dump/g3p0.txt; srv_stop
  fi
done
for v in "lf16 1 f16" "lbf16 1 bf16" "lnofa 0 f16"; do
  set -- $v; mkdir -p $O/dump-$1
  [ $(left) -gt 300 ] && timeout 300 systemd-run --user --unit=hd512-llama --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 \
    $I/dump/g4dump $G12F $I/dump/g3p0.txt $O/dump-$1 99 $2 $3 '^(kqv_out|attn_out|l_out)-[0-9]+$' 2>&1 | grep -E "g4dump|error" | tail -2
done
for n in e0 e3 e1; do echo "-- dump $n vs llama f16-FA (and llama bf16-KV vs f16-FA)"; python3 $I/dump/cmpdump.py $O/dump-$n $O/dump-lf16 $O/dump-lbf16; done
echo "-- llama FA-off vs f16-FA"; python3 $I/dump/cmpdump.py $O/dump-lnofa $O/dump-lf16 | head -60

echo "== [S] Spark identity vs integ ($(el))"
SPD=$M/spark-x2.5; SPF=Spark-X2.5-4B-Q4_K_M.gguf
if [ $(left) -gt 300 ] && srv_start sp $SPD $SPF "" --max-model-len 16384 --max-seq-len 16384; then
  # integ's procedure (window2 [1]): g5 x3 on the same server first (G3 prompt 0 then hits the prefix cache)
  for c in 2000 15000 33000; do gate g5 sp-g5-$c $c 128; done
  gate g3 sp-g3; gate g1 sp-g1 $G4/prompts/spark.json
  srv_stop
  tmap spark $O/sp-g1.json $O/sp-g3.json > /dev/null
  same $O/sp-g3.json $E/m4/integ/out/sp-g3.json; same $O/sp-g1.json $E/m4/integ/out/sp-g1.json
  tps $O/sp-g5-15000.json $O/sp-g5-33000.json
fi

echo "== [RC] REDCELL G3 ($(el))"
RD=$M/redcell-26b; RF=REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf
for v in "r1 TITAN_G4_ATTN_FLASH=1" "r3 TITAN_G4_ATTN_IN=f32 TITAN_G4_ATTN_KV=f16" "r0 TITAN_G4_ATTN_FLASH=0"; do
  set -- $v; n=$1; shift
  if [ $(left) -gt 300 ] && srv_start red-$n $RD $RF "$*" --max-model-len 8192 --max-seq-len 8192 -n 0:30 --dtype bf16 --prefix-cache-n 0; then
    gate g3 $n-g3; srv_stop; tmap gemma4 $O/$n-g3.json > /dev/null; cmpg g3 $O/$n-g3.json $R/ref-red-g3.json | head -1; tps $O/$n-g3.json
  else srv_stop; fi
done
free -m | head -2
