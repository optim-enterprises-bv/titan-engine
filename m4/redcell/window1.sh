#!/bin/bash
# redcell window 1 (of 3):
#  [1] oxide-redcell mistralrs-quant-a build (deterministic moe_gemv_down_aggregate, stable moe_dispatch scatter)
#      || integ binary: TITAN_G4_MEMLOG=2 probes of one G1 prompt (8 tokens) on two fresh servers + twice on one
#  [2] quant-a gate vs libmistralrsquant.a (+ slot-order fold and stable-dispatch checks)
#  [3] ncu of the integ binary: one ~3.5k-token prompt (G5 prompt 0), the profile-first baseline
#  [4] mistral.rs build (top-redcell/mr-redcell: unaligned-K prompts -> grouped MMQ pieces, IQ4_NL MoE MMQ)
#  [5] determinism: 3 fresh servers, baseline procedure (default prefix cache, G1 -> G3 -> G2 256)
#  [6] G5 3.5k new vs TITAN_MOE_UNALIGNED_DECODE=1 (the a7ea678f9 route) on the same binary; G1/G3/G2 vs llama.cpp
#  [7] ncu of the new binary (same prompt); [8] gemma4-12b G1 vs integ
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/redcell/window1.sh
source $HOME/titan-engine/m4/redcell/lib.sh
exec > >(tee -a $I/window1-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin redcell-window1 58
echo "src: $(git -C $SRC log --oneline -1 | cut -c1-70) $(git -C $SRC diff --stat | tail -1)"
echo "oxide: $(git -C $OX log --oneline -1 | cut -c1-60) $(git -C $OX diff --stat | tail -1)"

echo "== [1] quant-a build || integ MEMLOG probes ($(el))"
oxbuild mistralrs-quant-a red-ox 900 > $O/oxbuild-qa.status 2>&1 &
OXP=$!
BIN=$INTEG
for rep in a b; do
  if srv_start ev-integ-$rep $M/redcell-26b $RED "TITAN_G4_MEMLOG=2" "${RA[@]}" --prefix-cache-n 0; then
    timeout 300 python3 $I/one.py $PORT $RP 0 8
    [ $rep = a ] && timeout 300 python3 $I/one.py $PORT $RP 0 8
    srv_stop
  fi
done
python3 $I/firstdiff.py $O/ev-integ-a.server.log --halves
python3 $I/firstdiff.py $O/ev-integ-a.server.log $O/ev-integ-b.server.log
wait $OXP; cat $O/oxbuild-qa.status
until [ $OX/mistralrs-quant-a/mistralrs_quant_a.ptx -nt $OX/mistralrs-quant-a/src/main.rs ]; do
  echo "quant-a build failed: waiting for $I/oxretry.flag ($(left)s left)"; date -Is > $I/out/oxbuild-failed.flag
  while [ ! -e $I/oxretry.flag ] && [ $(left) -gt 2400 ]; do sleep 10; done
  [ -e $I/oxretry.flag ] || { echo "no quant-a PTX: ending the window"; exit 1; }
  rm -f $I/oxretry.flag $I/out/oxbuild-failed.flag
  oxbuild mistralrs-quant-a red-ox 900
done
grep -c "moe_dispatch_scatter_stable_kernel" $OX/mistralrs-quant-a/mistralrs_quant_a.ptx

echo "== [2] quant-a gate ($(el))"
(cd $OX/mistralrs-quant-a && timeout 900 ./target/release/mistralrs-quant-a > $O/gate-qa.log 2>&1); echo "gate rc=$?"
grep -E "slot-order|stable|launcher calls|order-only|FAIL|PASS" $O/gate-qa.log | head -30

echo "== [3] ncu integ binary, ~3.5k prompt ($(el))"
BIN=$INTEG
if ncu_start ncu-integ "" -m $M/redcell-26b -f $RED "${RA[@]}" --prefix-cache-n 0; then
  timeout 600 python3 $SP/req.py $PORT titan 200 1; timeout 900 python3 $SP/req.py $PORT titan 13500 1
fi
ncu_end

echo "== [4] mistral.rs build ($(el))"
until mbuild 1500; do
  [ $(left) -gt 1500 ] || { echo "no binary: ending the window"; exit 1; }
  echo "build failed: waiting for $I/rebuild.flag ($(left)s left)"; date -Is > $I/out/build-failed.flag
  while [ ! -e $I/rebuild.flag ] && [ $(left) -gt 1500 ]; do sleep 10; done
  [ -e $I/rebuild.flag ] || { echo "no binary: ending the window"; exit 1; }
  rm -f $I/rebuild.flag $I/out/build-failed.flag
done
cp $MBIN $NEWBIN; echo "binary sha256 $(sha256sum $NEWBIN | cut -c1-16)"
BIN=$NEWBIN

echo "== [5] determinism: 3 fresh servers, G1 -> G3 -> G2 ($(el))"
for rep in a b c; do
  if srv_start rd-$rep $M/redcell-26b $RED "" "${RA[@]}"; then
    gate g1 rd-$rep-g1 $RP; gate g3 rd-$rep-g3 2>&1 | tail -1; gate g2 rd-$rep-g2 $RP 256
    [ $rep = a ] && gate g5 rd-$rep-g5 13500 128
    srv_stop
  fi
done
for p in "a b" "a c"; do set -- $p
  for g in g1 g2 g3; do [ -e $O/rd-$1-$g.json ] && [ -e $O/rd-$2-$g.json ] && same $O/rd-$1-$g.json $O/rd-$2-$g.json; done
done

echo "== [6] G5: TITAN_MOE_UNALIGNED_DECODE=1 on the same binary; accuracy vs llama.cpp ($(el))"
if srv_start rd-dec $M/redcell-26b $RED "TITAN_MOE_UNALIGNED_DECODE=1" "${RA[@]}"; then gate g5 rd-dec-g5 13500 128; gate g3 rd-dec-g3 2>&1 | tail -1; srv_stop; fi
[ -e $O/rd-a-g1.json ] && { tmap gemma4 $O/rd-a-g1.json; cmpg g1 $O/rd-a-g1.json $R/ref-red-g1.json; }
[ -e $O/rd-a-g3.json ] && { tmap gemma4 $O/rd-a-g3.json; cmpg g3 $O/rd-a-g3.json $R/ref-red-g3.json; }
[ -e $O/rd-dec-g3.json ] && { tmap gemma4 $O/rd-dec-g3.json; cmpg g3 $O/rd-dec-g3.json $R/ref-red-g3.json; }
[ -e $O/rd-a-g2.json ] && cmpg g2 $O/rd-a-g2.json $R/ref-red-g2.json | head -2

echo "== [7] ncu new binary, ~3.5k prompt ($(el))"
if [ $(left) -gt 900 ] && ncu_start ncu-new "" -m $M/redcell-26b -f $RED "${RA[@]}" --prefix-cache-n 0; then
  timeout 600 python3 $SP/req.py $PORT titan 200 1; timeout 900 python3 $SP/req.py $PORT titan 13500 1
  ncu_end
fi

echo "== [8] gemma4-12b G1 vs integ ($(el))"
if srv_start g12 $M/gemma4-12b-qat gemma-4-12b-it-qat-q4_0.gguf "" --max-model-len 8192 --max-seq-len 8192 -n 0:48 --dtype bf16; then
  gate g1 g12-g1 $G4/prompts/gemma4-12b.json; srv_stop
  same $O/g12-g1.json $E/m4/integ/out/g12-g1.json
fi
echo "== window1 done ($(el))"
