#!/bin/bash
# noblas window 7 (round 2, localise REDCELL): gemm crate + f64 reference kernel (gate), export, build; then REDCELL
# (deployed roster block, standalone from-config) G1 / G3 with TITAN_GEMM_REF routing one call class at a time to the
# f64-accumulating reference kernel: none, all, b1 (router: the only unbatched product), bN (attention batches).
source $HOME/titan-engine/top-noblas/m4/noblas/lib.sh
restic_wait
exec > >(tee -a $I/window7-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin noblas-w7 58
gemm_build || exit 1
until kgate $OX gemm $O/gate-w7.log 1200; do grep -E "FAIL|SURVIVED|panicked" $O/gate-w7.log | head; waitfix gate 1500 || exit 1; gemm_build || exit 1; done
grep -E -- "-> (PASS|FAIL)|worst error" $O/gate-w7.log | cut -c1-200
bash $OX/gemm/export.sh $E/top-noblas/candle || exit 1
until mbuild 1500; do waitfix mbuild 900 || exit 1; done
cp $MBIN $O/mistralrs-noblas-w7; BIN=$O/mistralrs-noblas-w7
mkdir -p $O/tomlg; python3 $I/mktoml.py $O/tomlg $PORT > /dev/null
RP=$E/m4/g4s/prompts/redcell.json
for v in none all b1 bN; do
  rm -f $O/ab-$v.glog
  if SENV="TITAN_GEMM_REF=$v TITAN_GEMM_LOG=$O/ab-$v.glog" cfg_start ab-$v $O/tomlg/redcell-26b.toml; then
    gm redcell-26b g1 ab-$v-g1 $RP
    [ $v = none ] || [ $v = all ] && gm redcell-26b g3 ab-$v-g3 13500
    srv_stop
  fi
  grep "^X " $O/ab-$v.glog | awk '{print $2}' | sort | uniq -c | sort -rn | head -6
done
echo "== noblas-w7 end $(el)"
