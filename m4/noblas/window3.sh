#!/bin/bash
# noblas window 3 (phase 1, kernel tuning loop): build the gemm crate (straight-line unrolled loops: the accumulators
# were in local memory), gate, bench on every inventoried shape; then serve flags while time allows:
#   $I/flag.gemm  rebuild + gate + bench (shapes: $I/bench-shapes, default out/shapes-all.txt)
#   $I/flag.full  export the kernels to candle, build the noblas mistral.rs binary, smoke it (Spark, 35B short prompts)
#   $I/flag.done  end the window
source $HOME/titan-engine/top-noblas/m4/noblas/lib.sh
restic_wait
exec > >(tee -a $I/window3-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin noblas-w3 58
rm -f $I/flag.*
round=0
gemm_round() {
  round=$((round + 1))
  gemm_build || { echo "!! gemm build failed"; return 1; }
  kgate $OX gemm $O/gate-w3-$round.log 900
  grep -E -- "-> (PASS|FAIL)|worst error|SURVIVED|^FAIL|coverage|local" $O/gate-w3-$round.log | grep -v " 0 B local" | head -40 | cut -c1-250
  local sh=$(cat $I/bench-shapes 2>/dev/null || echo $O/shapes-all.txt)
  kgate $OX gemm $O/bench-w3-$round.log 900 GEMM_BENCH=$sh
  sed -n '/^per model/,$p' $O/bench-w3-$round.log | head -50
}
full_round() {
  bash $OX/gemm/export.sh $E/top-noblas/candle || return 1
  until mbuild 1500; do waitfix mbuild 900 || return 1; done
  cp $MBIN $O/mistralrs-noblas-w3; BIN=$O/mistralrs-noblas-w3
  ldd $BIN | grep -E "cublas|curand|cuda"; echo "nm cublas/curand symbols: $(nm -D $BIN | grep -ciE 'cublas|curand')"
  mkdir -p $O/tomlg; python3 $I/mktoml.py $O/tomlg $PORT > /dev/null
  for m in spark-x2.5 qwen3.6-35b; do
    [ $(left) -gt 300 ] || break
    rm -f $O/smoke-$m.glog
    if SENV="TITAN_GEMM_LOG=$O/smoke-$m.glog" cfg_start smoke-$m $O/tomlg/$m.toml; then
      timeout 600 python3 $I/inv.py $PORT $m $O/smoke-$m.glog $O/smoke-$m.json
      srv_stop
    fi
    grep "^X " $O/smoke-$m.glog | awk '{print $2, $3}' | sort | uniq -c | sort -rn | head -8
  done
}
gemm_round
while [ $(left) -gt 600 ]; do
  [ -e $I/flag.done ] && break
  if [ -e $I/flag.gemm ]; then rm -f $I/flag.gemm; echo "== flag.gemm ($(el))"; gemm_round; continue; fi
  if [ -e $I/flag.full ]; then rm -f $I/flag.full; echo "== flag.full ($(el))"; full_round; continue; fi
  sleep 10
done
echo "== noblas-w3 end $(el)"
