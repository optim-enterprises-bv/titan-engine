#!/bin/bash
# noblas window 2 (phase 1, kernels): [K] build the oxide gemm crate (oxide-noblas/gemm), run its gate (accuracy vs
# f64, mutations, Philox, coverage) and its bench (every inventoried shape vs cuBLAS); fix loop on failure.
# [T] if time is left: nsys timing (100 ms flush, deployed roster blocks with CUDA graphs) of the deployed binary on
# the models whose cuBLAS share matters, for the GPU-time shares.
source $HOME/titan-engine/top-noblas/m4/noblas/lib.sh
restic_wait
exec > >(tee -a $I/window2-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin noblas-w2 58
echo "src: oxide-kernels $(git -C $OX log --oneline -1 | cut -c1-60) $(git -C $OX status --short gemm | wc -l) changed files in gemm/"
gemm_build || { echo "no gemm build: ending the window"; exit 1; }
while :; do
  kgate $OX gemm $O/gate-w2.log 900 && break
  grep -E "FAIL|SURVIVED|panicked|error" $O/gate-w2.log | head -40 | cut -c1-300
  waitfix gate 1200 || break
  gemm_build || break
done
grep -E "PASS|FAIL|coverage|JIT|registers" $O/gate-w2.log | head -80 | cut -c1-300
[ $(left) -gt 600 ] && kgate $OX gemm $O/bench-w2.log 900 GEMM_BENCH=$O/shapes-all.txt
tail -45 $O/bench-w2.log
if [ $(left) -gt 900 ]; then
  echo "== [T] nsys timing of the deployed binary ($(el))"
  mkdir -p $O/tomlg; python3 $I/mktoml.py $O/tomlg $PORT > /dev/null
  BIN=$OLDBIN
  for m in qwen3.6-35b gpt-oss-20b redcell-26b bonsai2-27b gpt-oss-120b orcasaq2-cyber-27b qwen3-next-80b; do
    [ $(left) -gt 300 ] || break
    nsys_run tim-$m $O/tomlg/$m.toml $m
  done
fi
echo "== noblas-w2 end $(el)"
