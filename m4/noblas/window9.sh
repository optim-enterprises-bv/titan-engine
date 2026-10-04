#!/bin/bash
# noblas window 9 (round 3): the 35B's G3 flip vs reduction order. noblas2, deployed 35B roster block, G3 (and G1)
# with TITAN_GEMM_REF routing call classes to the f64-accumulating reference kernel: none, b1 (router / gates /
# GDN projections: every unbatched product), bN (attention batches), all.
source $HOME/titan-engine/top-noblas/m4/noblas/lib.sh
restic_wait
exec > >(tee -a $I/window9-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin noblas-w9 58
BIN=$E/bin/mistralrs-titan-noblas2
sha256sum $BIN
for v in none b1 bN all; do
  if SENV="TITAN_GEMM_REF=$v" cfg_start q$v $O/tomlg/qwen3.6-35b.toml; then
    gm qwen3.6-35b g3 q35ref-$v-g3 13500; gm qwen3.6-35b g1 q35ref-$v-g1 $I/plain40.json; srv_stop; fi
done
echo "== noblas-w9 end $(el)"
