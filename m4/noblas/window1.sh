#!/bin/bash
# noblas window 1 (phase 1, inventory): [B] build the inventory binary (model-swap ddfe94d8f + TITAN_GEMM_LOG logging in
# candle's cuBLAS matmul, mistralrs-quant's cuBLASLt matmul and candle's cuRAND calls; cuBLAS itself unchanged);
# [I] per roster model: nsys (cuda + cublas API trace) around a single-model from-config server (the deployed roster
# block, CUDA graphs off so every decode step reaches the dispatch) serving a warm-up, a short chat prompt (32 tokens)
# and a ~4k-token prompt (16 tokens), with TITAN_GEMM_LOG and phase markers.
source $HOME/titan-engine/top-noblas/m4/noblas/lib.sh
restic_wait
exec > >(tee -a $I/window1-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin noblas-w1 58
echo "src: mistral.rs $(git -C $SRC log --oneline -1 | cut -c1-60) $(git -C $SRC diff --quiet && echo clean || echo DIRTY); candle $(git -C $E/top-noblas log --oneline -1 | cut -c1-60)"
until mbuild 1500; do waitfix mbuild 1500 || { echo "no binary: ending the window"; exit 1; }; done
cp $MBIN $O/mistralrs-inv; BIN=$O/mistralrs-inv; sha256sum $BIN | tee $O/mistralrs-inv.sha256
ldd $BIN | grep -E "cublas|curand"
mkdir -p $O/toml; python3 $I/mktoml.py $O/toml $PORT --no-graphs > /dev/null
for m in qwen3.6-35b qwen3-next-80b gpt-oss-120b gpt-oss-20b qwen3-14b spark-x2.5 gemma4-12b redcell-26b bonsai-27b \
         bonsai2-27b orcasaq2-cyber-27b qwen3.6-35b-iq2m qwen3.6-35b-mxfp4; do
  [ $(left) -gt 300 ] || { echo "!! $m skipped: $(left)s left"; continue; }
  echo "== [I] $m ($(el), $(left)s left)"
  rm -f $O/inv-$m.glog $O/inv-$m.nsys-rep
  mkdir -p $O/nsys-tmp; if SENV="TITAN_GEMM_LOG=$O/inv-$m.glog TMPDIR=$O/nsys-tmp" KSIG=SIGINT KTO=300 cfg_start inv-$m $O/toml/$m.toml \
       $NSYS profile -t cuda,cublas --sample=none --cpuctxsw=none --cuda-graph-trace=node -o $O/inv-$m -f true; then
    timeout 900 python3 $I/inv.py $PORT $m $O/inv-$m.glog $O/inv-$m.json
    srv_stop
  fi
  ls -la $O/inv-$m.nsys-rep 2>&1 | cut -c1-200; echo "glog: $(grep -c '^[GLR] ' $O/inv-$m.glog 2>/dev/null) calls"
  grep -v "^M " $O/inv-$m.glog 2>/dev/null | cut -d' ' -f1-6 | sort | uniq -c | sort -rn | head -12
done
free -m | head -2
echo "== noblas-w1 end $(el)"
