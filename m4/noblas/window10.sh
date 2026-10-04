#!/bin/bash
# noblas window 10 (round 3): build noblas3 (TITAN_GEMM_REF grid guard only; default path unchanged), stage it, proof;
# 35B G3 / G1 default (identity vs noblas2) and with TITAN_GEMM_REF=b1 / f32 / all (f64-accumulated router, gates,
# GDN projections); nsys proof on the 35B.
source $HOME/titan-engine/top-noblas/m4/noblas/lib.sh
restic_wait
exec > >(tee -a $I/window10-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin noblas-w10 58
until mbuild 1500; do waitfix mbuild 900 || exit 1; done
cp $MBIN $E/bin/mistralrs-titan-noblas3; (cd $E/bin && sha256sum mistralrs-titan-noblas3 > mistralrs-titan-noblas3.sha256 && cat mistralrs-titan-noblas3.sha256)
BIN=$E/bin/mistralrs-titan-noblas3
echo "ldd cublas/curand: $(ldd $BIN | grep -ciE 'cublas|curand'), nm -D: $(nm -D $BIN | grep -ciE 'cublas|curand'), strings API names: $(strings $BIN | grep -oE '(cublas|cublasLt|curand)[A-Z][A-Za-z_0-9]*' | sort -u | wc -l)"
for v in none b1 f32 all; do
  if SENV="TITAN_GEMM_REF=$v" cfg_start q3$v $O/tomlg/qwen3.6-35b.toml; then
    gm qwen3.6-35b g3 q35r3-$v-g3 13500; gm qwen3.6-35b g1 q35r3-$v-g1 $I/plain40.json; srv_stop; fi
done
rm -f $O/proof3-q35.nsys-rep
if SENV="TMPDIR=$O/nsys-tmp" KSIG=SIGINT KTO=300 cfg_start proof3-q35 $O/tomlg/qwen3.6-35b.toml \
     $NSYS profile -t cuda,cublas --sample=none --cpuctxsw=none --cuda-graph-trace=node --cuda-flush-interval=100 -o $O/proof3-q35 -f true; then
  timeout 300 python3 $I/dec.py $PORT qwen3.6-35b $O/proof3-q35.json 1 64; srv_stop; fi
echo "== noblas-w10 end $(el)"
