#!/bin/bash
# noblas window 8 (round 2, fix): build with the gemma4 MoE router in F32 (ggml's precision: F32 norm, the GGUF's F32
# router weight, F32 logits; TITAN_G4_ROUTER_F32=0 reverts), stage bin/mistralrs-titan-noblas2; REDCELL G1 / G3 (on,
# and with TITAN_GEMM_REF=all on top); then the roster gates (G1 / G3 / G5) + decode through the swap server, nsys
# proof on REDCELL.
source $HOME/titan-engine/top-noblas/m4/noblas/lib.sh
restic_wait
exec > >(tee -a $I/window8-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin noblas-w8 58
until mbuild 1500; do waitfix mbuild 900 || exit 1; done
cp $MBIN $E/bin/mistralrs-titan-noblas2; (cd $E/bin && sha256sum mistralrs-titan-noblas2 > mistralrs-titan-noblas2.sha256 && cat mistralrs-titan-noblas2.sha256)
BIN=$E/bin/mistralrs-titan-noblas2
ldd $BIN | grep -ciE "cublas|curand"; nm -D $BIN | grep -ciE "cublas|curand"
RP=$E/m4/g4s/prompts/redcell.json
for v in "f32 " "f32ref TITAN_GEMM_REF=all"; do set -- $v; t=$1; shift
  if SENV="$*" cfg_start r2-$t $O/tomlg/redcell-26b.toml; then
    gm redcell-26b g1 r2-$t-g1 $RP; gm redcell-26b g3 r2-$t-g3 13500; srv_stop; fi
done
BIN=$E/bin/mistralrs-titan-noblas2 roster_gates new2
BIN=$E/bin/mistralrs-titan-noblas2 roster_dec new2
rm -f $O/proof2-red.nsys-rep
if SENV="TMPDIR=$O/nsys-tmp" KSIG=SIGINT KTO=300 cfg_start proof2-red $O/tomlg/redcell-26b.toml \
     $NSYS profile -t cuda,cublas --sample=none --cpuctxsw=none --cuda-flush-interval=100 -o $O/proof2-red -f true; then
  timeout 300 python3 $I/dec.py $PORT redcell-26b $O/proof2-red.json 1 32; srv_stop; fi
echo "== noblas-w8 end $(el)"
