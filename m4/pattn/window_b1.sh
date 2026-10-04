#!/bin/bash
# pattn phase B window 1: [K] cargo oxide builds: mistralrs-core-cuda (v0.9.4 cuda_graph_copy_2d_bytes, pad_decode_input_u32,
# pack_completion_input_u32) and mistralrs-paged-attn-b (FP8 E4M3 KV decode, 432 stage-2 instances); PTX vs committed;
# gates vs the v0.9.4 nvcc objects; mutants (graph helpers, FP8 byte order, v_scale reorder on f32); [B] mistral.rs build;
# [G] qwen3-14b paged ON with upstream CUDA decode graphs on until it loads and decodes (fix loop: each nvcc-only stub hit
# is ported, then rebuilt), G1 / G5 graphs on vs off; [F] Spark with an FP8 E4M3 paged KV cache: G1 vs the f16/bf16 KV,
# G3, G5, VRAM.
source $HOME/titan-engine/top-pattn/m4/pattn/lib.sh
source $I/mut.sh
restic_wait
exec > >(tee -a $I/window_b1-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin pattn-b1 58
echo "src: mistral.rs $(git -C $SRC log --oneline -1 | cut -c1-50) $(git -C $SRC diff --quiet && echo clean || echo DIRTY); oxide $(git -C $OX log --oneline -1 | cut -c1-50) $(git -C $OX diff --quiet && echo clean || echo DIRTY)"
for c in mistralrs-core-cuda mistralrs-paged-attn-b; do cp $OX/$c/$(echo $c | tr - _).ptx $O/b1base-$c.ptx; done
echo "== [K] oxide builds ($(el))"
for c in mistralrs-core-cuda mistralrs-paged-attn-b; do
  until oxbuild $OX $c pattn-ox 1500; do waitfix ox-$c 1800 || exit 1; done
  python3 $I/ptxkind.py $O/b1base-$c.ptx $OX/$c/$(echo $c | tr - _).ptx
done
echo "== [B] mistral.rs build, background ($(el))"
( build_bin2 1500 > $O/mbuild-b1.status 2>&1 ) &
BP=$!
echo "== [K] gates ($(el))"
kgate $OX mistralrs-core-cuda $O/gate-core-b1.log 900 GATE_ONLY=graph GATE_ROUNDS=2
grep -E "graph|FAIL" $O/gate-core-b1.log | head -12
kgate $OX mistralrs-paged-attn-b $O/gate-b-b1.log 2400
grep -E "family|coverage|not reachable|NOT LAUNCHED|FAIL|first diff|status diff|RACE|skipped" $O/gate-b-b1.log | head -30
echo "== [M] mutants ($(el))"
mutants_core; mutants_fp8dec; mutants_b_vscale
wait $BP; cat $O/mbuild-b1.status
BIN=$NEWBIN2; [ -x $BIN ] && [ $BIN -nt $OX/mistralrs-paged-attn-b/mistralrs_paged_attn_b.ptx ] || { echo "no fresh binary"; exit 1; }
nm -g --defined-only $BIN | grep -cE " T (pad_decode_input_u32|pack_completion_input_u32|cuda_graph_copy_2d_bytes)$" | sed 's/^/new graph-helper exports in the binary: /'

QM=$M/qwen3-14b; QF=Qwen3-14B-vanilla-Q5_K_M.gguf; QP=$E/m4/devmap/prompts/qwen3-14b.json
echo "== [G] qwen3-14b paged ON + upstream CUDA decode graphs ($(el))"
tries=0
while [ $tries -lt 4 ]; do
  tries=$((tries + 1))
  if PA=on srv_start g1-on $QM $QF "TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64" --max-seq-len 16384 --prefix-cache-n 0; then
    echo "VRAM after load: $(smi)"; sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -iE "graph" | grep -v " DEBUG " | head -4 | cut -c1-200
    gate g1 qg-on-g1 $QP | tail -1; gate g5 qg-on-g5 8000 128 | tail -1; srv_stop
    sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -iE "graph (capture|replay)|captured|nvcc-only| ERROR " | sort | uniq -c | head -8 | cut -c1-220
    break
  fi
  stub=$(grep -o "CUDA launcher \`[a-z0-9_]*\` is nvcc-only" $SLOG | head -1); echo "!! load failed: ${stub:-(no nvcc-only stub named)}"
  [ -n "$stub" ] || break
  echo "$stub" >> $O/graph-stubs.txt
  waitfix graph-stub 1500 || break
  if [ -e $I/fixbuild.sh ]; then bash $I/fixbuild.sh; mv $I/fixbuild.sh $O/fixbuild-$tries.sh; else oxbuild $OX mistralrs-core-cuda pattn-ox 1200; fi
  (cd $OX/titan-oxide-ffi && python3 gen.py > /dev/null); build_bin2 1500
done
if PA=on srv_start g1-off $QM $QF "TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64 MISTRALRS_CUDA_GRAPHS=0" --max-seq-len 16384 --prefix-cache-n 0; then
  gate g1 qg-off-g1 $QP | tail -1; gate g5 qg-off-g5 8000 128 | tail -1; srv_stop; fi
same $O/qg-on-g1.json $O/qg-off-g1.json; tps $O/qg-on-g5.json $O/qg-off-g5.json $O/q-off-g5.json

echo "== [F] Spark with an FP8 E4M3 paged KV cache ($(el))"
SK=Spark-X2.5-4B-Q4_K_M.gguf; SP=$G/prompts/spark.json; SA="--max-model-len 16384 --max-seq-len 16384 --prefix-cache-n 0"
if PA=on srv_start f8-on $M/spark-x2.5 $SK "" $SA --pa-cache-type f8e4m3; then
  echo "VRAM after load: $(smi)"; palines $SLOG 5
  gate g1 f8-on-g1 $SP | tail -1; gate g3 f8-on-g3 | tail -1; gate g3 f8-on-g3l 28000 | tail -1; echo "VRAM after 8k: $(smi)"
  gate g5 f8-on-g5 8000 128 | tail -1; srv_stop
  sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -E " WARN | ERROR " | sort | uniq -c | head -5 | cut -c1-200
  for f in f8-on-g1 f8-on-g3 f8-on-g3l; do cp $O/$f.json $O/$f.ids.json; done; tmaps $O/f8-on-g1.json $O/f8-on-g3.json $O/f8-on-g3l.json
  cmpg g1 $O/f8-on-g1.json $O/s6-on-g1.json; cmpg g1 $O/f8-on-g1.json $G/out/refc-spark-g1.json
  cmpg g3 $O/f8-on-g3.json $O/s6-on-g3.json; cmpg g3 $O/f8-on-g3l.json $O/s6-on-g3l.json
  tps $O/f8-on-g5.json $O/s6-on-g5.json
fi
free -m | head -2
echo "== pattn-b1 end $(el)"
