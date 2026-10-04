#!/bin/bash
# pattn window 1 (of 3): [K] cargo oxide build of mistralrs-paged-attn-a / -b (v0.9.4 ABI: FP8 E4M3 reshape / gather,
# copy_blocks_u8, flashinfer_decode v_scale / k_scale / groups 5-7), regenerated PTX vs the committed PTX per family;
# [G] launcher gates vs the v0.9.4 nvcc objects; [B] nvcc-free mistral.rs build (background, after the PTX exists);
# [M] mutation checks; [S] Spark-X2.5 smoke with paged attention on / off (G1).
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/top-pattn/m4/pattn/window1.sh
source $HOME/titan-engine/top-pattn/m4/pattn/lib.sh
source $I/mut.sh
exec > >(tee -a $I/window1-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin pattn-window1 58
echo "src: mistral.rs $(git -C $SRC log --oneline -1 | cut -c1-70) $(git -C $SRC diff --quiet && echo clean || echo DIRTY)"
echo "     oxide $(git -C $OX log --oneline -1 | cut -c1-60) $(git -C $OX diff --quiet && echo clean || echo DIRTY); symlink $(readlink $E/top-pattn/oxide-kernels)"
waitfix() { # TAG MIN_LEFT_S: wait for $I/fix.flag (written after a fix) while time allows
  rm -f $I/fix.flag; date -Is > $O/$1-failed.flag; echo "!! $1 failed: waiting for $I/fix.flag ($(left)s left)"
  while [ ! -e $I/fix.flag ] && [ $(left) -gt $2 ]; do sleep 10; done
  rm -f $O/$1-failed.flag; [ -e $I/fix.flag ] && { rm -f $I/fix.flag; return 0; }; return 1
}
for c in a b; do git -C $OX show HEAD:mistralrs-paged-attn-$c/mistralrs_paged_attn_$c.ptx > $O/base-$c.ptx; done

echo "== [K] oxide builds ($(el))"
for c in a b; do
  until oxbuild $OX mistralrs-paged-attn-$c pattn-ox 1500; do waitfix ox-$c 1500 || { echo "no crate $c: ending the window"; exit 1; }; done
  python3 $I/ptxkind.py $O/base-$c.ptx $OX/mistralrs-paged-attn-$c/mistralrs_paged_attn_$c.ptx
done

echo "== [B] mistral.rs build, background ($(el))"
( mbuild 1800 > $O/mbuild.status 2>&1 ) &
BP=$!

echo "== [G] launcher gates vs the v0.9.4 nvcc objects ($(el))"
kgate $OX mistralrs-paged-attn-a $O/gate-a.log 900
grep -E "family|FAIL|first diff" $O/gate-a.log | head -30
kgate $OX mistralrs-paged-attn-b $O/gate-b.log 1800
grep -E "family|coverage|not reachable|NOT LAUNCHED|FAIL|first diff|status diff|RACE|skipped" $O/gate-b.log | head -40

echo "== [M] mutation checks ($(el))"
mutants_b
mutants_a

echo "== [B] waiting for the build ($(el))"
wait $BP; cat $O/mbuild.status
until grep -q "Finished" $O/build-$WN.log && [ $MBIN -nt $OX/mistralrs-paged-attn-b/mistralrs_paged_attn_b.ptx ]; do
  waitfix mbuild 900 || { echo "no binary: ending the window"; exit 1; }; mbuild 1500
done
cp $MBIN $NEWBIN; BIN=$NEWBIN; sha256sum $BIN | tee $NEWBIN.sha256
echo "binary: $BIN from $(git -C $SRC log --oneline -1 | cut -c1-60) $(git -C $SRC diff --quiet && echo clean || echo DIRTY)"
nm -g --defined-only $BIN 2>/dev/null | grep -cE " T (titan_nvcc_only_(reshape_and_cache|gather_kv_cache|flashinfer_decode)|copy_blocks_u8)" | sed 's/^/symbols: nvcc-only stubs left for the four launchers (want 0 or only copy_blocks_u8 from oxide): /'
nm -g --defined-only $BIN 2>/dev/null | grep -E " T (reshape_and_cache_flashinfer|gather_kv_cache_flashinfer|flashinfer_decode|copy_blocks_u8)$"

echo "== [S] Spark-X2.5 smoke: paged attention on / off, G1 ($(el))"
SA="--max-model-len 16384 --max-seq-len 16384"; SP=$G/prompts/spark.json; SK=Spark-X2.5-4B-Q4_K_M.gguf
for pa in on off; do
  if PA=$pa srv_start s-$pa $M/spark-x2.5 $SK "" $SA $( [ $pa = on ] && echo --pa-context-len 16384 ); then
    echo "VRAM after load: $(smi)"; palines $SLOG
    gate g1 s-$pa-g1 $SP | tail -1; echo "VRAM after G1: $(smi)"; srv_stop
  fi
done
[ -e $O/s-on-g1.json ] && [ -e $O/s-off-g1.json ] && { same $O/s-on-g1.json $O/s-off-g1.json; cmpg g1 $O/s-on-g1.json $O/s-off-g1.json; cmpg g1 $O/s-on-g1.json $G/out/refc-spark-g1.json; cmpg g1 $O/s-off-g1.json $G/out/refc-spark-g1.json; }
free -m | head -2
echo "== pattn-window1 end $(el)"
