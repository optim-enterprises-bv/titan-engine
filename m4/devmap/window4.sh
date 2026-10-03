#!/bin/bash
# devmap2 window 1 (of 3): [K] oxide flash-prefill with the head-dim-128 kernels: build (rebuild.flag loop on error),
# gate (f64 reference, splits, mutations; every existing kernel vs the committed PTX bit for bit), export the PTX;
# [B] mistral.rs build (qwen3 flash prompt attention + chunked MLP, kv_cache fixes, bf16 CPU matmul) in the
# background while [L] llama.cpp references for qwen3-14b (G1, G3, G3l and llama's own G3 spread); [Q] qwen3 transients
# on the new path; [P] kv_cache:857 panic: v3 binary vs new; [C] forced CPU layers on gemma4 f32 and REDCELL bf16;
# [T] qwen3 G1 / G3 on the new binary.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/top-devmap/m4/devmap/window4.sh
source $HOME/titan-engine/top-devmap/m4/devmap/lib.sh
exec > >(tee -a $I/window4-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin devmap2-window4 60
rm -f $I/rebuild.flag $I/stop.flag
echo "src: mistral.rs $(git -C $SRC log --oneline -1 | cut -c1-70) $(git -C $SRC diff --quiet && echo clean || echo DIRTY); candle $(git -C $E/top-devmap log --oneline -1 | cut -c1-50) $(git -C $E/top-devmap diff --quiet -- candle && echo clean || echo DIRTY); oxide $(git -C $OX log --oneline -1 | cut -c1-50) $(git -C $OX diff --quiet && echo clean || echo DIRTY)"
QM="$M/qwen3-14b Qwen3-14B-vanilla-Q5_K_M.gguf"; QF=$M/qwen3-14b/Qwen3-14B-vanilla-Q5_K_M.gguf; QE="TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64"

echo "== [K] oxide flash-prefill build ($(el))"
until oxbuild flash-prefill 600; do
  echo "oxide build failed: waiting for $I/rebuild.flag ($(left)s left)"; date -Is > $O/oxbuild-failed.flag
  while [ ! -e $I/rebuild.flag ] && [ ! -e $I/stop.flag ] && [ $(left) -gt 2400 ]; do sleep 5; done
  [ -e $I/rebuild.flag ] || { echo "no kernel: ending the window"; exit 1; }
  rm -f $I/rebuild.flag $O/oxbuild-failed.flag
done
echo "== [KG] kernel gate ($(el))"
(cd $OX/flash-prefill && FP_BASE_PTX=$O/base_flash_prefill.ptx timeout 1200 ./target/release/flash-prefill > $O/gate-fp.log 2>&1); echo "flash-prefill gate rc=$?"
grep -E "registers|accuracy|splits|mutation|differ in any bit|gate:|WEAK|FAIL" $O/gate-fp.log | grep -v "^  " | tail -40
if grep -q "^flash-prefill gate: PASS" $O/gate-fp.log; then
  bash $OX/flash-prefill/export_ptx.sh $SRC
else
  echo "kernel gate did not pass: stopping before the mistral.rs build"; exit 1
fi

echo "== [B] mistral.rs build, background ($(el))"
( mbuild 1500 > $O/mbuild.status 2>&1 ) &
BP=$!
echo "== [L] llama.cpp references, qwen3-14b ($(el))"
lref lref $QF lref
mkdir -p $O/spread && timeout 1500 bash $E/m4/g4acc/g3spread.sh $QF $O/spread 2>&1 | grep -v "^$" | tail -8
wait $BP; cat $O/mbuild.status
[ -x $MBIN ] && grep -q Finished $O/build-$WN.log || { echo "no binary"; exit 1; }
cp $MBIN $NEWBIN; BIN=$NEWBIN; echo "binary: $BIN sha256 $(sha256sum $BIN | cut -d' ' -f1)"

echo "== [Q] qwen3 transients on the new path, all 40 layers, prefix cache off ($(el))"
if srv_start nq40 $QM "$QE TITAN_DEVMAP_LOG=1" --max-seq-len 16384 --prefix-cache-n 0 -n 0:40; then
  probe $O/nq40.json 1000,4000,8000,12000,16200 64; memlines $SLOG | grep after; srv_stop
fi
echo "== [P] kv_cache:857: v3 binary (before the fix) vs new, prefix cache on, 40 layers ($(el))"
BIN=$V3BIN
if srv_start pold $QM "$QE" --max-seq-len 40960 -n 0:40; then
  probe $O/pold.json 1000,16000,1000 16; sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -E "panicked|is dead|rebooting" | cut -c1-200 | head -4; srv_stop
fi
BIN=$NEWBIN
if srv_start pnew $QM "$QE" --max-seq-len 40960 -n 0:40; then
  probe $O/pnew.json 1000,16000,36000,1000 16; sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -E "panicked|is dead|rebooting|titan: CUDA out of memory" | cut -c1-220 | head -6
  echo "server after the run: $(systemctl --user is-active devmap-srv)"; srv_stop
fi

echo "== [C] forced CPU layers ($(el))"
GA="--dtype f32 --max-model-len 11264 --max-seq-len 11264 --prefix-cache-n 0"
RA="--dtype bf16 --max-model-len 8192 --max-seq-len 8192 --prefix-cache-n 0"
for v in "cg-gpu -n 0:48" "cg-cpu -n 0:44"; do set -- $v; n=$1; shift
  if srv_start $n $M/gemma4-12b-qat gemma-4-12b-it-qat-q4_0.gguf "" $GA "$@"; then tgate g1 $O/$n-g1.json $G/prompts/gemma4-12b.json | tail -1; srv_stop; fi
done
tmapg gemma4 $O/cg-gpu-g1.json $O/cg-cpu-g1.json > /dev/null
cmpg g1 $O/cg-cpu-g1.json $O/cg-gpu-g1.json; cmpg g1 $O/cg-cpu-g1.json $G/out/ref-g12-g1.json; cmpg g1 $O/cg-gpu-g1.json $G/out/ref-g12-g1.json
for v in "cr-gpu -n 0:30" "cr-cpu -n 0:27"; do set -- $v; n=$1; shift
  if srv_start $n $M/redcell-26b REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf "" $RA "$@"; then tgate g1 $O/$n-g1.json $G/prompts/redcell.json | tail -1; srv_stop; fi
done
cmpg g1 $O/cr-cpu-g1.json $O/cr-gpu-g1.json

echo "== [T] qwen3-14b G1 / G3 / G3l, new binary, 40 layers ($(el), $(left)s left)"
if srv_start tq $QM "$QE" --max-seq-len 16384 --prefix-cache-n 0 -n 0:40; then
  tgate g1 $O/tq-g1.json $I/prompts/qwen3-14b.json | tail -1; tgate g3 $O/tq-g3.json | tail -1; tgate g3 $O/tq-g3l.json 28000 | tail -1; srv_stop
fi
tmapq $O/tq-g1.json $O/tq-g3.json $O/tq-g3l.json
cmpg g1 $O/tq-g1.json $O/lref-g1.json; cmpg g3 $O/tq-g3.json $O/lref-g3.json; cmpg g3 $O/tq-g3l.json $O/lref-g3l.json
free -m | head -2
echo "== devmap2-window4 end $(el)"
