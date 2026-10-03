#!/bin/bash
# integ2-20261003 window 1: [0] merge sanity (cargo metadata: candle-core must resolve to top-integ2/candle, titan-oxide-ffi
# to oxide-integ2 via the symlink; no branch-worktree paths); [K] oxide-integ2 crate builds (flash-prefill incl. hd512,
# mistralrs-quant-a incl. redcell, iq3_s, candle-quantized) + kernel gates on the regenerated PTX, then the committed PTX
# restored and the file-loading gates re-run on it; [B] nvcc-free mistral.rs build -> bin/mistralrs-titan-integ2;
# [R] regression core (35B MTP=2 / MTP-off 40/40, Bonsai-2 B2-off 8/8, IQ2_M and Bonsai-27B Q1_0 vs integ 8/8);
# [S] Spark Q4_K_M in integ's request order (g5 x3, g3, g1, g2) vs emb / integ; [T] config unit tests if time is left.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/top-integ2/m4/integ2/window1.sh
source $HOME/titan-engine/top-integ2/m4/integ2/lib.sh
exec > >(tee -a $I/window1-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin integ2-window1 58
echo "src: mistral.rs $(git -C $SRC log --oneline -1 | cut -c1-70) $(git -C $SRC diff --quiet && echo clean || echo DIRTY)"
echo "     candle $(git -C $E/top-integ2 log --oneline -1 | cut -c1-60); oxide $(git -C $OX log --oneline -1 | cut -c1-60) $(git -C $OX diff --quiet && echo clean || echo DIRTY)"
echo "     oxide-kernels symlink: $(readlink $E/top-integ2/oxide-kernels); TITAN_* in env: $(env | grep -c '^TITAN_')"

echo "== [0] cargo metadata, branch paths ($(el))"
echo "git grep branch paths (expect none): $(git -C $SRC grep -n 'top-emb\|top-hd512\|top-redcell\|top-pcache\|oxide-hd512\|oxide-redcell' | wc -l)"
systemctl --user reset-failed integ2-meta 2>/dev/null
timeout 300 systemd-run --user --unit=integ2-meta --collect --wait --pipe -q -p MemoryMax=4G -p MemorySwapMax=0 -p WorkingDirectory=$SRC \
  --setenv=PATH=$HOME/.cargo/bin:/usr/bin:/bin cargo metadata --offline --format-version 1 > $O/metadata.json 2> $O/metadata.err
echo "metadata rc=$?"; tail -3 $O/metadata.err
python3 -c "
import json,sys,os
m=json.load(open(sys.argv[1]))
for p in m['packages']:
    if p['name'] in ('candle-core','candle-nn','candle-kernels','titan-oxide-ffi'): print('RESOLVE',p['name'],p['manifest_path'],'->',os.path.realpath(p['manifest_path']))
" $O/metadata.json

echo "== [K] oxide-integ2 builds ($(el))"
oxbuild candle-quantized integ2-oxcq 1200 > $O/oxbuild-cq.status 2>&1 &
CQ=$!
oxbuild mistralrs-quant-a integ2-ox 900; QA=$?
oxbuild iq3_s integ2-ox 900; IQ=$?
oxbuild flash-prefill integ2-ox 900; FP=$?
wait $CQ; cat $O/oxbuild-cq.status
echo "-- regenerated vs committed PTX:"
git -C $OX status --short
for f in $(git -C $OX diff --name-only -- '*.ptx'); do
  if cmp -s <(git -C $OX show HEAD:$f | sed 's#//.*##') <(sed 's#//.*##' $OX/$f); then echo "  $f: differs only inside // comments"; else echo "  $f: CODE DIFFERS ($(git -C $OX diff --stat -- $f | tail -1))"; fi
done
echo "== [K1] kernel gates, regenerated PTX ($(el))"
if [ $QA = 0 ] && [ -x $OX/mistralrs-quant-a/target/release/mistralrs-quant-a ]; then
  (cd $OX/mistralrs-quant-a && timeout 900 ./target/release/mistralrs-quant-a > $O/gate-qa.log 2>&1); echo "quant-a gate rc=$?"
  grep -E "slot-order|stable|launcher calls|order-only|FAIL|PASS" $O/gate-qa.log | head -30
fi
if [ $IQ = 0 ] && [ -x $OX/iq3_s/target/release/iq3_s ]; then
  (cd $OX/iq3_s && timeout 600 ./target/release/iq3_s > $O/gate-iq3s.log 2>&1); echo "iq3_s gate rc=$?"
  grep -E "FAIL|mutation|cpu oracle|PASS|launches" $O/gate-iq3s.log | head -12
fi
if [ -x $OX/candle-quantized/target/release/candle-quantized ]; then
  (cd $OX/candle-quantized && OXIDE_ROOT=$OX OXIDE_PTX=$OX/mistralrs-quant-a/mistralrs_quant_a.ptx timeout 600 \
    ./target/release/candle-quantized iqmoe > $O/gate-iqmoe.log 2>&1); echo "iq-moe gate rc=$?"
  grep -E "indexed_moe_forward_|launches|PASS|FAIL|not ported|failing" $O/gate-iqmoe.log | head -12
fi
if [ $FP = 0 ] && [ -x $OX/flash-prefill/target/release/flash-prefill ]; then
  (cd $OX/flash-prefill && timeout 600 ./target/release/flash-prefill > $O/gate-flash-prefill.log 2>&1); echo "flash-prefill gate rc=$?"
  grep -vE "^  (flash|time)" $O/gate-flash-prefill.log | tail -30
fi
mkdir -p $O/regen; for f in $(git -C $OX diff --name-only -- '*.ptx'); do cp -a $OX/$f $O/regen/; done
echo "-- restoring the committed (branch-gated) PTX for the embed"
git -C $OX checkout -- . ; git -C $OX status --short
echo "== [K2] file-loading gates on the committed PTX ($(el))"
(cd $OX/iq3_s && timeout 300 ./target/release/iq3_s > $O/gate2-iq3s.log 2>&1); echo "iq3_s gate rc=$?"; grep -E "PASS|FAIL" $O/gate2-iq3s.log | tail -2
(cd $OX/candle-quantized && OXIDE_ROOT=$OX OXIDE_PTX=$OX/mistralrs-quant-a/mistralrs_quant_a.ptx timeout 300 ./target/release/candle-quantized iqmoe > $O/gate2-iqmoe.log 2>&1); echo "iq-moe gate rc=$?"; grep -E "PASS|FAIL|failing" $O/gate2-iqmoe.log | tail -2
(cd $OX/flash-prefill && timeout 600 ./target/release/flash-prefill > $O/gate2-flash-prefill.log 2>&1); echo "flash-prefill gate rc=$?"; grep -E "gate:|vs base PTX|^vs v1|^window:" $O/gate2-flash-prefill.log
git -C $OX status --short | grep -v "^??"
cmp <(tr -d '\r' < $SRC/mistralrs-core/src/attention/flash_prefill_oxide.ptx) <(tr -d '\r' < $OX/flash-prefill/flash_prefill.ptx) && echo "embedded flash_prefill_oxide.ptx == oxide-integ2 committed (CR-normalised)"

echo "== [B] mistral.rs build ($(el))"
rm -f $I/rebuild.flag
until mbuild 1800; do
  echo "build failed: waiting for $I/rebuild.flag ($(left)s left)"; date -Is > $O/build-failed.flag
  while [ ! -e $I/rebuild.flag ] && [ $(left) -gt 1500 ]; do sleep 10; done
  [ -e $I/rebuild.flag ] || { echo "no binary: ending the window"; exit 1; }
  rm -f $I/rebuild.flag $O/build-failed.flag; echo "rebuilding ($(el)): $(git -C $SRC log --oneline -1 | cut -c1-80)"
done
cp $MBIN $NEWBIN; BIN=$NEWBIN; echo "binary: $BIN sha256 $(sha256sum $BIN | cut -d' ' -f1) from $(git -C $SRC log --oneline -1 | cut -c1-60)"
grep -E "^warning: unused|^error" $O/build-$WN.log | sort | uniq -c | head

echo "== [R1] 35B gate pair ($(el))"
coll integ2-m2 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2 && cmp40 $E/m6/out/h-off.json $E/m3/out/integ2-m2.json
coll integ2-mf0 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=0 && cmp40 $E/m6/out/mtpfile-off.json $E/m3/out/integ2-mf0.json
echo "== [R2] Bonsai-2 27B B2-off vs o-off ($(el))"
if srv_start b-off $M/bonsai2-27b Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf "TITAN_MTP=0 TITAN_REASONING_EFFORT=medium" --max-seq-len 65536; then
  $C $CL greedy titan $PORT b-off 8 300; srv_stop
fi
$C $CL cmp o-off b-off
echo "== [R3] IQ2_M, Bonsai-27B Q1_0 new vs integ (recorded runs of bin/mistralrs-titan-integ), 8 x 300 ($(el))"
if srv_start i-new $M/q35-lowbit Qwen3.6-35B-A3B-UD-IQ2_M.gguf "TITAN_MTP=2 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64" --max-seq-len 65536; then
  $C $CL greedy titan $PORT i-new 8 300; srv_stop
fi
$C $CL cmp i-integ i-new; $C $CL cmp i-old-b2merge i-new
if srv_start q-new $M/bonsai/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf "TITAN_ATTN_FLASH_PREFILL=1" --max-seq-len 32768; then
  $C $CL greedy titan $PORT q-new 8 300; srv_stop
fi
$C $CL cmp q-integ q-new; $C $CL cmp q-old-b2merge q-new

echo "== [S] Spark-X2.5 Q4_K_M, 16384, integ request order ($(el))"
if srv_start spark $M/spark-x2.5 Spark-X2.5-4B-Q4_K_M.gguf "" --max-model-len 16384 --max-seq-len 16384; then
  for c in 2000 15000 33000; do gate g5 sp-g5-$c $c 128; done
  gate g3 sp-g3; gate g1 sp-g1 $SP/prompts/spark.json; gate g2 sp-g2 $SP/prompts/spark.json 256
  srv_stop
fi
tmap spark $O/sp-g1.json $O/sp-g3.json
same $O/sp-g2.json $E/m4/emb/out/sp-g2.json; same $O/sp-g2.json $IO/sp-g2.json
same $O/sp-g1.json $E/m4/emb/out/sp-g1.json; same $O/sp-g3.json $E/m4/emb/out/sp-g3.json
cmpg g1 $O/sp-g1.json $R/refc-spark-g1.json; cmpg g3 $O/sp-g3.json $R/refc-spark-g3.json; cmpg g2 $O/sp-g2.json $R/baseline-spark-x2.5-q4km-g2.json
g5s $O/sp-g5-2000.json $O/sp-g5-15000.json $O/sp-g5-33000.json $E/m4/emb/out/sp-g5-2000.json $E/m4/emb/out/sp-g5-33000.json

echo "== [T] config unit tests ($(el))"
if [ $(left) -gt 1000 ]; then mtest cli $(( $(left) - 120 )) -p mistralrs-cli --features oxide --bin mistralrs config::tests; else echo "skip [T]: $(left)s left"; fi
free -m | head -2
echo "== integ2-window1 end $(el)"
