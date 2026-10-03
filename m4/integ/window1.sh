#!/bin/bash
# integ-20261002 window 1: oxide-integ crate builds + kernel gates (iq3_s, iq-moe incl. IQ4_NL + five i-quants,
# flash-prefill incl. win=0 == v1/v2), cargo metadata (candle-core must resolve to top-integ/candle), nvcc-free
# mistral.rs build of top-integ/mr-integ -> bin/mistralrs-titan-integ, then the regression core:
# 35B MTP=2 vs m6/out/h-off.json, 35B MTP-off (MTP-dir file, TITAN_MTP=0) vs m6/out/mtpfile-off.json,
# IQ2_M old vs new 8x300, Bonsai-27B Q1_0 old vs new 8x300, Bonsai-2 B2-off / B2-mtp1 vs m4/bonsai2/out/o-off.json.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/integ/window1.sh
source $HOME/titan-engine/m4/integ/lib.sh
exec > >(tee -a $I/window1-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin integ-window1 58
echo "src: $(git -C $SRC log --oneline -1 | cut -c1-80) $(git -C $SRC diff --quiet && echo clean || echo DIRTY)"
echo "candle: $(git -C $E/top-integ log --oneline -1 | cut -c1-60); oxide: $(git -C $OX log --oneline -1 | cut -c1-60) $(git -C $OX diff --quiet && echo clean || echo DIRTY)"

echo "== [0] cargo metadata ($(el))"
systemctl --user reset-failed integ-meta 2>/dev/null
timeout 300 systemd-run --user --unit=integ-meta --collect --wait --pipe -q -p MemoryMax=4G -p MemorySwapMax=0 -p WorkingDirectory=$SRC \
  --setenv=PATH=$HOME/.cargo/bin:/usr/bin:/bin cargo metadata --offline --format-version 1 > $O/metadata.json 2> $O/metadata.err
echo "metadata rc=$?"; tail -3 $O/metadata.err
python3 -c "
import json,sys
m=json.load(open(sys.argv[1]))
for p in m['packages']:
    if p['name'] in ('candle-core','candle-nn','candle-kernels','titan-oxide-ffi'): print('RESOLVE',p['name'],p['manifest_path'])
" $O/metadata.json

echo "== [1] oxide-integ builds ($(el))"
oxbuild candle-quantized integ-oxcq 1200 > $O/oxbuild-cq.status 2>&1 &
CQ=$!
oxbuild mistralrs-quant-a integ-ox 900; QA=$?
oxbuild iq3_s integ-ox 900; IQ=$?
oxbuild flash-prefill integ-ox 900; FP=$?
wait $CQ; cat $O/oxbuild-cq.status
echo "-- regenerated vs committed PTX:"
git -C $OX status --short
for f in $(git -C $OX diff --name-only -- '*.ptx'); do
  if cmp -s <(git -C $OX show HEAD:$f | sed 's#//.*##') <(sed 's#//.*##' $OX/$f); then echo "  $f: differs only inside // comments"; else echo "  $f: CODE DIFFERS ($(git -C $OX diff --stat -- $f | tail -1))"; fi
done
cp -a $OX/mistralrs-quant-a/mistralrs_quant_a.ptx $O/regen-mistralrs_quant_a.ptx 2>/dev/null

echo "== [1b] kernel gates ($(el))"
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
  (cd $OX/flash-prefill && timeout 400 ./target/release/flash-prefill > $O/gate-flash-prefill.log 2>&1); echo "flash-prefill gate rc=$?"
  grep -vE "^  flash" $O/gate-flash-prefill.log | tail -8
fi
echo "-- restoring the committed (branch-gated) PTX for the embed"
git -C $OX checkout -- . ; git -C $OX status --short

echo "== [2] mistral.rs build ($(el))"
mbuild 1800 || { echo "no new binary: ending the window"; exit 1; }
cp $MBIN $NEWBIN; BIN=$NEWBIN; echo "binary: $BIN sha256 $(sha256sum $BIN | cut -d' ' -f1)"
grep -E "^warning: unused|^error" $O/build-$WN.log | sort | uniq -c | head
C=python3; CL=$I/b2client.py; BT=$I/bigtest.py

echo "== [3] 35B gate pair ($(el))"
coll integ-m2 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2 && cmp40 $E/m6/out/h-off.json $E/m3/out/integ-m2.json
coll integ-mf0 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=0 && cmp40 $E/m6/out/mtpfile-off.json $E/m3/out/integ-mf0.json

echo "== [4] Bonsai-2 27B: B2-off, B2-mtp1 vs o-off ($(el))"
B2=(--max-seq-len 65536)
if srv_start b-off $M/bonsai2-27b Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf "TITAN_MTP=0 TITAN_REASONING_EFFORT=medium" "${B2[@]}"; then
  $C $CL greedy titan $PORT b-off 8 300; $C $CL cmp o-off b-off; srv_stop
fi
if srv_start b-n1 $M/bonsai2-27b Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf "TITAN_MTP=1 TITAN_REASONING_EFFORT=medium" "${B2[@]}"; then
  $C $CL greedy titan $PORT b-n1 8 300; $C $CL cmp o-off b-n1; grep -o "titan mtp stats.*" $SLOG | tail -1; srv_stop
fi

echo "== [5] IQ2_M old vs new, 8 x 300 ($(el))"
for v in new old; do
  [ $v = old ] && BIN=$OLDBIN || BIN=$NEWBIN
  if srv_start i-$v $M/q35-lowbit Qwen3.6-35B-A3B-UD-IQ2_M.gguf "TITAN_MTP=2 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64" --max-seq-len 65536; then
    $C $CL greedy titan $PORT i-$v 8 300; srv_stop
  fi
done
$C $CL cmp i-old i-new; $C $CL cmp i-old-b2merge i-new

echo "== [6] Bonsai-27B Q1_0 old vs new, 8 x 300 ($(el))"
for v in new old; do
  [ $v = old ] && BIN=$OLDBIN || BIN=$NEWBIN
  if srv_start q-$v $M/bonsai/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf "TITAN_ATTN_FLASH_PREFILL=1" --max-seq-len 32768; then
    $C $CL greedy titan $PORT q-$v 8 300; srv_stop
  fi
done
$C $CL cmp q-old q-new; $C $CL cmp q-old-b2merge q-new
BIN=$NEWBIN
free -m | head -2
echo "== integ-window1 end $(el)"
