#!/bin/bash
# orca window 1 (of 3): [B] mistral.rs build of the merged orca branch (iq4xs + devmap) with the qwen35 VRAM fixes,
# in the background; [K] oxide crate builds in oxide-orca-gate (flash-prefill, iq4_xs, fmt_mmq, iq3_s, quant-a, quant-c),
# regenerated PTX vs the committed tree / the embedded copies, kernel gates; [V] OrcaSAQ-2 27B VRAM decomposition:
# iq4xs binary (before), new binary with 9 recurrent slots (merge only), new binary default (2 slots); [O] context
# probes MTP 0 / 1; [C] CPU/GPU split (the fix for "device mismatch in mul") G1 + chunked prompt + prefix-cache hit.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/top-orca/m4/orca/window1.sh
source $HOME/titan-engine/top-orca/m4/orca/lib.sh
exec > >(tee -a $I/window1-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin orca-window1 58
echo "src: mistral.rs $(git -C $SRC log --oneline -1 | cut -c1-70) $(git -C $SRC diff --quiet && echo clean || echo DIRTY)"
echo "     candle $(git -C $E/top-orca log --oneline -1 | cut -c1-60) $(git -C $E/top-orca diff --quiet -- candle && echo clean || echo DIRTY); oxide $(git -C $OX log --oneline -1 | cut -c1-60) $(git -C $OX diff --quiet && echo clean || echo DIRTY)"
echo "     oxide-kernels symlink: $(readlink $E/top-orca/oxide-kernels); TITAN_* in env: $(env | grep -c '^TITAN_')"

echo "== [B] mistral.rs build, background ($(el))"
( mbuild 1800 > $O/mbuild.status 2>&1 ) &
BP=$!

echo "== [K] oxide builds in oxide-orca-gate ($(el))"
declare -A RC
for c in flash-prefill iq4_xs iq3_s mistralrs-quant-a fmt_mmq mistralrs-quant-c; do oxbuild $c orca-ox 1200; RC[$c]=$?; done
echo "-- regenerated (oxide-orca-gate) vs committed (oxide-orca) PTX:"
for c in flash-prefill iq4_xs iq3_s mistralrs-quant-a fmt_mmq mistralrs-quant-c; do
  for f in $OXG/$c/*.ptx; do r=${f#$OXG/}
    if cmp -s $f $OX/$r; then echo "  $r: byte-identical"
    elif cmp -s <(sed 's#//.*##' $OX/$r) <(sed 's#//.*##' $f); then echo "  $r: differs only inside // comments"
    else echo "  $r: DIFFERS; normalised: $(python3 $I/ptxcmp.py $OX/$r $f | head -1)"; fi
  done
done
cmp <(tr -d '\r' < $SRC/mistralrs-core/src/attention/flash_prefill_oxide.ptx) <(tr -d '\r' < $OXG/flash-prefill/flash_prefill.ptx) \
  && echo "  regenerated flash_prefill.ptx == embedded mistralrs-core/src/attention/flash_prefill_oxide.ptx (byte-identical)" \
  || echo "  regenerated flash_prefill.ptx != embedded flash_prefill_oxide.ptx"
cmp $E/top-orca/candle/candle-core/src/quantized/iq4_xs_oxide.ptx $OXG/iq4_xs/iq4_xs.ptx && echo "  regenerated iq4_xs.ptx == embedded candle iq4_xs_oxide.ptx"
cmp $E/top-orca/candle/candle-core/src/quantized/iq3_s_oxide.ptx $OXG/iq3_s/iq3_s.ptx && echo "  regenerated iq3_s.ptx == embedded candle iq3_s_oxide.ptx"
mkdir -p $O/expx/mistralrs-quant/src/gguf; (cd $OXG/fmt_mmq && python3 export_ptx.py $O/expx > /dev/null)
for f in iq4_nl mxfp4 nvfp4 iq4_xs; do echo "  export $f: $(python3 $I/ptxcmp.py $SRC/mistralrs-quant/src/gguf/${f}_mmq_oxide.ptx $O/expx/mistralrs-quant/src/gguf/${f}_mmq_oxide.ptx | head -1)"; done
echo "== [K1] kernel gates on the regenerated PTX ($(el))"
[ ${RC[flash-prefill]} = 0 ] && { (cd $OXG/flash-prefill && FP_BASE_PTX=$OX/flash-prefill/flash_prefill.ptx timeout 1200 ./target/release/flash-prefill > $O/gate-fp.log 2>&1); echo "flash-prefill gate rc=$?"
  grep -E "registers|accuracy|splits|mutation|differ in any bit|gate:|WEAK|FAIL|^vs v1|^window:" $O/gate-fp.log | grep -v "^  \|informational" | tail -40; }
[ ${RC[iq4_xs]} = 0 ] && { (cd $OXG/iq4_xs && timeout 900 ./target/release/iq4_xs > $O/gate-iq4xs.log 2>&1); echo "iq4_xs gate rc=$?"; grep -E "mutation|cpu oracle|FAIL|PASS|launches|failing" $O/gate-iq4xs.log | grep -v "^CASE" | tail -10; }
[ ${RC[iq3_s]} = 0 ] && { (cd $OXG/iq3_s && timeout 600 ./target/release/iq3_s > $O/gate-iq3s.log 2>&1); echo "iq3_s gate rc=$?"; grep -E "FAIL|PASS|launches|mutation|cpu oracle" $O/gate-iq3s.log | tail -6; }
[ ${RC[mistralrs-quant-a]} = 0 ] && { (cd $OXG/mistralrs-quant-a && timeout 900 ./target/release/mistralrs-quant-a > $O/gate-qa.log 2>&1); echo "quant-a gate rc=$?"; grep -E "launcher calls|FAIL|PASS" $O/gate-qa.log | tail -4; }
[ ${RC[fmt_mmq]} = 0 ] && for f in iq4_xs iq4_nl; do (cd $OXG/fmt_mmq && timeout 900 ./target/release/fmt_mmq --fmt=$f > $O/gate-fm-$f.log 2>&1); echo "fmt_mmq $f gate rc=$?"; grep -E "mutation|launches|FAIL|PASS|failing|panicked" $O/gate-fm-$f.log | tail -5; done
[ ${RC[mistralrs-quant-c]} = 0 ] && { (cd $OXG/mistralrs-quant-c && MRQC_ONLY=q4_0,q4_1,q5_0,q5_1,q8_0,q2_k,q3_k,q4_k,q5_k,q6_k timeout 1200 ./target/release/mistralrs-quant-c > $O/gate-qc.log 2>&1); echo "quant-c gate rc=$?"; grep -E "launcher calls|FAIL|PASS|failing" $O/gate-qc.log | tail -4; }

echo "== [B] waiting for the build ($(el))"
wait $BP; cat $O/mbuild.status
if ! grep -q "Finished" $O/build-$WN.log || ! [ $MBIN -nt $SRC/mistralrs-core/src/models/quantized_qwen35_moe.rs ]; then build_bin 1500
else cp $MBIN $NEWBIN; BIN=$NEWBIN; echo "binary: $BIN sha256 $(sha256sum $BIN | cut -d' ' -f1) from $(git -C $SRC log --oneline -1 | cut -c1-60)"; fi
grep -E "^warning: unused|^error" $O/build-$WN.log | sort | uniq -c | head

vram() { # TAG BIN ENV: Orca all-GPU, 8192, one short request with the allocator log
  BIN=$2
  if srv_start $1 $OD $OF "TITAN_DEVMAP_LOG=1 $3" -n 0:64 --max-seq-len 8192; then
    echo "VRAM $1 after load: $(smi)"; probe $O/$1-p.json 50 8; echo "VRAM $1 after a request: $(smi)"; memlines $SLOG | head -2; srv_stop
  fi
}
echo "== [V] OrcaSAQ VRAM decomposition, all 64 layers on the GPU, 8192 ($(el))"
vram v-iq4xs $IQBIN "TITAN_MTP=0"
vram v-new9 $NEWBIN "TITAN_MTP=0 TITAN_RECURRENT_SLOTS=9"
vram v-new $NEWBIN "TITAN_MTP=0"
vram v-new-m1 $NEWBIN "TITAN_MTP=1"
BIN=$NEWBIN

echo "== [M] auto device map at 8192 / 16384 / 32768, MTP 0 and 1 (load only) ($(el))"
for m in 0 1; do for s in 8192 16384 32768; do
  if srv_start map-m$m-$s $OD $OF "TITAN_MTP=$m" --max-seq-len $s; then echo "VRAM after load: $(smi)"; srv_stop; fi
done; done

echo "== [O] context probes, all-GPU (-n 0:64), --max-seq-len 32768 ($(el))"
for m in 0 1; do
  [ $(left) -gt 700 ] || { echo "skip probes MTP $m: $(left)s left"; continue; }
  if srv_start o-m$m $OD $OF "TITAN_MTP=$m TITAN_DEVMAP_LOG=1" -n 0:64 --max-seq-len 32768; then
    PT=600 probe $O/o-m$m-ctx.json 1000,4000,8000,12000,16000,24000 64
    sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -E "titan mtp stats" | tail -1 | sed 's/^.*INFO [a-z_:]*: //'; srv_stop
  fi
done

echo "== [C] CPU/GPU split: -n 0:58 (layers 58-63 on the CPU), MTP 0: G1, a 2k chunked prompt twice (prefix-cache hit) ($(el))"
if [ $(left) -gt 400 ] && srv_start c58 $OD $OF "TITAN_MTP=0" -n 0:58 --max-seq-len 8192; then
  gate g1 c58-g1 $GP; PT=400 probe $O/c58-p.json 2000,2000 32; srv_stop
  echo "device mismatch lines: $(grep -c 'device mismatch' $SLOG), errors: $(grep -c ' ERROR ' $SLOG), prefix hits: $(sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -ci 'prefix cache hit\|resum')"
  tmo $O/c58-g1.json > /dev/null; cmpg g1 $O/c58-g1.json $O/ref-orca-g1.json | head -3
fi
free -m | head -2
echo "== orca-window1 end $(el)"
