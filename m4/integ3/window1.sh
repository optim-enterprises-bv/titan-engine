#!/bin/bash
# integ3-20261003 window 1: [0] merge sanity (cargo metadata, no branch-worktree paths); [B] nvcc-free mistral.rs build
# (background) while [K] the oxide-integ3 crates build and gate (flash-prefill incl. hd512 / r2 / decode rows + existing
# kernels vs integ2 PTX, mistralrs-quant-a, mistralrs-quant-c incl. packed stride, fmt_mmq iq4_nl incl. MoE K=704,
# iq3_s, iq-moe), regenerated PTX then the committed PTX; [R] regression core; [S] Spark G5/G3/G1/G2 vs integ2;
# [L] Spark LoRA G6 on/off vs integ2's recorded run (m4/deq spl-old-*); [I] IQ3_S 9B e2e vs integ2;
# [OOM] REDCELL default prefix cache, G1 -> G3 -> G2 (every request on its own) on the new binary, then the redcell2
# binary as the negative control (it panicked at kv_cache/mod.rs:542).
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/top-integ3/m4/integ3/window1.sh
source $HOME/titan-engine/top-integ3/m4/integ3/lib.sh
exec > >(tee -a $I/window1-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin integ3-window1 58
echo "src: mistral.rs $(git -C $SRC log --oneline -1 | cut -c1-70) $(git -C $SRC diff --quiet && echo clean || echo DIRTY)"
echo "     candle $(git -C $E/top-integ3 log --oneline -1 | cut -c1-60); oxide $(git -C $OX log --oneline -1 | cut -c1-60) $(git -C $OX diff --quiet && echo clean || echo DIRTY)"
echo "     oxide-kernels symlink: $(readlink $E/top-integ3/oxide-kernels); TITAN_* in env: $(env | grep -c '^TITAN_')"
RED=REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf; RP=$G4/prompts/redcell.json; RA=(--max-model-len 8192 --max-seq-len 8192 -n 0:30 --dtype bf16)

echo "== [0] cargo metadata, branch paths ($(el))"
echo "git grep branch paths (expect 0): $(git -C $SRC grep -n 'top-deq\|top-g4acc\|top-redcell2\|oxide-g4acc\|oxide-redcell2\|top-integ2\|oxide-integ2\|top-emb\|top-hd512\|top-pcache' | wc -l)"
timeout 300 systemd-run --user --unit=integ3-meta --collect --wait --pipe -q -p MemoryMax=4G -p MemorySwapMax=0 -p WorkingDirectory=$SRC \
  --setenv=PATH=$HOME/.cargo/bin:/usr/bin:/bin cargo metadata --offline --all-features --format-version 1 > $O/metadata-all.json 2> $O/metadata-all.err
echo "metadata rc=$?"; tail -2 $O/metadata-all.err
python3 -c "
import json,sys,os
m=json.load(open(sys.argv[1]))
for p in m['packages']:
    if p['name'] in ('candle-core','candle-nn','candle-kernels','titan-oxide-ffi'): print('RESOLVE',p['name'],p['manifest_path'],'->',os.path.realpath(p['manifest_path']))
" $O/metadata-all.json

echo "== [B] mistral.rs build, background ($(el))"
( mbuild 1800 > $O/mbuild.status 2>&1 ) &
BP=$!

echo "== [K] oxide-integ3 builds ($(el))"
declare -A RC
for c in flash-prefill mistralrs-quant-a iq3_s candle-quantized mistralrs-quant-c fmt_mmq; do oxbuild $c integ3-ox 1200; RC[$c]=$?; done
echo "-- regenerated vs committed PTX:"; git -C $OX status --short | grep -v "^??"
for f in $(git -C $OX diff --name-only -- '*.ptx'); do
  if cmp -s <(git -C $OX show HEAD:$f | sed 's#//.*##') <(sed 's#//.*##' $OX/$f); then echo "  $f: differs only inside // comments"; else echo "  $f: CODE DIFFERS ($(git -C $OX diff --stat -- $f | tail -1))"; fi
done
kgates() { # TAG (TAG 2 = committed PTX: only the gates that load the PTX file run again)
  local t=$1
  [ "$t" = "" ] && [ ${RC[mistralrs-quant-a]} = 0 ] && { (cd $OX/mistralrs-quant-a && timeout 900 ./target/release/mistralrs-quant-a > $O/gate$t-qa.log 2>&1); echo "quant-a gate rc=$?"; grep -E "launcher calls|FAIL|PASS" $O/gate$t-qa.log | tail -4; }
  [ "$t" = "" ] && [ ${RC[mistralrs-quant-c]} = 0 ] && { (cd $OX/mistralrs-quant-c && MRQC_ONLY=q4_0,q4_1,q5_0,q5_1,q8_0,q2_k,q3_k,q4_k,q5_k,q6_k timeout 1200 ./target/release/mistralrs-quant-c > $O/gate$t-qc.log 2>&1); echo "quant-c gate rc=$?"; grep -E "launcher calls|FAIL|PASS|failing" $O/gate$t-qc.log | tail -4; echo "packed-stride calls: $(grep -c '_moe_packed' $O/gate$t-qc.log)"; }
  [ "$t" = "" ] && [ ${RC[fmt_mmq]} = 0 ] && { (cd $OX/fmt_mmq && timeout 900 ./target/release/fmt_mmq --fmt=iq4_nl > $O/gate$t-fm.log 2>&1); echo "fmt_mmq iq4_nl gate rc=$?"; grep -E "launches|FAIL|PASS|failing" $O/gate$t-fm.log | tail -4; echo "moe-mmq lines: $(grep -ci 'moe' $O/gate$t-fm.log), K=704 lines: $(grep -c '704' $O/gate$t-fm.log)"; }
  [ ${RC[iq3_s]} = 0 ] && { (cd $OX/iq3_s && timeout 600 ./target/release/iq3_s > $O/gate$t-iq3s.log 2>&1); echo "iq3_s gate rc=$?"; grep -E "FAIL|PASS|launches" $O/gate$t-iq3s.log | tail -3; }
  [ ${RC[candle-quantized]} = 0 ] && { (cd $OX/candle-quantized && OXIDE_ROOT=$OX OXIDE_PTX=$OX/mistralrs-quant-a/mistralrs_quant_a.ptx timeout 600 ./target/release/candle-quantized iqmoe > $O/gate$t-iqmoe.log 2>&1); echo "iq-moe gate rc=$?"; grep -E "launches|PASS|FAIL|failing" $O/gate$t-iqmoe.log | tail -3; }
  [ ${RC[flash-prefill]} = 0 ] && { (cd $OX/flash-prefill && timeout 900 ./target/release/flash-prefill > $O/gate$t-fp.log 2>&1); echo "flash-prefill gate rc=$?"; grep -E "accuracy|mutations|differ in any bit|gate:|WEAK|FAIL|^vs v1|^window:" $O/gate$t-fp.log | grep -v informational | tail -24; }
}
echo "== [K1] kernel gates, regenerated PTX ($(el))"; kgates ""
mkdir -p $O/regen; for f in $(git -C $OX diff --name-only -- '*.ptx'); do cp -a $OX/$f $O/regen/; done
echo "-- restoring the committed PTX"; git -C $OX checkout -- . ; git -C $OX status --short | grep -v "^??"
echo "== [K2] kernel gates, committed PTX ($(el))"; kgates 2
git -C $OX status --short | grep -v "^??"
cmp <(tr -d '\r' < $SRC/mistralrs-core/src/attention/flash_prefill_oxide.ptx) <(tr -d '\r' < $OX/flash-prefill/flash_prefill.ptx) && echo "embedded flash_prefill_oxide.ptx == oxide-integ3 committed (CR-normalised)"

echo "== [B] waiting for the build ($(el))"
wait $BP; cat $O/mbuild.status
if ! [ -x $MBIN ] || ! [ $MBIN -nt $SRC/mistralrs-core/src/kv_cache/mod.rs ] || ! grep -q "Finished" $O/build-$WN.log; then
  rm -f $I/rebuild.flag
  until mbuild 1500; do
    echo "build failed: waiting for $I/rebuild.flag ($(left)s left)"; date -Is > $O/build-failed.flag
    while [ ! -e $I/rebuild.flag ] && [ $(left) -gt 1500 ]; do sleep 10; done
    [ -e $I/rebuild.flag ] || { echo "no binary: ending the window"; exit 1; }
    rm -f $I/rebuild.flag $O/build-failed.flag; echo "rebuilding ($(el)): $(git -C $SRC log --oneline -1 | cut -c1-80)"
  done
fi
cp $MBIN $NEWBIN; BIN=$NEWBIN; echo "binary: $BIN sha256 $(sha256sum $BIN | cut -d' ' -f1) from $(git -C $SRC log --oneline -1 | cut -c1-60)"
grep -E "^warning: unused|^error" $O/build-$WN.log | sort | uniq -c | head

echo "== [OOM] REDCELL default prefix cache, G1 -> G3 -> G2 per request: new ($(el))"
if srv_start oom-new $M/redcell-26b $RED "" "${RA[@]}"; then
  timeout 1200 python3 $I/oomgate.py $PORT oom-new $RP
  echo "server unit after the run: $(systemctl --user is-active integ3-srv)"; srv_stop
  echo "OOM lines $(grep -c OUT_OF_MEMORY $SLOG), recoveries $(grep -c 'titan: CUDA out of memory' $SLOG), panics $(grep -c panicked $SLOG)"
  sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -E "Model failed|titan: CUDA out of memory|panicked" | cut -c1-220
fi
same $O/oom-new-g2.json $R/baseline-redcell-g2.json
echo "== [OOM-] negative control: redcell2 binary, same procedure ($(el))"
BIN=$E/m4/redcell2/mistralrs-redcell2
if srv_start oom-old $M/redcell-26b $RED "" "${RA[@]}"; then
  timeout 1200 python3 $I/oomgate.py $PORT oom-old $RP
  srv_stop
  echo "OOM lines $(grep -c OUT_OF_MEMORY $SLOG), recoveries $(grep -c 'titan: CUDA out of memory' $SLOG), panics $(grep -c panicked $SLOG)"
  sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -E "Model failed|titan: CUDA out of memory|panicked" | cut -c1-220
fi
BIN=$NEWBIN

echo "== [R1] 35B gate pair ($(el))"
coll integ3-m2 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2 && cmp40 $E/m6/out/h-off.json $E/m3/out/integ3-m2.json
coll integ3-mf0 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=0 && cmp40 $E/m6/out/mtpfile-off.json $E/m3/out/integ3-mf0.json
echo "== [R2] Bonsai-2 B2-off, IQ2_M, Bonsai-27B Q1_0 ($(el))"
if srv_start b-off $M/bonsai2-27b Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf "TITAN_MTP=0 TITAN_REASONING_EFFORT=medium" --max-seq-len 65536; then
  $C $CL greedy titan $PORT b-off 8 300; srv_stop
fi
$C $CL cmp o-off b-off
if srv_start i-new $M/q35-lowbit Qwen3.6-35B-A3B-UD-IQ2_M.gguf "TITAN_MTP=2 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64" --max-seq-len 65536; then
  $C $CL greedy titan $PORT i-new 8 300; srv_stop
fi
$C $CL cmp i-integ2 i-new; $C $CL cmp i-integ i-new
if srv_start q-new $M/bonsai/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf "TITAN_ATTN_FLASH_PREFILL=1" --max-seq-len 32768; then
  $C $CL greedy titan $PORT q-new 8 300; srv_stop
fi
$C $CL cmp q-integ2 q-new; $C $CL cmp q-integ q-new

echo "== [S] Spark-X2.5 Q4_K_M, 16384, integ request order ($(el))"
I2O=$E/top-integ2/m4/integ2/out
if srv_start spark $M/spark-x2.5 Spark-X2.5-4B-Q4_K_M.gguf "" --max-model-len 16384 --max-seq-len 16384; then
  for c in 2000 15000 33000; do gate g5 sp-g5-$c $c 128; done
  gate g3 sp-g3; gate g1 sp-g1 $SP/prompts/spark.json; gate g2 sp-g2 $SP/prompts/spark.json 256
  srv_stop
fi
tmap spark $O/sp-g1.json $O/sp-g3.json > /dev/null
same $O/sp-g2.json $I2O/sp-g2.json; same $O/sp-g1.json $I2O/sp-g1.json; same $O/sp-g3.json $I2O/sp-g3.json
cmpg g1 $O/sp-g1.json $R/refc-spark-g1.json | head -1
g5s $O/sp-g5-2000.json $O/sp-g5-15000.json $O/sp-g5-33000.json

echo "== [L] Spark LoRA G6 on/off vs integ2's recorded run ($(el))"
DQ=$E/m4/deq
if srv_start spl $M/spark-x2.5 Spark-X2.5-4B-Q4_K_M.gguf "" --max-model-len 8192 --max-seq-len 8192 --lora spark=$M/spark-lora-r16; then
  gate g6 spl-g6-off $G4/prompts/g6.json off; gate g6 spl-g6-on $G4/prompts/g6.json on
  gate g6 spl-g1-off $DQ/spark-g1-rows.json off; gate g6 spl-g1-on $DQ/spark-g1-rows.json on
  srv_stop
fi
tmap spark $O/spl-g1-off.json $O/spl-g1-on.json > /dev/null
for f in g6-off g6-on g1-off g1-on; do same $O/spl-$f.json $DQ/out/spl-old-$f.json; done

echo "== [I] IQ3_S 9B e2e vs integ2 ($(el))"
for T in iq3m embiq3s; do
  F=$E/m4/iq3s/Qwen3.5-9B-IQ3_M.gguf; [ $T = embiq3s ] && F=$E/m4/iq3s/Qwen3.5-9B-IQ3_M-embIQ3S.gguf
  if srv_start iq3s-$T $(dirname $F) $(basename $F) "" --max-seq-len 4096; then
    (cd $I/iq3s && timeout 900 python3 e2e.py mistral $PORT $T); srv_stop
    (cd $I/iq3s && python3 e2e.py cmp $T $F | tail -4)
    ref=$E/top-integ2/m4/integ2/iq3s/out
    python3 -c "import json,sys;a,b=json.load(open(sys.argv[1])),json.load(open(sys.argv[2]));print('E2E vs integ2',sys.argv[1].split('/')[-1],'greedy texts identical',sum(x['text']==y['text'] for x,y in zip(a['greedy'],b['greedy'])),'/',len(a['greedy']),'; tf byte-equal',sum(x==y for x,y in zip(a['tf'],b['tf'])),'/',len(a['tf']))" $I/iq3s/out/${T}_mistral.json $ref/${T}_mistral.json
  fi
done
free -m | head -2
echo "== integ3-window1 end $(el)"
