#!/bin/bash
# g4acc window 2: [K] gate re-run (decode-row mutation fix); [B] build (empty F32-preallocated cache dropped before
# the F16 append); [E] P2 (--dtype f32, F16 KV, f16 flash) gemma4-12b G3 ~4k/~8k, G1, G5, KTRACE; G2 + llama tie;
# integ2 binary G5 decode reference; [D] P2 dump vs llama.cpp; [RC] REDCELL P2 G3 x2 + G1.
set -u
source $HOME/titan-engine/m4/g4acc/lib.sh
exec > >(tee -a $I/window2-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin window2 58
K=$OX/flash-prefill; G12D=$M/gemma4-12b-qat; G12=gemma-4-12b-it-qat-q4_0.gguf; GP=$G4/prompts/gemma4-12b.json
same() { python3 $I/same.py "$@"; }
tps() { for f in "$@"; do python3 -c "import json,sys;r=json.load(open(sys.argv[1]));print('TPS',sys.argv[1].split('/')[-1],[(x['prompt_tokens'],round(x['prefill_tps'] or 0,1),round(x['decode_tps'] or 0,1)) for x in r])" $f; done; }
CTX=(--max-model-len 12288 --max-seq-len 12288 -n 0:48 --prefix-cache-n 0)
echo "src: $(git -C $SRC log --oneline -1 | cut -c1-60) $(git -C $SRC diff --quiet && echo clean || echo DIRTY)"

echo "== [K] gate re-run ($(el))"
cp $K/flash_prefill.ptx $O/fp-w1.ptx
if oxbuild flash-prefill g4acc-ox 600; then
  cmp -s $K/flash_prefill.ptx $O/fp-w1.ptx && echo "PTX unchanged by the gate fix" || echo "!! PTX changed"
  (cd $K && timeout 900 ./target/release/flash-prefill > $O/gate2-fp.log 2>&1); echo "gate rc=$?"
  grep -vE "^  (flash|time)" $O/gate2-fp.log | grep -E "accuracy|splits|mutation|differ|gate|WEAK|FAIL" | tail -30
fi

echo "== [B] build ($(el))"
rm -f $I/rebuild.flag
until mbuild 1500; do
  echo "build failed: waiting for $I/rebuild.flag ($(left)s left)"; date -Is > $O/build-failed.flag
  while [ ! -e $I/rebuild.flag ] && [ $(left) -gt 1500 ]; do sleep 10; done
  [ -e $I/rebuild.flag ] || { echo "no binary"; exit 1; }
  rm -f $I/rebuild.flag $O/build-failed.flag
done
cp $MBIN $I/bin/mistralrs-w2; BIN=$I/bin/mistralrs-w2; echo "binary sha256 $(sha256sum $BIN | cut -c1-16)"

echo "== [E] P2 ($(el))"
if srv_start g12-q2 $G12D $G12 "TITAN_KTRACE=1" "${CTX[@]}" --dtype f32; then
  gate g3 q2-g3; gate g3 q2-g3l 28000; echo "VRAM after G3 8k: $(smi); OOM $(grep -c OUT_OF_MEMORY $SLOG)"
  gate g1 q2-g1 $GP; gate g5 q2-g5 13500 128; gate g5 q2-g5s 2000 128; gate g2 q2-g2 $GP 256
  srv_stop; grep "^KTRACE" $SLOG | sort > $O/q2.ktrace; echo "ktrace dequant kernels:"; grep -i "dequant" $O/q2.ktrace
  grep -E "ERROR" $SLOG | head -3
  tmap gemma4 $O/q2-g3.json $O/q2-g3l.json $O/q2-g1.json > /dev/null
  cmpg g3 $O/q2-g3.json $R/ref-g12-g3.json | head -1; cmpg g3 $O/q2-g3l.json $R/ref-g12-g3l.json | head -1
  cmpg g1 $O/q2-g1.json $R/ref-g12-g1.json | head -1; cmpg g2 $O/q2-g2.json $R/ref-g12-g2.json
  tps $O/q2-g3.json $O/q2-g3l.json $O/q2-g5.json $O/q2-g5s.json
fi
BIN=$INTEG
if srv_start g12-i2 $G12D $G12 "" "${CTX[@]}" --dtype bf16; then
  gate g5 i2-g5 13500 128; gate g5 i2-g5s 2000 128; gate g3 i2-g3l 28000; echo "VRAM: $(smi)"; srv_stop; tps $O/i2-g5.json $O/i2-g5s.json $O/i2-g3l.json
fi
BIN=$I/bin/mistralrs-w2
if lsrv l-tie $G12D/$G12 -c 8192 -ngl 99 -ctk f16 -ctv f16; then
  KIND=llama gate tie tie-q2 $GP $O/q2-g2.json $R/ref-g12-g2.json; lstop
fi

echo "== [D] P2 dump ($(el))"
mkdir -p $O/dump-q2
if srv_start g12-dq2 $G12D $G12 "TITAN_G4_DUMP=$O/dump-q2" "${CTX[@]}" --dtype f32; then
  timeout 300 python3 $I/dump/req1.py $PORT $I/dump/g3p0.txt; srv_stop
fi
python3 $I/dump/cmpdump.py $O/dump-q2 $E/m4/hd512/out/dump-lf16 $E/m4/hd512/out/dump-lbf16

echo "== [RC] REDCELL P2 ($(el))"
RD=$M/redcell-26b; RF=REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf
for n in rq2a rq2b; do
  if [ $(left) -gt 240 ] && srv_start red-$n $RD $RF "" --max-model-len 8192 --max-seq-len 8192 -n 0:30 --dtype f32 --prefix-cache-n 0; then
    gate g3 $n-g3; [ $n = rq2a ] && { gate g1 $n-g1 $G4/prompts/redcell.json; gate g5 $n-g5 13500 64; }; echo "VRAM: $(smi)"; srv_stop
    grep -E "ERROR" $SLOG | head -2
    tmap gemma4 $O/$n-g3.json > /dev/null; cmpg g3 $O/$n-g3.json $R/ref-red-g3.json | head -1; tps $O/$n-g3.json
  else srv_stop; fi
done
[ -e $O/rq2a-g1.json ] && tmap gemma4 $O/rq2a-g1.json > /dev/null && cmpg g1 $O/rq2a-g1.json $R/ref-red-g1.json | head -1 && tps $O/rq2a-g5.json
free -m | head -2
