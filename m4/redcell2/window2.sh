#!/bin/bash
# redcell2 window 2 (of 3): gates on the guard-off build, kernel gates (quant-c all MMQ types incl. the packed-stride MoE
# check; fmt_mmq iq4_nl incl. MoE mode), piece size, extended long-prompt accuracy (g3x), determinism, regression core.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/redcell2/window2.sh
source $HOME/titan-engine/m4/redcell2/lib.sh
exec > >(tee -a $I/window2-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin redcell2-window2 58
echo "src: $(git -C $SRC log --oneline -1 | cut -c1-70) $(git -C $SRC diff --stat | tail -1)"
echo "oxide: $(git -C $OX log --oneline -1 | cut -c1-60) $(git -C $OX diff --stat | tail -1)"
systemctl --user stop red-ref 2>/dev/null
C=python3; CL=$I/b2client.py
g5s() { for f in "$@"; do [ -e $f ] && python3 -c "import json,sys;r=json.load(open(sys.argv[1]));print('TPS',sys.argv[1].split('/')[-1],[(x['prompt_tokens'],round(x['prefill_tps'] or 0,1),round(x['decode_tps'] or 0,1)) for x in r])" $f; done; }
g3x() { [ $(left) -gt 300 ] && timeout 900 python3 $I/gates2.py g3x $PORT titan $O/$1.json 13500 4 12 2>&1 | tail -1; }
rm -f $I/rebuild.flag
echo "== [B] mistral.rs build ($(el))"
until mbuild 1500; do
  [ $(left) -gt 2100 ] || { echo "no binary: ending the window"; exit 1; }
  echo "build failed: waiting for $I/rebuild.flag ($(left)s left)"; date -Is > $I/out/build-failed.flag
  while [ ! -e $I/rebuild.flag ] && [ $(left) -gt 2100 ]; do sleep 5; done
  [ -e $I/rebuild.flag ] || { echo "no binary: ending the window"; exit 1; }
  rm -f $I/rebuild.flag $I/out/build-failed.flag
done
cp $MBIN $NEWBIN; echo "binary sha256 $(sha256sum $NEWBIN | cut -c1-16)"
BIN=$NEWBIN
echo "== [K] kernel gates ($(el))"
oxbuild mistralrs-quant-c red-ox 1200
(cd $OX/mistralrs-quant-c && MRQC_ONLY=q4_0,q4_1,q5_0,q5_1,q8_0,q2_k,q3_k,q4_k,q5_k,q6_k timeout 1200 ./target/release/mistralrs-quant-c > $O/gate-qc2.log 2>&1); echo "quant-c gate rc=$?"
grep -E "_moe|launcher calls|FAIL|PASS|failing" $O/gate-qc2.log | tail -26
git -C $OX checkout -- '*.ptx'
(cd $OX/fmt_mmq && timeout 900 ./target/release/fmt_mmq --fmt=iq4_nl > $O/gate-fm2.log 2>&1); echo "fmt_mmq gate rc=$?"
grep -E "launches|FAIL|PASS|failing" $O/gate-fm2.log | tail -6; echo "moe-mmq failures: $(grep -c 'moe-mmq' $O/gate-fm2.log)"
git -C $OX status --short | grep -v "^??"

echo "== [P] speed (pc 0): piece 2048 (default) vs 1024; accuracy g3 + g3x ($(el))"
if srv_start pa $M/redcell-26b $RED "" "${RA[@]}" --prefix-cache-n 0; then
  gate g5 pa-g5 13500 128 2>&1 | tail -1; gate g5 pa-g5s 2000 128 2>&1 | tail -1; gate g5 pa-g5l 26000 64 2>&1 | tail -1
  gate g1 pa-g1 $RP; gate g3 pa-g3 2>&1 | tail -1; g3x pa-g3x; srv_stop
fi
if srv_start pb $M/redcell-26b $RED "TITAN_MOE_PIECE=1024" "${RA[@]}" --prefix-cache-n 0; then gate g5 pb-g5 13500 128 2>&1 | tail -1; gate g5 pb-g5l 26000 64 2>&1 | tail -1; srv_stop; fi
if srv_start pq $M/redcell-26b $RED "TITAN_MOE_UNALIGNED_DECODE=1" "${RA[@]}" --prefix-cache-n 0; then g3x pq-g3x; srv_stop; fi
g5s $O/pa-g5.json $O/pa-g5s.json $O/pa-g5l.json $O/pb-g5.json $O/pb-g5l.json
[ -e $O/pa-g1.json ] && { tmap gemma4 $O/pa-g1.json; cmpg g1 $O/pa-g1.json $R/ref-red-g1.json; }
[ -e $O/pa-g3.json ] && { tmap gemma4 $O/pa-g3.json; cmpg g3 $O/pa-g3.json $R/ref-red-g3.json; }
for f in pa-g3x pq-g3x; do [ -e $O/$f.json ] && tmap gemma4 $O/$f.json; done

echo "== [D] determinism: 2 fresh servers, prefix cache 0 (the roster's REDCELL setting), G1 -> G3 -> G2 ($(el))"
for rep in a b; do
  if srv_start dz-$rep $M/redcell-26b $RED "" "${RA[@]}" --prefix-cache-n 0; then
    gate g1 dz-$rep-g1 $RP; gate g3 dz-$rep-g3 2>&1 | tail -1; gate g2 dz-$rep-g2 $RP 256 2>&1 | tail -1; srv_stop
  fi
done
for g in g1 g3 g2; do [ -e $O/dz-a-$g.json ] && [ -e $O/dz-b-$g.json ] && same $O/dz-a-$g.json $O/dz-b-$g.json; done
echo "== [D2] default prefix cache, baseline procedure, piece 1024 ($(el))"
if srv_start dd $M/redcell-26b $RED "TITAN_MOE_PIECE=1024" "${RA[@]}"; then
  gate g1 dd-g1 $RP; gate g3 dd-g3 2>&1 | tail -1; gate g2 dd-g2 $RP 256 2>&1 | tail -1; srv_stop
fi
[ -e $O/dd-g2.json ] && same $O/dd-g2.json $E/m4/g4s/out/baseline-redcell-g2.json
[ -e $O/dd-g1.json ] && same $O/dd-g1.json $O/dz-a-g1.json

echo "== [G] gemma4-12b G1 vs integ2 ($(el))"
if srv_start g12 $M/gemma4-12b-qat gemma-4-12b-it-qat-q4_0.gguf "" --max-model-len 8192 --max-seq-len 8192 -n 0:48 --dtype bf16; then
  gate g1 g12-g1 $G4/prompts/gemma4-12b.json; srv_stop
  python3 -c "import json,sys;a=json.load(open(sys.argv[1]));b=json.load(open(sys.argv[2]));print('G12 logprob-identical vs integ2',sum(all(x[1]==y[1] for x,y in zip(p['top'][0],q['top'][0])) and p['text']==q['text'] for p,q in zip(a,b)),'/',len(a))" $O/g12-g1.json $O/g12-integ2-g1.json
fi
echo "== [R] regression core ($(el))"
coll rc2b-m2 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2 && cmp40 $E/m6/out/h-off.json $E/m3/out/rc2b-m2.json
coll rc2b-mf0 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=0 && cmp40 $E/m6/out/mtpfile-off.json $E/m3/out/rc2b-mf0.json
if srv_start b-off2 $M/bonsai2-27b Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf "TITAN_MTP=0 TITAN_REASONING_EFFORT=medium" --max-seq-len 65536; then
  $C $CL greedy titan $PORT b-off2 8 300; $C $CL cmp o-off b-off2; srv_stop
fi
if srv_start i-new2 $M/q35-lowbit Qwen3.6-35B-A3B-UD-IQ2_M.gguf "TITAN_MTP=2 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64" --max-seq-len 65536; then
  $C $CL greedy titan $PORT i-new2 8 300; srv_stop; $C $CL cmp i-integ2 i-new2
fi
if srv_start q-new2 $M/bonsai/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf "TITAN_ATTN_FLASH_PREFILL=1" --max-seq-len 32768; then
  $C $CL greedy titan $PORT q-new2 8 300; srv_stop; $C $CL cmp q-integ2 q-new2
fi
echo "== [N] ncu ($(el))"
if [ $(left) -gt 420 ] && ncu_start ncu-final "" -m $M/redcell-26b -f $RED "${RA[@]}" --prefix-cache-n 0; then
  timeout 300 python3 $SP/req.py $PORT titan 200 1; timeout 400 python3 $SP/req.py $PORT titan 13500 1
  ncu_end
fi
echo "== window2 done ($(el))"
