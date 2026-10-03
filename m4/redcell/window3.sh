#!/bin/bash
# redcell window 3 (of 3): window 2's grouped prefill completed every kernel of a 3.5k prompt (ncu) but the request
# returned 500 (72-token prompts were fine).
#  [B] build (TITAN_MOE_UNALIGNED_DECODE=iq4_nl|q4_0 per-format toggles)
#  [X] diagnosis, fresh server per config, prefix cache 0, TITAN_G4_MEMLOG=2, 547- and 3547-token prompts:
#      decode route (=1) as reference, grouped, iq4_nl-on-decode, q4_0-on-decode; nan/inf + per-layer expert sums
#  [W] wait for $I/fix.flag (rebuild + re-diagnose) or $I/go.flag (keep the binary) or $I/env.flag (contents: default
#      server env for the tests, e.g. TITAN_MOE_UNALIGNED_DECODE=iq4_nl)
#  [T] final gates: determinism 3 servers (G1 -> G3 -> G2), G5 + accuracy, ncu, gemma4-12b, regression core
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/redcell/window3.sh
source $HOME/titan-engine/m4/redcell/lib.sh
exec > >(tee -a $I/window3-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin redcell-window3 58
echo "src: $(git -C $SRC log --oneline -1 | cut -c1-70) $(git -C $SRC diff --stat | tail -1)"
C=python3; CL=$I/b2client.py
g5s() { for f in "$@"; do [ -e $f ] && python3 -c "import json,sys;r=json.load(open(sys.argv[1]));print('TPS',sys.argv[1].split('/')[-1],[(x['prompt_tokens'],round(x['prefill_tps'] or 0,1),round(x['decode_tps'] or 0,1)) for x in r])" $f; done; }
rm -f $I/fix.flag $I/go.flag $I/env.flag $I/rebuild.flag
build_loop() {
  until mbuild 1500; do
    [ $(left) -gt 1800 ] || { echo "no binary: ending the window"; exit 1; }
    echo "build failed: waiting for $I/rebuild.flag ($(left)s left)"; date -Is > $I/out/build-failed.flag
    while [ ! -e $I/rebuild.flag ] && [ $(left) -gt 1800 ]; do sleep 5; done
    [ -e $I/rebuild.flag ] || { echo "no binary: ending the window"; exit 1; }
    rm -f $I/rebuild.flag $I/out/build-failed.flag
  done
  cp $MBIN $NEWBIN; echo "binary sha256 $(sha256sum $NEWBIN | cut -c1-16) src $(git -C $SRC log --oneline -1 | cut -c1-12) $(git -C $SRC diff --stat | tail -1)"
}
diag() { # TAG ENV
  if srv_start x-$1 $M/redcell-26b $RED "TITAN_G4_MEMLOG=2 $2" "${RA[@]}" --prefix-cache-n 0; then
    timeout 300 python3 $I/diag.py $PORT 2000 2; srv_stop
  fi
  if srv_start y-$1 $M/redcell-26b $RED "$2" "${RA[@]}" --prefix-cache-n 0; then
    timeout 300 python3 $I/diag.py $PORT 13500 2; timeout 300 python3 $I/diag.py $PORT 2000 2; srv_stop
  fi
}
echo "== [B] build ($(el))"; build_loop; BIN=$NEWBIN

echo "== [X] diagnosis ($(el))"
diag ref TITAN_MOE_UNALIGNED_DECODE=1
diag grp ""
diag iq4dec TITAN_MOE_UNALIGNED_DECODE=iq4_nl
diag q40dec TITAN_MOE_UNALIGNED_DECODE=q4_0
for t in grp iq4dec q40dec; do echo "-- probes $t vs ref"; python3 $I/nanprobe.py $O/x-ref.server.log $O/x-$t.server.log | head -75; done
date -Is > $I/out/diag-done.flag

echo "== [W] waiting for fix.flag / go.flag ($(el))"
while [ ! -e $I/fix.flag ] && [ ! -e $I/go.flag ] && [ $(left) -gt 2100 ]; do sleep 5; done
if [ -e $I/fix.flag ]; then
  rm -f $I/fix.flag; echo "== [F] rebuild with the fix ($(el))"; build_loop
  diag fix ""
  python3 $I/nanprobe.py $O/x-ref.server.log $O/x-fix.server.log | head -75
fi
TENV=""; [ -e $I/env.flag ] && TENV=$(cat $I/env.flag); echo "test env: '$TENV' ($(el))"

echo "== [T1] determinism: 3 fresh servers, G1 -> G3 -> G2 ($(el))"
for rep in a b c; do
  if srv_start fd-$rep $M/redcell-26b $RED "$TENV" "${RA[@]}"; then
    gate g1 fd-$rep-g1 $RP; gate g3 fd-$rep-g3 2>&1 | tail -1; gate g2 fd-$rep-g2 $RP 256 2>&1 | tail -1; srv_stop
  fi
done
for p in "a b" "a c"; do set -- $p
  for g in g1 g3 g2; do [ -e $O/fd-$1-$g.json ] && [ -e $O/fd-$2-$g.json ] && same $O/fd-$1-$g.json $O/fd-$2-$g.json; done
done
echo "== [T2] G5 + G3, prefix cache 0 ($(el))"
if srv_start fp $M/redcell-26b $RED "$TENV" "${RA[@]}" --prefix-cache-n 0; then
  gate g5 fp-g5 13500 128 2>&1 | tail -1; gate g5 fp-g5s 2000 128 2>&1 | tail -1; gate g3 fp-g3 2>&1 | tail -1; srv_stop
fi
g5s $O/fp-g5.json $O/fp-g5s.json $O/p-integ-g5.json $O/p-integ-g5s.json
[ -e $O/fd-a-g1.json ] && { tmap gemma4 $O/fd-a-g1.json; cmpg g1 $O/fd-a-g1.json $R/ref-red-g1.json; }
for f in fd-a-g3 fp-g3; do [ -e $O/$f.json ] && { tmap gemma4 $O/$f.json; cmpg g3 $O/$f.json $R/ref-red-g3.json; }; done
[ -e $O/fd-a-g2.json ] && cmpg g2 $O/fd-a-g2.json $R/ref-red-g2.json | tail -2
echo "== [T3] gemma4-12b G1 ($(el))"
if srv_start g12f $M/gemma4-12b-qat gemma-4-12b-it-qat-q4_0.gguf "" --max-model-len 8192 --max-seq-len 8192 -n 0:48 --dtype bf16; then
  gate g1 g12f-g1 $G4/prompts/gemma4-12b.json; srv_stop
  python3 -c "import json,sys;a=json.load(open(sys.argv[1]));b=json.load(open(sys.argv[2]));print('G12 logprob-identical vs integ',sum(all(x[1]==y[1] for x,y in zip(p['top'][0],q['top'][0])) and p['text']==q['text'] for p,q in zip(a,b)),'/',len(a))" $O/g12f-g1.json $E/m4/integ/out/g12-g1.json
fi
echo "== [T4] regression core ($(el))"
coll redf-m2 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2 && cmp40 $E/m6/out/h-off.json $E/m3/out/redf-m2.json
coll redf-mf0 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=0 && cmp40 $E/m6/out/mtpfile-off.json $E/m3/out/redf-mf0.json
if srv_start bf-off $M/bonsai2-27b Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf "TITAN_MTP=0 TITAN_REASONING_EFFORT=medium" --max-seq-len 65536; then
  $C $CL greedy titan $PORT bf-off 8 300; $C $CL cmp o-off bf-off; srv_stop
fi
if srv_start if-new $M/q35-lowbit Qwen3.6-35B-A3B-UD-IQ2_M.gguf "TITAN_MTP=2 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64" --max-seq-len 65536; then
  $C $CL greedy titan $PORT if-new 8 300; srv_stop; $C $CL cmp i-integ if-new
fi
if srv_start qf-new $M/bonsai/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf "TITAN_ATTN_FLASH_PREFILL=1" --max-seq-len 32768; then
  $C $CL greedy titan $PORT qf-new 8 300; srv_stop; $C $CL cmp q-integ qf-new
fi
echo "== [T5] ncu final binary ($(el))"
if [ $(left) -gt 420 ] && ncu_start ncu-final "$TENV" -m $M/redcell-26b -f $RED "${RA[@]}" --prefix-cache-n 0; then
  timeout 300 python3 $SP/req.py $PORT titan 200 1; timeout 400 python3 $SP/req.py $PORT titan 13500 1
  ncu_end
fi
echo "== window3 done ($(el))"
