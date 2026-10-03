#!/bin/bash
# redcell window 2 (of 3): window 1's tests again on the fixed build (pieces now force_contiguous: window 1's G3 panicked on
# a narrowed, offset ids buffer and poisoned the engine), then the regression core.
#  [B] build  [E] MEMLOG probes new binary, 2 fresh servers  [D] determinism: 3 fresh servers, G1 -> G3 -> G2 256 (+G5 on a)
#  [P] G5 before/after: integ and new, --prefix-cache-n 0  [A] accuracy vs llama.cpp  [N] ncu new  [G] gemma4-12b G1 vs integ
#  [R] regression core: 35B MTP=2, MTP-off file, Bonsai-2 B2-off, IQ2_M and Bonsai-27B Q1_0 new vs integ runs
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/redcell/window2.sh
source $HOME/titan-engine/m4/redcell/lib.sh
exec > >(tee -a $I/window2-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin redcell-window2 58
echo "src: $(git -C $SRC log --oneline -1 | cut -c1-70) $(git -C $SRC diff --stat | tail -1)"
echo "oxide: $(git -C $OX log --oneline -1 | cut -c1-60) $(git -C $OX diff --stat | tail -1)"
C=python3; CL=$I/b2client.py
g5s() { for f in "$@"; do [ -e $f ] && python3 -c "import json,sys;r=json.load(open(sys.argv[1]));print('TPS',sys.argv[1].split('/')[-1],[(x['prompt_tokens'],round(x['prefill_tps'] or 0,1),round(x['decode_tps'] or 0,1)) for x in r])" $f; done; }

echo "== [B] mistral.rs build ($(el))"
rm -f $I/rebuild.flag
until mbuild 1500; do
  [ $(left) -gt 1800 ] || { echo "no binary: ending the window"; exit 1; }
  echo "build failed: waiting for $I/rebuild.flag ($(left)s left)"; date -Is > $I/out/build-failed.flag
  while [ ! -e $I/rebuild.flag ] && [ $(left) -gt 1800 ]; do sleep 10; done
  [ -e $I/rebuild.flag ] || { echo "no binary: ending the window"; exit 1; }
  rm -f $I/rebuild.flag $I/out/build-failed.flag
done
cp $MBIN $NEWBIN; echo "binary sha256 $(sha256sum $NEWBIN | cut -c1-16)"
BIN=$NEWBIN

echo "== [E] MEMLOG probes, new binary ($(el))"
for rep in a b; do
  if srv_start ev-new-$rep $M/redcell-26b $RED "TITAN_G4_MEMLOG=2" "${RA[@]}" --prefix-cache-n 0; then
    timeout 300 python3 $I/one.py $PORT $RP 0 8; srv_stop
  fi
done
python3 $I/firstdiff.py $O/ev-new-a.server.log $O/ev-new-b.server.log

echo "== [D] determinism: 3 fresh servers, G1 -> G3 -> G2 ($(el))"
for rep in a b c; do
  if srv_start rd-$rep $M/redcell-26b $RED "" "${RA[@]}"; then
    gate g1 rd-$rep-g1 $RP; gate g3 rd-$rep-g3 2>&1 | tail -1; gate g2 rd-$rep-g2 $RP 256 2>&1 | tail -1
    [ $rep = a ] && gate g5 rd-a-g5 13500 128 2>&1 | tail -1
    srv_stop
  fi
done
for p in "a b" "a c"; do set -- $p
  for g in g1 g3 g2; do [ -e $O/rd-$1-$g.json ] && [ -e $O/rd-$2-$g.json ] && same $O/rd-$1-$g.json $O/rd-$2-$g.json; done
done

echo "== [P] G5 before/after, --prefix-cache-n 0 ($(el))"
if srv_start p-new $M/redcell-26b $RED "" "${RA[@]}" --prefix-cache-n 0; then gate g5 p-new-g5 13500 128 2>&1 | tail -1; gate g5 p-new-g5s 2000 128 2>&1 | tail -1; srv_stop; fi
BIN=$INTEG
if srv_start p-integ $M/redcell-26b $RED "" "${RA[@]}" --prefix-cache-n 0; then gate g5 p-integ-g5 13500 128 2>&1 | tail -1; gate g5 p-integ-g5s 2000 128 2>&1 | tail -1; srv_stop; fi
BIN=$NEWBIN
g5s $O/rd-a-g5.json $O/p-new-g5.json $O/p-integ-g5.json $O/p-new-g5s.json $O/p-integ-g5s.json

echo "== [A] accuracy vs llama.cpp ($(el))"
[ -e $O/rd-a-g1.json ] && { tmap gemma4 $O/rd-a-g1.json; cmpg g1 $O/rd-a-g1.json $R/ref-red-g1.json; }
[ -e $O/rd-a-g3.json ] && { tmap gemma4 $O/rd-a-g3.json; cmpg g3 $O/rd-a-g3.json $R/ref-red-g3.json; }
[ -e $O/rd-a-g2.json ] && cmpg g2 $O/rd-a-g2.json $R/ref-red-g2.json | tail -3

echo "== [N] ncu new binary, ~3.5k prompt ($(el))"
if [ $(left) -gt 1500 ] && ncu_start ncu-new "" -m $M/redcell-26b -f $RED "${RA[@]}" --prefix-cache-n 0; then
  timeout 600 python3 $SP/req.py $PORT titan 200 1; timeout 900 python3 $SP/req.py $PORT titan 13500 1
  ncu_end
fi

echo "== [G] gemma4-12b G1 vs integ ($(el))"
if srv_start g12 $M/gemma4-12b-qat gemma-4-12b-it-qat-q4_0.gguf "" --max-model-len 8192 --max-seq-len 8192 -n 0:48 --dtype bf16; then
  gate g1 g12-g1 $G4/prompts/gemma4-12b.json; srv_stop
  same $O/g12-g1.json $E/m4/integ/out/g12-g1.json
fi

echo "== [R] regression core ($(el))"
coll red-m2 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2 && cmp40 $E/m6/out/h-off.json $E/m3/out/red-m2.json
coll red-mf0 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=0 && cmp40 $E/m6/out/mtpfile-off.json $E/m3/out/red-mf0.json
if srv_start b-off $M/bonsai2-27b Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf "TITAN_MTP=0 TITAN_REASONING_EFFORT=medium" --max-seq-len 65536; then
  $C $CL greedy titan $PORT b-off 8 300; $C $CL cmp o-off b-off; srv_stop
fi
if srv_start i-new $M/q35-lowbit Qwen3.6-35B-A3B-UD-IQ2_M.gguf "TITAN_MTP=2 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64" --max-seq-len 65536; then
  $C $CL greedy titan $PORT i-new 8 300; srv_stop; $C $CL cmp i-integ i-new
fi
if srv_start q-new $M/bonsai/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf "TITAN_ATTN_FLASH_PREFILL=1" --max-seq-len 32768; then
  $C $CL greedy titan $PORT q-new 8 300; srv_stop; $C $CL cmp q-integ q-new
fi
echo "== window2 done ($(el))"
