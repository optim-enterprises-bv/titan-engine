#!/bin/bash
# redcell2 window 1 (of 3). Root cause found by reading: the oxide twin of launch_mmq_gguf_<t>_moe ignored stride_col_dst
# (used nrows_x as the dst column stride); fast_mmq::grouped_pair_packed writes gate and up into one buffer with column
# stride 2 * nrows, so they overwrote each other and half the buffer stayed unwritten. Fixed in oxide-redcell2 quant-c.
#  [B] mistral.rs build (grouped route default for unaligned K, TITAN_MOE_XCHECK)
#  [K] oxide builds + kernel gates: mistralrs-quant-c (q3_k / q4_0 / q4_k incl. the new packed-stride MoE cases) and
#      fmt_mmq iq4_nl (incl. the new MoE-mode cases, K = 704); committed PTX restored afterwards
#  [X] TITAN_MOE_XCHECK=1: per-layer max |grouped - per-token| on 547- and 3547-token prompts
#  [A] G1, G3 (pc 0) and G5 on the default (grouped) route; G3 on TITAN_MOE_UNALIGNED_DECODE=1 (per-token) for comparison
#  [D] determinism: 2 fresh servers, baseline procedure (default cache, G1 -> G3 -> G2 256)
#  [G] gemma4-12b G1 vs integ2   [R] regression core vs integ2 references   [N] ncu
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/redcell2/window1.sh
source $HOME/titan-engine/m4/redcell2/lib.sh
exec > >(tee -a $I/window1-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin redcell2-window1 58
echo "src: $(git -C $SRC log --oneline -1 | cut -c1-70) $(git -C $SRC diff --stat | tail -1)"
echo "oxide: $(git -C $OX log --oneline -1 | cut -c1-60) $(git -C $OX diff --stat | tail -1)"
C=python3; CL=$I/b2client.py
g5s() { for f in "$@"; do [ -e $f ] && python3 -c "import json,sys;r=json.load(open(sys.argv[1]));print('TPS',sys.argv[1].split('/')[-1],[(x['prompt_tokens'],round(x['prefill_tps'] or 0,1),round(x['decode_tps'] or 0,1)) for x in r])" $f; done; }
guard() { grep -c "non-finite output" $O/$1.server.log; }
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

echo "== [K] oxide builds + kernel gates ($(el))"
oxbuild mistralrs-quant-c red-ox 1200 > $O/oxbuild-qc.status 2>&1 &
P1=$!
oxbuild fmt_mmq red-oxcq 1200 > $O/oxbuild-fm.status 2>&1 &
P2=$!
echo "== [X] XCHECK while the oxide crates build ($(el))"
if srv_start xc $M/redcell-26b $RED "TITAN_MOE_XCHECK=1" "${RA[@]}" --prefix-cache-n 0; then
  timeout 300 python3 $I/diag.py $PORT 2000 2; timeout 300 python3 $I/diag.py $PORT 13500 2; srv_stop
  grep -o "moe xcheck.*" $O/xc.server.log | head -70; echo "guard firings: $(guard xc)"
fi
wait $P1 $P2; cat $O/oxbuild-qc.status $O/oxbuild-fm.status
(cd $OX/mistralrs-quant-c && MRQC_ONLY=q3_k,q4_0,q4_k timeout 900 ./target/release/mistralrs-quant-c > $O/gate-qc.log 2>&1); echo "quant-c gate rc=$?"
grep -E "launcher calls|FAIL|PASS|failing" $O/gate-qc.log | tail -12
(cd $OX/fmt_mmq && timeout 900 ./target/release/fmt_mmq --quick --fmt=iq4_nl > $O/gate-fm.log 2>&1); echo "fmt_mmq gate rc=$?"
grep -E "moe-mmq|launches|FAIL|PASS|failing" $O/gate-fm.log | tail -8; grep -c "moe-mmq" $O/gate-fm.log
git -C $OX status --short | grep -v "^??"; git -C $OX checkout -- '*.ptx'; git -C $OX status --short | grep -v "^??"

echo "== [A] accuracy + speed ($(el))"
if srv_start fa $M/redcell-26b $RED "" "${RA[@]}" --prefix-cache-n 0; then
  gate g1 fa-g1 $RP; gate g3 fa-g3 2>&1 | tail -1; gate g5 fa-g5 13500 128 2>&1 | tail -1; gate g5 fa-g5s 2000 128 2>&1 | tail -1; srv_stop
  echo "guard firings: $(guard fa)"
fi
if srv_start fq $M/redcell-26b $RED "TITAN_MOE_UNALIGNED_DECODE=1" "${RA[@]}" --prefix-cache-n 0; then gate g3 fq-g3 2>&1 | tail -1; gate g5 fq-g5 13500 128 2>&1 | tail -1; srv_stop; fi
g5s $O/fa-g5.json $O/fa-g5s.json $O/fq-g5.json
[ -e $O/fa-g1.json ] && { tmap gemma4 $O/fa-g1.json; cmpg g1 $O/fa-g1.json $R/ref-red-g1.json; }
for f in fa-g3 fq-g3; do [ -e $O/$f.json ] && { tmap gemma4 $O/$f.json; cmpg g3 $O/$f.json $R/ref-red-g3.json; }; done

echo "== [D] determinism: 2 fresh servers, G1 -> G3 -> G2 ($(el))"
for rep in a b; do
  if srv_start fd-$rep $M/redcell-26b $RED "" "${RA[@]}"; then
    gate g1 fd-$rep-g1 $RP; gate g3 fd-$rep-g3 2>&1 | tail -1; gate g2 fd-$rep-g2 $RP 256 2>&1 | tail -1; srv_stop
    echo "guard firings: $(guard fd-$rep)"
  fi
done
for g in g1 g3 g2; do [ -e $O/fd-a-$g.json ] && [ -e $O/fd-b-$g.json ] && same $O/fd-a-$g.json $O/fd-b-$g.json; done
same $O/fd-a-g2.json $E/m4/g4s/out/baseline-redcell-g2.json

echo "== [G] gemma4-12b G1 vs integ2 ($(el))"
if srv_start g12 $M/gemma4-12b-qat gemma-4-12b-it-qat-q4_0.gguf "" --max-model-len 8192 --max-seq-len 8192 -n 0:48 --dtype bf16; then
  gate g1 g12-g1 $G4/prompts/gemma4-12b.json; srv_stop
  same $O/g12-g1.json $O/g12-integ2-g1.json
fi
echo "== [R] regression core ($(el))"
coll rc2-m2 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2 && cmp40 $E/m6/out/h-off.json $E/m3/out/rc2-m2.json
coll rc2-mf0 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=0 && cmp40 $E/m6/out/mtpfile-off.json $E/m3/out/rc2-mf0.json
if srv_start b-off $M/bonsai2-27b Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf "TITAN_MTP=0 TITAN_REASONING_EFFORT=medium" --max-seq-len 65536; then
  $C $CL greedy titan $PORT b-off 8 300; $C $CL cmp o-off b-off; srv_stop
fi
if srv_start i-new $M/q35-lowbit Qwen3.6-35B-A3B-UD-IQ2_M.gguf "TITAN_MTP=2 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64" --max-seq-len 65536; then
  $C $CL greedy titan $PORT i-new 8 300; srv_stop; $C $CL cmp i-integ2 i-new
fi
if srv_start q-new $M/bonsai/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf "TITAN_ATTN_FLASH_PREFILL=1" --max-seq-len 32768; then
  $C $CL greedy titan $PORT q-new 8 300; srv_stop; $C $CL cmp q-integ2 q-new
fi
echo "== [N] ncu, ~3.5k prompt ($(el))"
if [ $(left) -gt 420 ] && ncu_start ncu-new "" -m $M/redcell-26b -f $RED "${RA[@]}" --prefix-cache-n 0; then
  timeout 300 python3 $SP/req.py $PORT titan 200 1; timeout 400 python3 $SP/req.py $PORT titan 13500 1
  ncu_end
fi
echo "== window1 done ($(el))"
