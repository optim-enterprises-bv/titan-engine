#!/bin/bash
# integ2-20261003 window 2 (model gates on bin/mistralrs-titan-integ2, no build):
# [V] draft roster negative control (prefix_cache_n = -1 must be refused by from-config's parser);
# [G] gemma4-12b G1 (integ config) vs integ / hd512; G3 ~4k / ~8k --prefix-cache-n 0 vs the hd512 branch (f-g3, f-g3l);
# [RD] REDCELL determinism: 2 fresh servers, baseline procedure (default cache, G1 -> G3 -> G2), a vs b and vs
#      m4/g4s/out/baseline-redcell-g2.json; G1 vs llama.cpp; [RP] REDCELL --prefix-cache-n 0: G5 13500 / 2000 + G3;
# [P] from-config on the combined draft roster (roster-draft.toml = deploy/models.toml + models-toml-draft.diff, port
#     18640): 35B shared-prefix TTFT -> gemma4-12b 6 x ~4k -> 35B TTFT -> redcell-26b G3 (4 x ~4k, per-model 0);
# [I] IQ3_S 9B IQ3_M / embIQ3S e2e vs the iq3s branch and integ.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/top-integ2/m4/integ2/window2.sh
source $HOME/titan-engine/top-integ2/m4/integ2/lib.sh
exec > >(tee -a $I/window2-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin integ2-window2 58
BIN=$NEWBIN; echo "binary: $BIN sha256 $(sha256sum $BIN | cut -d' ' -f1); TITAN_* in env: $(env | grep -c '^TITAN_')"
H5=$E/m4/hd512/out; P=$I/pcache.py
G12D=$M/gemma4-12b-qat; G12=gemma-4-12b-it-qat-q4_0.gguf; GP=$G4/prompts/gemma4-12b.json
RED=REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf; RP=$G4/prompts/redcell.json; RA=(--max-model-len 8192 --max-seq-len 8192 -n 0:30 --dtype bf16)

echo "== [V] from-config parser negative control: roster-bad.toml (prefix_cache_n = -1) ($(el))"
timeout 60 systemd-run --user --unit=integ2-meta --collect --wait --pipe -q -p MemoryMax=4G -p MemorySwapMax=0 \
  --setenv=CUDA_VISIBLE_DEVICES= --setenv=LD_LIBRARY_PATH=$E/lib $BIN from-config --file $I/roster-bad.toml > $O/roster-bad.log 2>&1
echo "roster-bad rc=$? (non-zero expected)"; sed 's/\x1b\[[0-9;]*m//g' $O/roster-bad.log | grep -v " INFO " | tail -6 | cut -c1-300

echo "== [K3] iq3_s gate (window 1 lacked the untracked nvcc reference cubins in iq3_s/ref, copied from oxide-integ) ($(el))"
(cd $OX/iq3_s && timeout 600 ./target/release/iq3_s > $O/gate3-iq3s.log 2>&1); echo "iq3_s gate (committed PTX) rc=$?"
grep -E "FAIL|mutation|cpu oracle|PASS|launches" $O/gate3-iq3s.log | head -12
cp -a $OX/iq3_s/iq3_s.ptx $O/committed-iq3_s.ptx; cp -a $O/regen/iq3_s.ptx $OX/iq3_s/iq3_s.ptx
(cd $OX/iq3_s && timeout 600 ./target/release/iq3_s > $O/gate3r-iq3s.log 2>&1); echo "iq3_s gate (regenerated PTX) rc=$?"
grep -E "FAIL|PASS" $O/gate3r-iq3s.log | tail -2
git -C $OX checkout -- iq3_s/iq3_s.ptx; git -C $OX status --short | grep -v "^??"

echo "== [V2] cargo metadata --all-features: titan-oxide-ffi path ($(el))"
timeout 300 systemd-run --user --unit=integ2-meta --collect --wait --pipe -q -p MemoryMax=4G -p MemorySwapMax=0 -p WorkingDirectory=$SRC \
  --setenv=PATH=$HOME/.cargo/bin:/usr/bin:/bin cargo metadata --offline --all-features --format-version 1 > $O/metadata-all.json 2> $O/metadata-all.err
echo "metadata rc=$?"; tail -2 $O/metadata-all.err
python3 -c "
import json,sys,os
m=json.load(open(sys.argv[1]))
for p in m['packages']:
    if p['name'] in ('candle-core','candle-nn','titan-oxide-ffi'): print('RESOLVE',p['name'],p['manifest_path'],'->',os.path.realpath(p['manifest_path']))
" $O/metadata-all.json

echo "== [G1] gemma4-12b G1 (8192, 0:48, bf16, default cache) ($(el))"
if srv_start g12s $G12D $G12 "" --max-model-len 8192 --max-seq-len 8192 -n 0:48 --dtype bf16; then
  gate g1 g12-g1 $GP; srv_stop
  grep -E "titan gemma4 prompt attention" $SLOG | sed 's/.*INFO //' | head -1
fi
tmap gemma4 $O/g12-g1.json > /dev/null
same $O/g12-g1.json $IO/g12-g1.json; same $O/g12-g1.json $H5/f-g1.json; cmpg g1 $O/g12-g1.json $R/ref-g12-g1.json | head -1
echo "== [G3] gemma4-12b G3 ~4k / ~8k, 12288, --prefix-cache-n 0 ($(el))"
if srv_start g12p $G12D $G12 "" --max-model-len 12288 --max-seq-len 12288 -n 0:48 --dtype bf16 --prefix-cache-n 0; then
  gate g3 g12-g3; echo "VRAM after 4k: $(smi)"; gate g3 g12-g3l 28000; echo "VRAM after 8k: $(smi); OOM lines $(grep -c OUT_OF_MEMORY $SLOG)"; srv_stop
fi
tmap gemma4 $O/g12-g3.json $O/g12-g3l.json > /dev/null
cmpg g3 $O/g12-g3.json $R/ref-g12-g3.json | head -1; cmpg g3 $O/g12-g3l.json $R/ref-g12-g3l.json | head -1
same $O/g12-g3.json $H5/f-g3.json; same $O/g12-g3l.json $H5/f-g3l.json
g5s $O/g12-g3.json $O/g12-g3l.json $H5/f-g3.json $H5/f-g3l.json

echo "== [RD] REDCELL determinism: 2 fresh servers, default cache, G1 -> G3 -> G2 ($(el))"
for rep in a b; do
  if srv_start fd-$rep $M/redcell-26b $RED "" "${RA[@]}"; then
    gate g1 fd-$rep-g1 $RP; gate g3 fd-$rep-g3 2>&1 | tail -1; gate g2 fd-$rep-g2 $RP 256 2>&1 | tail -1; srv_stop
    echo "non-finite guard lines: $(grep -c 'non-finite' $SLOG); OOM lines: $(grep -c OUT_OF_MEMORY $SLOG)"
  fi
done
tmap gemma4 $O/fd-a-g1.json $O/fd-b-g1.json > /dev/null
same $O/fd-a-g1.json $O/fd-b-g1.json; same $O/fd-a-g2.json $O/fd-b-g2.json
same $O/fd-a-g2.json $R/baseline-redcell-g2.json; same $O/fd-b-g2.json $R/baseline-redcell-g2.json
same $O/fd-a-g1.json $E/m4/redcell/out/fd-a-g1.json
cmpg g1 $O/fd-a-g1.json $R/ref-red-g1.json | head -2; cmpg g2 $O/fd-a-g2.json $R/ref-red-g2.json | head -2
echo "== [RP] REDCELL --prefix-cache-n 0: G5 13500 / 2000, G3 ($(el))"
if srv_start rp $M/redcell-26b $RED "" "${RA[@]}" --prefix-cache-n 0; then
  gate g5 rp-g5 13500 128 2>&1 | tail -1; gate g5 rp-g5s 2000 128 2>&1 | tail -1; gate g3 rp-g3 2>&1 | tail -1; srv_stop
  echo "non-finite guard lines: $(grep -c 'non-finite' $SLOG)"
fi
g5s $O/rp-g5.json $O/rp-g5s.json $E/m4/redcell/out/fp-g5.json $E/m4/redcell/out/p-integ-g5.json
tmap gemma4 $O/rp-g3.json > /dev/null; cmpg g3 $O/rp-g3.json $R/ref-red-g3.json | head -1

echo "== [P] from-config, combined draft roster (port 18640): 35B -> gemma4-12b -> 35B -> redcell-26b ($(el))"
if cfg_start roster $I/roster-draft.toml; then
  echo "VRAM after 35B (default) load: $(smi)"
  timeout 600 $C $P pc $PORT qwen3.6-35b q35-before
  timeout 300 $C $P touch $PORT gemma4-12b
  timeout 600 $C $P long $PORT gemma4-12b g12-pc0 6
  timeout 600 $C $P touch $PORT qwen3.6-35b
  timeout 600 $C $P pc $PORT qwen3.6-35b q35-after
  timeout 300 $C $P touch $PORT redcell-26b
  timeout 600 $C $P long $PORT redcell-26b red-pc0 6
  srv_stop
  echo "-- server log: swaps, per-model prefix cache, OOMs"
  grep -E "titan swap: (loaded|unloaded|\[)|titan settings for the next|Prefix caching enabled|Prefix cache:|OUT_OF_MEMORY|out of memory|panicked|non-finite" $SLOG | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-240
fi
$C $P same $O/g12-pc0.json $H5/f-g3.json; $C $P same $O/g12-pc0.json $O/g12-g3.json
$C $P same $O/red-pc0.json $O/rp-g3.json
python3 -c "import json;r=json.load(open('$O/red-pc0.json'));json.dump(r[:4],open('$O/red-pc0-g3.json','w'),ensure_ascii=False)"
tmap gemma4 $O/red-pc0-g3.json > /dev/null; cmpg g3 $O/red-pc0-g3.json $R/ref-red-g3.json | head -1

echo "== [I] IQ3_S: Qwen3.5-9B IQ3_M e2e vs the iq3s branch and integ ($(el))"
for T in iq3m embiq3s; do
  F=$E/m4/iq3s/Qwen3.5-9B-IQ3_M.gguf; [ $T = embiq3s ] && F=$E/m4/iq3s/Qwen3.5-9B-IQ3_M-embIQ3S.gguf
  if srv_start iq3s-$T $(dirname $F) $(basename $F) "" --max-seq-len 4096; then
    (cd $I/iq3s && timeout 900 python3 e2e.py mistral $PORT $T); srv_stop
    (cd $I/iq3s && python3 e2e.py cmp $T $F | tail -4)
    for ref in $E/m4/iq3s/out $E/m4/integ/iq3s/out; do
      python3 -c "import json,sys;a,b=json.load(open(sys.argv[1])),json.load(open(sys.argv[2]));print('E2E',sys.argv[2],'greedy texts identical',sum(x['text']==y['text'] for x,y in zip(a['greedy'],b['greedy'])),'/',len(a['greedy']),'; tf byte-equal',sum(x==y for x,y in zip(a['tf'],b['tf'])),'/',len(a['tf']))" $I/iq3s/out/${T}_mistral.json $ref/${T}_mistral.json
      python3 -c "import json,sys;a,b=json.load(open(sys.argv[1])),json.load(open(sys.argv[2]));k=[x for x in a if not x.endswith('tok_s')];print('CMPJSON',sys.argv[2],'== ref' if all(a[x]==b[x] for x in k) else 'DIFFERS', {x:a[x] for x in k})" $I/iq3s/out/${T}_cmp.json $ref/${T}_cmp.json
    done
  fi
done
free -m | head -2
echo "== integ2-window2 end $(el)"
