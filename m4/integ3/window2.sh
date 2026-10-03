#!/bin/bash
# integ3-20261003 window 2 (bin/mistralrs-titan-integ3, no build): [G] gemma4-12b G1 + G2 (integ2 procedure) vs the g4acc
# final binary's runs (m4/g4acc/out/nb-g1/nb-g2, bin/mistralrs-w4); G3 ~4k / ~8k + G5 decode at --prefix-cache-n 0;
# [RD] REDCELL --prefix-cache-n 0, 2 fresh servers G1 -> G3 -> G2 256 (redcell2's dz procedure) vs each other and vs
# redcell2's dz-a; the same with TITAN_G4_FLASH_DECODE=0 (attribution of any G2 change to g4acc's flash decode); G5;
# [P] pcache swap test on the deploy roster (m4/integ3/roster-dry.toml = models.toml with port 18640): 35B TTFT ->
# gemma4-12b 6 x ~4k -> redcell-26b 3 x ~4k -> 35B TTFT x2; [D] deploy dry run: fresh server, every roster entry once.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/top-integ3/m4/integ3/window2.sh
source $HOME/titan-engine/top-integ3/m4/integ3/lib.sh
exec > >(tee -a $I/window2-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin integ3-window2 58
BIN=$NEWBIN; echo "binary: $BIN sha256 $(sha256sum $BIN | cut -d' ' -f1); TITAN_* in env: $(env | grep -c '^TITAN_')"
GA=$E/m4/g4acc/out; RC2=$E/top-integ3/m4/redcell2/out; P=$I/pcache.py
G12D=$M/gemma4-12b-qat; G12=gemma-4-12b-it-qat-q4_0.gguf; GP=$G4/prompts/gemma4-12b.json
RED=REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf; RP=$G4/prompts/redcell.json; RA=(--max-model-len 8192 --max-seq-len 8192 -n 0:30 --dtype bf16)
PC0=(--max-model-len 12288 --max-seq-len 12288 -n 0:48 --prefix-cache-n 0 --dtype bf16)

echo "== [K3] fmt_mmq iq4_nl gate on the committed PTX (window 1's regenerated fmt_mmq.ptx failed cuModuleLoadData 218, as in redcell2 window 1) ($(el))"
git -C $OX status --short | grep -v "^??"
(cd $OX/fmt_mmq && timeout 900 ./target/release/fmt_mmq --fmt=iq4_nl > $O/gate2-fm.log 2>&1); echo "fmt_mmq gate (committed) rc=$?"
grep -E "launches|FAIL|PASS|failing" $O/gate2-fm.log | tail -4; echo "moe lines: $(grep -ci moe $O/gate2-fm.log), 704 lines: $(grep -c 704 $O/gate2-fm.log)"
/usr/local/cuda/bin/ptxas -arch=sm_120 -o /dev/null $O/fmregen/fmt_mmq.ptx > $O/fmregen/ptxas.log 2>&1; echo "ptxas on window 1's regenerated fmt_mmq.ptx rc=$?"; head -5 $O/fmregen/ptxas.log

echo "== [G1] gemma4-12b G1 + G2 (8192, 0:48, bf16, default cache) ($(el))"
if srv_start g12s $G12D $G12 "" --max-model-len 8192 --max-seq-len 8192 -n 0:48 --dtype bf16; then
  gate g1 g12-g1 $GP; gate g2 g12-g2 $GP 256; srv_stop
fi
tmap gemma4 $O/g12-g1.json > /dev/null
same $O/g12-g1.json $GA/nb-g1.json; same $O/g12-g2.json $GA/nb-g2.json
cmpg g1 $O/g12-g1.json $R/ref-g12-g1.json | head -1; cmpg g2 $O/g12-g2.json $R/ref-g12-g2.json | head -1
echo "== [G3] gemma4-12b G3 ~4k / ~8k + G5 decode, --prefix-cache-n 0, 12288 ($(el))"
if srv_start g12p $G12D $G12 "" "${PC0[@]}"; then
  gate g3 g12-g3; gate g3 g12-g3l 28000; echo "VRAM after 8k: $(smi); OOM lines $(grep -c OUT_OF_MEMORY $SLOG)"
  gate g5 g12-d0 2000 128; gate g5 g12-d1 13500 128; gate g5 g12-d2 28000 128; srv_stop
fi
tmap gemma4 $O/g12-g3.json $O/g12-g3l.json > /dev/null
same $O/g12-g3.json $GA/p0-g3.json; same $O/g12-g3l.json $GA/p0-g3l.json
cmpg g3 $O/g12-g3.json $R/ref-g12-g3.json | head -1; cmpg g3 $O/g12-g3l.json $R/ref-g12-g3l.json | head -1
python3 $E/m4/g4acc/g3spread.py $R/ref-g12-g3.json $GA/lub256-g3.json,$GA/lub128-g3.json,$GA/lb512-g3.json -- $O/g12-g3.json $GA/p0-g3.json
python3 $E/m4/g4acc/g3spread.py $R/ref-g12-g3l.json $GA/lub256-g3l.json,$GA/lub128-g3l.json,$GA/lb512-g3l.json -- $O/g12-g3l.json $GA/p0-g3l.json
g5s $O/g12-g3.json $O/g12-g3l.json $O/g12-d0.json $O/g12-d1.json $O/g12-d2.json $GA/nb-d0.json $GA/nb-d1.json $GA/nb-d2.json

echo "== [RD] REDCELL pc 0: 2 fresh servers G1 -> G3 -> G2 256, then TITAN_G4_FLASH_DECODE=0 ($(el))"
for rep in a b e; do
  env=""; [ $rep = e ] && env="TITAN_G4_FLASH_DECODE=0"
  if srv_start rz-$rep $M/redcell-26b $RED "$env" "${RA[@]}" --prefix-cache-n 0; then
    gate g1 rz-$rep-g1 $RP; gate g3 rz-$rep-g3 2>&1 | tail -1; gate g2 rz-$rep-g2 $RP 256 2>&1 | tail -1
    curl -s -m 60 localhost:$PORT/v1/completions -H 'Content-Type: application/json' -d '{"model":"default","prompt":"The capital of France is","max_tokens":8,"temperature":0,"seed":0,"logprobs":1}' | python3 -c "import json,sys;print('PROBE rz-$rep',repr(json.load(sys.stdin)['choices'][0]['text']))"
    srv_stop
    echo "non-finite guard lines: $(grep -c 'non-finite' $SLOG); OOM lines: $(grep -c OUT_OF_MEMORY $SLOG)"
  fi
done
tmap gemma4 $O/rz-a-g1.json $O/rz-b-g1.json $O/rz-e-g1.json $O/rz-a-g3.json $O/rz-b-g3.json $O/rz-e-g3.json > /dev/null
for g in g1 g3 g2; do same $O/rz-a-$g.json $O/rz-b-$g.json; done
for g in g1 g3 g2; do same $O/rz-a-$g.json $RC2/dz-a-$g.json; done
for g in g1 g3 g2; do same $O/rz-e-$g.json $RC2/dz-a-$g.json; done
cmpg g1 $O/rz-a-g1.json $R/ref-red-g1.json | head -2; cmpg g3 $O/rz-a-g3.json $R/ref-red-g3.json | head -1
cmpg g2 $O/rz-a-g2.json $R/ref-red-g2.json | head -1; cmpg g2 $RC2/dz-a-g2.json $R/ref-red-g2.json | head -1
echo "== [RP] REDCELL G5 pc 0 ($(el))"
if srv_start rp $M/redcell-26b $RED "" "${RA[@]}" --prefix-cache-n 0; then
  gate g5 rp-g5 13500 128 2>&1 | tail -1; gate g5 rp-g5s 2000 128 2>&1 | tail -1; srv_stop
fi
g5s $O/rp-g5.json $O/rp-g5s.json $RC2/pa-g5.json

echo "== [P] pcache swap test, deploy roster on port $PORT ($(el))"
if cfg_start roster-p $I/roster-dry.toml; then
  echo "VRAM after 35B (default) load: $(smi)"
  timeout 600 $C $P pc $PORT qwen3.6-35b q35-before
  timeout 300 $C $P touch $PORT gemma4-12b
  timeout 600 $C $P long $PORT gemma4-12b g12-pc0 6
  timeout 300 $C $P touch $PORT redcell-26b
  timeout 600 $C $P long $PORT redcell-26b red-pc0 3
  timeout 600 $C $P touch $PORT qwen3.6-35b
  timeout 600 $C $P pc $PORT qwen3.6-35b q35-after
  timeout 600 $C $P pc $PORT qwen3.6-35b q35-after2
  srv_stop
  grep -E "titan swap: (loaded|unloaded)|titan settings for the next|Prefix caching enabled|OUT_OF_MEMORY|out of memory|panicked|non-finite" $SLOG | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-240
fi
$C $P same $O/g12-pc0.json $O/g12-g3.json; $C $P same $O/red-pc0.json $O/rz-a-g3.json

echo "== [D] deploy dry run: fresh from-config server, service env, every roster entry once ($(el))"
if CWD=$HOME cfg_start dry $I/roster-dry.toml; then
  echo "VRAM after the default load: $(smi)"
  timeout 2400 python3 $I/dryrun.py $PORT $I/roster-dry.toml $SLOG $O/dryrun.json
  srv_stop
  grep -E "titan swap: (loaded|unloaded)|OUT_OF_MEMORY|out of memory|panicked|ERROR" $SLOG | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-240
fi
free -m | head -2
echo "== integ3-window2 end $(el)"
