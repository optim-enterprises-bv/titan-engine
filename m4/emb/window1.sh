#!/bin/bash
# m4/emb window 1: build the emb branch (candle row-only QTensor::embedding + the iquant fixes, mistral.rs
# d0a3987ba), gate (a) embcheck, then the model gates and the regression core, in priority order:
# [B] nvcc-free build (retries on $I/rebuild.flag) -> m4/emb/mistralrs-emb; [C] plain CPU cargo check of
# candle-core (iquant cfg fix), embcheck build + run (row-only == whole-table bit for bit, every dtype, CPU + CUDA;
# CPU dequantize vs llama.cpp); [1] Spark Q4_K_M in integ's request order (g5 x3, g3, g1, g2) on the new binary,
# then g5 x3 on bin/mistralrs-titan-integ (before numbers); [2] gemma4-12b G1 + g5 (new), g5 (integ);
# [3] REDCELL G1; [R] regression core.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/emb/window1.sh
source $HOME/titan-engine/m4/emb/lib.sh
exec > >(tee -a $I/window1-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin emb-window1 58
echo "src: mistral.rs $(git -C $SRC log --oneline -1 | cut -c1-70) $(git -C $SRC diff --quiet && echo clean || echo DIRTY); candle $(git -C $E/top-emb log --oneline -1 | cut -c1-70) $(git -C $E/top-emb diff --quiet && echo clean || echo DIRTY)"
G4=$E/m4/g4s; R=$G4/out; SP=$E/m4/sparkpf; IO=$E/m4/integ/out; C=python3; CL=$I/b2client.py
gate() { # MODE OUT ARGS... (titan on $PORT)
  [ $(left) -gt 240 ] || { echo "skip $1 $2: $(left)s left"; return 1; }
  local mode=$1 out=$2; shift 2
  timeout 900 python3 $SP/gates.py $mode $PORT titan $O/$out.json "$@"
}
cmpg() { python3 $SP/cmp.py "$@" 2>&1; }
same() { python3 $E/m4/integ/same.py "$@"; }
tmap() { timeout 300 systemd-run --user --unit=emb-tmap --collect --wait --pipe -q -p MemoryMax=4G $HOME/ai/convert-env/bin/python3 $G4/tokmap.py "$@"; }
g5s() { for f in "$@"; do [ -e $f ] && python3 -c "import json,sys;r=json.load(open(sys.argv[1]));print('TPS',sys.argv[1].split('/')[-1],[(x['prompt_tokens'],round(x['prefill_tps'] or 0,1),round(x['decode_tps'] or 0,1)) for x in r])" $f; done; }

echo "== [B] mistral.rs build ($(el))"
rm -f $I/rebuild.flag
until mbuild 1500; do
  echo "build failed: waiting for $I/rebuild.flag ($(left)s left)"; date -Is > $O/build-failed.flag
  while [ ! -e $I/rebuild.flag ] && [ $(left) -gt 2100 ]; do sleep 10; done
  [ -e $I/rebuild.flag ] || { echo "no binary: ending the window"; exit 1; }
  rm -f $I/rebuild.flag $O/build-failed.flag; echo "rebuilding ($(el)): $(git -C $SRC log --oneline -1 | cut -c1-80)"
done
cp $MBIN $NEWBIN; BIN=$NEWBIN; echo "binary: $BIN sha256 $(sha256sum $BIN | cut -d' ' -f1)"

echo "== [C] candle-core plain CPU check, embcheck ($(el))"
cbuild cpuonly $I/cpuonly 600 check --offline --release
if cbuild embcheck $I/embcheck 900 build --offline --release; then
  timeout 600 systemd-run --user --unit=emb-coll --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 \
    --setenv=LD_LIBRARY_PATH=$E/lib $TGT/release/embcheck $I/ref > $O/embcheck.log 2>&1
  echo "embcheck rc=$? ($(el))"; grep -E "FAIL|not comparable|panic|error|embcheck:|timings|ms" $O/embcheck.log | head -60
fi

echo "== [1] Spark-X2.5 Q4_K_M, 16384, integ request order ($(el))"
SPF=Spark-X2.5-4B-Q4_K_M.gguf
if srv_start spark $M/spark-x2.5 $SPF "" --max-model-len 16384 --max-seq-len 16384; then
  for c in 2000 15000 33000; do gate g5 sp-g5-$c $c 128; done
  gate g3 sp-g3; gate g1 sp-g1 $SP/prompts/spark.json; gate g2 sp-g2 $SP/prompts/spark.json 256
  srv_stop
fi
tmap spark $O/sp-g1.json $O/sp-g3.json
same $O/sp-g2.json $IO/sp-g2.json; same $O/sp-g1.json $IO/sp-g1.json; same $O/sp-g3.json $IO/sp-g3.json
cmpg g1 $O/sp-g1.json $R/refc-spark-g1.json; cmpg g2 $O/sp-g2.json $R/baseline-spark-x2.5-q4km-g2.json
BIN=$OLDBIN
if srv_start spark-integ $M/spark-x2.5 $SPF "" --max-model-len 16384 --max-seq-len 16384; then
  for c in 2000 15000 33000; do gate g5 spi-g5-$c $c 128; done
  srv_stop
fi
BIN=$NEWBIN
g5s $O/sp-g5-2000.json $O/sp-g5-15000.json $O/sp-g5-33000.json $O/spi-g5-2000.json $O/spi-g5-15000.json $O/spi-g5-33000.json

echo "== [2] gemma4-12b G1 (8192, 0:48, bf16) + G5 ($(el))"
G12=gemma-4-12b-it-qat-q4_0.gguf
if srv_start g12s $M/gemma4-12b-qat $G12 "" --max-model-len 8192 --max-seq-len 8192 -n 0:48 --dtype bf16; then
  gate g1 g12-g1 $G4/prompts/gemma4-12b.json
  for c in 2000 28000; do gate g5 g12-g5-$c $c 128; done
  echo "VRAM: $(smi)"; srv_stop
fi
tmap gemma4 $O/g12-g1.json; same $O/g12-g1.json $IO/g12-g1.json; cmpg g1 $O/g12-g1.json $R/ref-g12-g1.json
BIN=$OLDBIN
if srv_start g12s-integ $M/gemma4-12b-qat $G12 "" --max-model-len 8192 --max-seq-len 8192 -n 0:48 --dtype bf16; then
  for c in 2000 28000; do gate g5 g12i-g5-$c $c 128; done
  echo "VRAM: $(smi)"; srv_stop
fi
BIN=$NEWBIN
g5s $O/g12-g5-2000.json $O/g12-g5-28000.json $O/g12i-g5-2000.json $O/g12i-g5-28000.json

echo "== [3] REDCELL G1, --prefix-cache-n 0 ($(el))"
RED=REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf
if srv_start red0 $M/redcell-26b $RED "" --max-model-len 8192 --max-seq-len 8192 -n 0:30 --dtype bf16 --prefix-cache-n 0; then
  gate g1 red-g1 $G4/prompts/redcell.json; srv_stop
fi
tmap gemma4 $O/red-g1.json; cmpg g1 $O/red-g1.json $R/ref-red-g1.json; same $O/red-g1.json $IO/red-g1.json

echo "== [R1] 35B gate pair ($(el))"
coll emb-m2 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2 && cmp40 $E/m6/out/h-off.json $E/m3/out/emb-m2.json
coll emb-mf0 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=0 && cmp40 $E/m6/out/mtpfile-off.json $E/m3/out/emb-mf0.json
echo "== [R2] Bonsai-2 27B B2-off vs o-off ($(el))"
if srv_start b-off $M/bonsai2-27b Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf "TITAN_MTP=0 TITAN_REASONING_EFFORT=medium" --max-seq-len 65536; then
  $C $CL greedy titan $PORT b-off 8 300; srv_stop
fi
$C $CL cmp o-off b-off
echo "== [R3] IQ2_M and Bonsai-27B Q1_0, new vs integ, 8 x 300 ($(el))"
if srv_start i-new $M/q35-lowbit Qwen3.6-35B-A3B-UD-IQ2_M.gguf "TITAN_MTP=2 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64" --max-seq-len 65536; then
  $C $CL greedy titan $PORT i-new 8 300; srv_stop
fi
$C $CL cmp integ-i-new i-new
if srv_start q-new $M/bonsai/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf "TITAN_ATTN_FLASH_PREFILL=1" --max-seq-len 32768; then
  $C $CL greedy titan $PORT q-new 8 300; srv_stop
fi
$C $CL cmp integ-q-new q-new
free -m | head -2
echo "== emb-window1 end $(el)"
