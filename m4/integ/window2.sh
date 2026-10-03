#!/bin/bash
# integ-20261002 window 2 (window 1 ended at the first build: a semantic merge conflict, flash_prefill::attend's
# new `win` argument vs swap-b2's GQA-padding recursion, fixed in 23296a4b8). Everything in one window, in priority order:
# [0] kernel gates re-run on the committed (embedded) PTX; [B] nvcc-free build (retries on $I/rebuild.flag) ->
# bin/mistralrs-titan-integ; [R] regression core: 35B MTP=2 vs m6/out/h-off.json, 35B MTP-off (MTP-dir file,
# TITAN_MTP=0) vs m6/out/mtpfile-off.json, Bonsai-2 B2-off / B2-mtp1 vs m4/bonsai2/out/o-off.json, IQ2_M and
# Bonsai-27B Q1_0 new vs the recorded old-binary runs (m4/b2merge/out/{i,q}-old.json); [1] Spark-X2.5 (sparkpf
# window3 order) vs the clean llama.cpp reference and spark-prefill's own outputs; [2] gemma4-12b G1 + G3 ~4k/~8k
# (--prefix-cache-n 0) vs the gemma4-spark w6 outputs; [3] REDCELL G1 + the baseline G2 procedure; [4] IQ3_S 9B e2e;
# [5] 35B prefix cache 16 vs 0 on a repeated-prefix workload; [6] old binary re-runs (IQ2_M, Q1_0) if time is left.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/integ/window2.sh
source $HOME/titan-engine/m4/integ/lib.sh
exec > >(tee -a $I/window2-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin integ-window2 58
echo "src: $(git -C $SRC log --oneline -1 | cut -c1-80) $(git -C $SRC diff --quiet && echo clean || echo DIRTY)"
G4=$E/m4/g4s; R=$G4/out; SP=$E/m4/sparkpf; SO=$SP/out; C=python3; CL=$I/b2client.py
gate() { # MODE OUT ARGS... (titan on $PORT)
  [ $(left) -gt 300 ] || { echo "skip $1 $2: $(left)s left"; return 1; }
  local mode=$1 out=$2; shift 2
  timeout 900 python3 $SP/gates.py $mode $PORT titan $O/$out.json "$@"
}
cmpg() { python3 $SP/cmp.py "$@" 2>&1; }
same() { python3 $I/same.py "$@"; }
tmap() { timeout 300 systemd-run --user --unit=integ-tmap --collect --wait --pipe -q -p MemoryMax=4G $HOME/ai/convert-env/bin/python3 $G4/tokmap.py "$@"; }
g5s() { for f in "$@"; do python3 -c "import json,sys;r=json.load(open(sys.argv[1]));print('TPS',sys.argv[1].split('/')[-1],[(x['prompt_tokens'],round(x['prefill_tps'] or 0,1),round(x['decode_tps'] or 0,1)) for x in r])" $f; done; }

echo "== [0] kernel gates re-run on the committed (embedded) PTX, window-1 host binaries ($(el))"
git -C $OX status --short | grep -v "^??"
(cd $OX/iq3_s && timeout 300 ./target/release/iq3_s > $O/gate2-iq3s.log 2>&1); echo "iq3_s gate rc=$?"; grep -E "PASS|FAIL" $O/gate2-iq3s.log | tail -2
(cd $OX/candle-quantized && OXIDE_ROOT=$OX OXIDE_PTX=$OX/mistralrs-quant-a/mistralrs_quant_a.ptx timeout 300 ./target/release/candle-quantized iqmoe > $O/gate2-iqmoe.log 2>&1); echo "iq-moe gate rc=$?"; grep -E "PASS|FAIL|failing" $O/gate2-iqmoe.log | tail -2
(cd $OX/flash-prefill && timeout 400 ./target/release/flash-prefill > $O/gate2-flash-prefill.log 2>&1); echo "flash-prefill gate rc=$?"; grep -E "^window:|^vs v1|gate:" $O/gate2-flash-prefill.log
git -C $OX status --short | grep -v "^??"

echo "== [B] mistral.rs build ($(el))"
rm -f $I/rebuild.flag
until mbuild 1500; do
  echo "build failed: waiting for $I/rebuild.flag ($(left)s left)"; date -Is > $O/build-failed.flag
  while [ ! -e $I/rebuild.flag ] && [ $(left) -gt 2100 ]; do sleep 10; done
  [ -e $I/rebuild.flag ] || { echo "no binary: ending the window"; exit 1; }
  rm -f $I/rebuild.flag $O/build-failed.flag; echo "rebuilding ($(el)): $(git -C $SRC log --oneline -1 | cut -c1-80)"
done
cp $MBIN $NEWBIN; BIN=$NEWBIN; echo "binary: $BIN sha256 $(sha256sum $BIN | cut -d' ' -f1) from $(git -C $SRC log --oneline -1 | cut -c1-60)"

echo "== [R1] 35B gate pair ($(el))"
coll integ-m2 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2 && cmp40 $E/m6/out/h-off.json $E/m3/out/integ-m2.json
coll integ-mf0 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=0 && cmp40 $E/m6/out/mtpfile-off.json $E/m3/out/integ-mf0.json

echo "== [R2] Bonsai-2 27B: B2-off, B2-mtp1 vs o-off ($(el))"
if srv_start b-off $M/bonsai2-27b Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf "TITAN_MTP=0 TITAN_REASONING_EFFORT=medium" --max-seq-len 65536; then
  $C $CL greedy titan $PORT b-off 8 300; $C $CL cmp o-off b-off; srv_stop
fi
if srv_start b-n1 $M/bonsai2-27b Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf "TITAN_MTP=1 TITAN_REASONING_EFFORT=medium" --max-seq-len 65536; then
  $C $CL greedy titan $PORT b-n1 8 300; $C $CL cmp o-off b-n1; grep -o "titan mtp stats.*" $SLOG | tail -1; srv_stop
fi
echo "== [R3] IQ2_M new, Bonsai-27B Q1_0 new, 8 x 300 ($(el))"
if srv_start i-new $M/q35-lowbit Qwen3.6-35B-A3B-UD-IQ2_M.gguf "TITAN_MTP=2 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64" --max-seq-len 65536; then
  $C $CL greedy titan $PORT i-new 8 300; srv_stop
fi
$C $CL cmp i-old-b2merge i-new
if srv_start q-new $M/bonsai/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf "TITAN_ATTN_FLASH_PREFILL=1" --max-seq-len 32768; then
  $C $CL greedy titan $PORT q-new 8 300; srv_stop
fi
$C $CL cmp q-old-b2merge q-new

echo "== [1] Spark-X2.5 Q4_K_M, 16384 ($(el))"
if srv_start spark $M/spark-x2.5 Spark-X2.5-4B-Q4_K_M.gguf "" --max-model-len 16384 --max-seq-len 16384; then
  for c in 2000 15000 33000; do gate g5 sp-g5-$c $c 128; done
  gate g3 sp-g3; gate g1 sp-g1 $SP/prompts/spark.json; gate g2 sp-g2 $SP/prompts/spark.json 256
  srv_stop
fi
tmap spark $O/sp-g1.json $O/sp-g3.json
cmpg g1 $O/sp-g1.json $R/refc-spark-g1.json; cmpg g3 $O/sp-g3.json $R/refc-spark-g3.json
cmpg g2 $O/sp-g2.json $R/baseline-spark-x2.5-q4km-g2.json
same $O/sp-g1.json $SO/t-d0-g1.json; same $O/sp-g3.json $SO/t-d0-g3.json; same $O/sp-g2.json $SO/t-d0-g2.json
g5s $O/sp-g5-2000.json $O/sp-g5-15000.json $O/sp-g5-33000.json

echo "== [2] gemma4-12b G1 (8192, 0:48, bf16, default cache) + G5 short ($(el))"
G12=gemma-4-12b-it-qat-q4_0.gguf
if srv_start g12s $M/gemma4-12b-qat $G12 "" --max-model-len 8192 --max-seq-len 8192 -n 0:48 --dtype bf16; then
  gate g1 g12-g1 $G4/prompts/gemma4-12b.json
  same $O/g12-g1.json $R/t-g12s6-g1.json
  gate g5 g12-g5-2000 2000 128
  srv_stop
fi
tmap gemma4 $O/g12-g1.json; cmpg g1 $O/g12-g1.json $R/ref-g12-g1.json
echo "== [2b] gemma4-12b G3 ~4k / ~8k, 12288, --prefix-cache-n 0 ($(el))"
if srv_start g12p $M/gemma4-12b-qat $G12 "" --max-model-len 12288 --max-seq-len 12288 -n 0:48 --dtype bf16 --prefix-cache-n 0; then
  gate g3 g12-g3; echo "VRAM after G3 4k: $(smi)"
  gate g3 g12-g3l 28000; echo "VRAM after G3 8k: $(smi)"
  echo "OOM lines: $(grep -c OUT_OF_MEMORY $SLOG)"
  srv_stop
fi
tmap gemma4 $O/g12-g3.json $O/g12-g3l.json
cmpg g3 $O/g12-g3.json $R/ref-g12-g3.json; cmpg g3 $O/g12-g3l.json $R/ref-g12-g3l.json
same $O/g12-g3.json $R/t-g12p-g3.json; same $O/g12-g3l.json $R/t-g12p-g3l.json
g5s $O/g12-g5-2000.json

echo "== [3] REDCELL G1, --prefix-cache-n 0 (w6 config) ($(el))"
RED=REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf; RP=$G4/prompts/redcell.json
if srv_start red0 $M/redcell-26b $RED "" --max-model-len 8192 --max-seq-len 8192 -n 0:30 --dtype bf16 --prefix-cache-n 0; then
  gate g1 red-g1 $RP; srv_stop
fi
tmap gemma4 $O/red-g1.json; cmpg g1 $O/red-g1.json $R/ref-red-g1.json; same $O/red-g1.json $R/t-red6-g1.json
echo "== [3b] REDCELL baseline G2 procedure: default cache, G1 -> G3 -> G2 ($(el))"
redproc() { # TAG
  if srv_start $1 $M/redcell-26b $RED "" --max-model-len 8192 --max-seq-len 8192 -n 0:30 --dtype bf16; then
    gate g1 $1-g1 $RP; gate g3 $1-g3; gate g2 $1-g2 $RP 256; srv_stop
  fi
  [ -e $O/$1-g2.json ] && same $O/$1-g2.json $R/baseline-redcell-g2.json && cmpg g2 $O/$1-g2.json $R/baseline-redcell-g2.json
}
redproc redb
if ! python3 -c "import json,sys;a,b=json.load(open(sys.argv[1])),json.load(open(sys.argv[2]));sys.exit(0 if [x['text'] for x in a]==[x['text'] for x in b] else 1)" $O/redb-g2.json $R/baseline-redcell-g2.json 2>/dev/null; then
  echo "-- REDCELL G2 differs from the baseline: same procedure on the gemma4-spark branch binary (bin/mistralrs-g4s-w6) ($(el))"
  BIN=$E/bin/mistralrs-g4s-w6; redproc redw6; BIN=$NEWBIN
  [ -e $O/redw6-g2.json ] && same $O/redb-g2.json $O/redw6-g2.json
fi

echo "== [4] IQ3_S: Qwen3.5-9B IQ3_M e2e, mistral side, vs the iq3s branch ($(el))"
for T in iq3m embiq3s; do
  F=$E/m4/iq3s/Qwen3.5-9B-IQ3_M.gguf; [ $T = embiq3s ] && F=$E/m4/iq3s/Qwen3.5-9B-IQ3_M-embIQ3S.gguf
  if srv_start iq3s-$T $(dirname $F) $(basename $F) "" --max-seq-len 4096; then
    (cd $I/iq3s && timeout 900 python3 e2e.py mistral $PORT $T); srv_stop
    (cd $I/iq3s && python3 e2e.py cmp $T | tail -2)
    same $I/iq3s/out/${T}_mistral.json $E/m4/iq3s/out/${T}_mistral.json
    python3 -c "import json,sys;a,b=json.load(open(sys.argv[1])),json.load(open(sys.argv[2]));print('CMPJSON',sys.argv[1].split('/')[-1],'== branch' if a==b else 'DIFFERS from branch', {k:a[k] for k in a if k not in ('llama_tok_s','mistral_tok_s')})" $I/iq3s/out/${T}_cmp.json $E/m4/iq3s/out/${T}_cmp.json
  fi
done

echo "== [5] 35B prefix cache: service env, 65536, repeated-prefix workload, default (16) vs --prefix-cache-n 0 ($(el))"
SVC="TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt TITAN_TIERED_RESERVE_MIB=1536 TITAN_MTP=2 TITAN_PFS=1 TITAN_PFS_HEADROOM_MIB=768 TITAN_DOORBELL=1 TITAN_CPU_ONEPASS=1 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64"
for pcn in 16 0; do
  if [ $(left) -gt 540 ] && srv_start q35-pc$pcn $M/Qwen3.6-35B-A3B-MTP Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf "$SVC" --max-seq-len 65536 --prefix-cache-n $pcn; then
    timeout 900 python3 $I/pc.py $PORT q35-pc$pcn; srv_stop
  fi
done

echo "== [6] old binary re-runs, IQ2_M and Q1_0 ($(el))"
BIN=$OLDBIN
if [ $(left) -gt 420 ] && srv_start i-old $M/q35-lowbit Qwen3.6-35B-A3B-UD-IQ2_M.gguf "TITAN_MTP=2 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64" --max-seq-len 65536; then
  $C $CL greedy titan $PORT i-old 8 300; srv_stop; $C $CL cmp i-old i-new
fi
if [ $(left) -gt 420 ] && srv_start q-old $M/bonsai/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf "TITAN_ATTN_FLASH_PREFILL=1" --max-seq-len 32768; then
  $C $CL greedy titan $PORT q-old 8 300; srv_stop; $C $CL cmp q-old q-new
fi
BIN=$NEWBIN
free -m | head -2
echo "== integ-window2 end $(el)"
