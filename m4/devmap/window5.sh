#!/bin/bash
# devmap2 window 2 (of 3): [B] build (MoE CPU f32 rows, qwen3 flash-path estimate) while [E] the deployed binary's
# qwen3 G3 / G3l (eager path, pinned 40, prefix cache off) for the spread comparison; [U] unit tests (background);
# [P] kv_cache:857 provoked on the v3 binary (window-3 mq40 config); [G1] the five models unpinned, near-max prompt,
# qwen3 short decode; [C] REDCELL with 3 CPU layers, G1 vs all-GPU; [G2] qwen3-14b and DeepSeek-R1-Qwen3-8B Q8_0
# @65536 (8k then ~30k prompt); [R] regression core on this binary.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/top-devmap/m4/devmap/window5.sh
source $HOME/titan-engine/top-devmap/m4/devmap/lib.sh
exec > >(tee -a $I/window5-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin devmap2-window5 60
echo "src: mistral.rs $(git -C $SRC log --oneline -1 | cut -c1-70) $(git -C $SRC diff --quiet && echo clean || echo DIRTY); candle $(git -C $E/top-devmap log --oneline -1 -- candle | cut -c1-50); oxide $(git -C $OX log --oneline -1 | cut -c1-50)"
QM="$M/qwen3-14b Qwen3-14B-vanilla-Q5_K_M.gguf"; QE="TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64"

echo "== [B] build, background ($(el))"
( mbuild 1500 > $O/mbuild.status 2>&1 ) &
BP=$!
echo "== [E] deployed binary qwen3 G3 / G3l (eager), 40 layers, prefix cache off, MemoryMax 14G ($(el))"
BIN=$OLDBIN
if MEM=14G srv_start eq $QM "$QE" --max-seq-len 16384 --prefix-cache-n 0 -n 0:40; then
  tgate g3 $O/eq-g3.json | tail -1; tgate g3 $O/eq-g3l.json 28000 | tail -1; srv_stop
fi
wait $BP; cat $O/mbuild.status
[ -x $MBIN ] && grep -q Finished $O/build-$WN.log || { echo "no binary"; exit 1; }
cp $MBIN $NEWBIN; BIN=$NEWBIN; echo "binary: $BIN sha256 $(sha256sum $BIN | cut -d' ' -f1)"
( mtest unit 1200 -p mistralrs-core --lib devmap_ > $O/mtest.status 2>&1 ) &
TP=$!
tmapq $O/eq-g3.json $O/eq-g3l.json
for g in g3 g3l; do python3 $E/m4/g4acc/g3spread.py $O/lref-$g.json $O/spread/lub256-$g.json,$O/spread/lub128-$g.json,$O/spread/lb512-$g.json -- $O/eq-$g.json | tail -1; done

echo "== [P] kv_cache:857 on the v3 binary: 40 layers, 16384, prefix cache off, 8k then 16k ($(el))"
BIN=$V3BIN
if MEM=14G srv_start p857 $QM "$QE" --max-seq-len 16384 --prefix-cache-n 0 -n 0:40; then
  probe $O/p857.json 8000,16000,1000 16
  sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -E "panicked|is dead|rebooting|titan: CUDA out of memory" | cut -c1-200 | head -4; srv_stop
fi
BIN=$NEWBIN

echo "== [G1] unpinned, near-max prompt ($(el))"
if MEM=14G srv_start g1-q16 $QM "$QE TITAN_DEVMAP_LOG=1" --max-seq-len 16384; then
  maplines $SLOG | grep -q ": cpu" && echo "G1 q16: FAIL mapping offloads" || echo "G1 q16: mapping all GPU"
  probe $O/g1c-q16s.json 50 256; probe $O/g1c-q16.json 16200 64; probe $O/g1c-q16t.json 50 256; memlines $SLOG | grep after | tail -2; srv_stop
fi
declare -A A=([gf]="--dtype f32 --max-model-len 11264 --max-seq-len 11264 --prefix-cache-n 0" [gb]="--dtype bf16 --max-model-len 12288 --max-seq-len 12288 --prefix-cache-n 0"
  [rc]="--dtype bf16 --max-model-len 8192 --max-seq-len 8192 --prefix-cache-n 0" [sp]="--max-model-len 16384 --max-seq-len 16384")
declare -A F=([gf]="gemma4-12b-qat gemma-4-12b-it-qat-q4_0.gguf" [gb]="gemma4-12b-qat gemma-4-12b-it-qat-q4_0.gguf" [rc]="redcell-26b REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf" [sp]="spark-x2.5 Spark-X2.5-4B-Q4_K_M.gguf")
declare -A N=([gf]=11200 [gb]=12150 [rc]=8150 [sp]=16200)
for k in gf gb rc sp; do set -- ${F[$k]}
  if MEM=14G srv_start g1-$k $M/$1 $2 "TITAN_DEVMAP_LOG=1" ${A[$k]}; then
    maplines $SLOG | grep -q ": cpu" && echo "G1 $k: FAIL mapping offloads" || echo "G1 $k: mapping all GPU"
    probe $O/g1c-$k.json 1000,${N[$k]} 64; srv_stop
  fi
done

echo "== [C] REDCELL, 3 CPU layers vs all GPU, G1 ($(el))"
RA="--dtype bf16 --max-model-len 8192 --max-seq-len 8192 --prefix-cache-n 0"
for v in "cr-gpu -n 0:30" "cr-cpu -n 0:27"; do set -- $v; n=$1; shift
  if MEM=14G srv_start $n $M/redcell-26b REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf "" $RA "$@"; then tgate g1 $O/$n-g1.json $G/prompts/redcell.json | tail -1; srv_stop; fi
done
cmpg g1 $O/cr-cpu-g1.json $O/cr-gpu-g1.json

echo "== [U] unit tests ($(el))"
wait $TP; cat $O/mtest.status

echo "== [R] regression core ($(el))"
coll devmap2-m2 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2 && cmp40 $E/m6/out/h-off.json $E/m3/out/devmap2-m2.json
coll devmap2-mf0 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=0 && cmp40 $E/m6/out/mtpfile-off.json $E/m3/out/devmap2-mf0.json
if srv_start b-off2 $M/bonsai2-27b Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf "TITAN_MTP=0 TITAN_REASONING_EFFORT=medium" --max-seq-len 65536; then
  $C $I/b2client.py greedy titan $PORT b-off2 8 300; srv_stop
fi
$C $I/b2client.py cmp integ3-o-off b-off2
if srv_start i-new2 $M/q35-lowbit Qwen3.6-35B-A3B-UD-IQ2_M.gguf "TITAN_MTP=2 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64" --max-seq-len 65536; then
  $C $I/b2client.py greedy titan $PORT i-new2 8 300; srv_stop
fi
$C $I/b2client.py cmp integ3-i-new i-new2
if srv_start q-new2 $M/bonsai/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf "TITAN_ATTN_FLASH_PREFILL=1" --max-seq-len 32768; then
  $C $I/b2client.py greedy titan $PORT q-new2 8 300; srv_stop
fi
$C $I/b2client.py cmp integ3-q-new q-new2

echo "== [G2] @65536: offload, 8k then ~30k prompt ($(el), $(left)s left)"
for k in q64 d64; do
  if [ $k = q64 ]; then set -- $QM; else set -- $I/models/ds8b DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf; fi
  [ $(left) -lt 420 ] && { echo "skip $k: $(left)s left"; continue; }
  if MEM=14G srv_start g2-$k $1 $2 "$QE TITAN_DEVMAP_LOG=1" --max-seq-len 65536; then
    maplines $SLOG | grep -q ": cpu" && echo "G2 $k: mapping offloads" || echo "G2 $k: mapping all GPU"
    PT=$(( $(left) / 2 - 60 )) probe $O/g2c-$k.json 50,8000,30000 32; memlines $SLOG | grep after | tail -3; srv_stop
  fi
done
free -m | head -2
echo "== devmap2-window5 end $(el)"
