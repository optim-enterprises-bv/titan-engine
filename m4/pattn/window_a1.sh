#!/bin/bash
# pattn phase A window 1: [B] build (Spark paged mask fix, titan flash-prefill in the paged prefill, planner/admission,
# TITAN_PATTN_TRACE / TITAN_PATTN_DUMP); [D] root-cause dumps of one ~8k Spark prompt (g3l #1): paged OFF, paged ON with
# the old full mask (TITAN_SPARK_PAGED_FULLMASK=1, generic path), ON with the fixed mask (generic), ON fixed + flash;
# [S] Spark paged ON G1 / G3 4k / G3 8k / G5 + OFF G3 4k; [L] llama.cpp GPU reference + self-spread for Spark (4k, 8k)
# and qwen3-14b (~16k); [Q] qwen3-14b paged ON (flash) G1 / 8k / 16k / G5 + OFF 16k.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/top-pattn/m4/pattn/window_a1.sh
source $HOME/titan-engine/top-pattn/m4/pattn/lib.sh
restic_wait
exec > >(tee -a $I/window_a1-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin pattn-a1 58
echo "src: mistral.rs $(git -C $SRC log --oneline -1 | cut -c1-60) $(git -C $SRC diff --quiet && echo clean || echo DIRTY)"
build_bin2 1500
SK=Spark-X2.5-4B-Q4_K_M.gguf; SP=$G/prompts/spark.json; SA="--max-model-len 16384 --max-seq-len 16384"
D=$SC/dump; rm -rf $D; mkdir -p $D
dumprun() { # TAG PA ENV
  if PA=$2 srv_start d-$1 $M/spark-x2.5 $SK "TITAN_PATTN_DUMP=$D/$1 TITAN_PATTN_DUMP_LAYERS=0,3,4 TITAN_PATTN_TRACE=1 $3" $SA $( [ $2 = on ] && echo --pa-context-len 16384 ); then
    python3 $I/pdump.py $PORT 28000 1 $O/d-$1.json; srv_stop
    grep "^pattn:" $SLOG | sort | uniq -c | sort -rn | head -8 | cut -c1-260
  fi
}
echo "== [D] root-cause dumps, Spark g3l prompt 1 ($(el))"
dumprun off off ""
dumprun oldmask on "TITAN_PAGED_FLASH_PREFILL=0 TITAN_SPARK_PAGED_FULLMASK=1"
dumprun fixmask on "TITAN_PAGED_FLASH_PREFILL=0"
dumprun flash on ""
N=$(python3 -c "import json;print(json.load(open('$O/d-off.json'))['prompt_tokens'])")
for r in oldmask fixmask flash; do for l in 0 3 4; do python3 $I/cmpdump.py $N $D/$r $D/off $l; done; done
for r in off oldmask fixmask flash; do python3 -c "import json;r=json.load(open('$O/d-$r.json'));print('$r', r['prompt_tokens'], r['top'][0][:4])"; done
du -sh $D

echo "== [S] Spark paged ON (flash) G1 / G3 4k / G3 8k / G5; OFF G3 4k ($(el))"
if PA=on srv_start s4-on $M/spark-x2.5 $SK "" $SA --pa-context-len 16384; then
  echo "VRAM after load: $(smi)"
  gate g1 s4-on-g1 $SP | tail -1; gate g3 s4-on-g3 | tail -1; gate g3 s4-on-g3l 28000 | tail -1; echo "VRAM after 8k: $(smi)"
  gate g5 s4-on-g5 8000 128 | tail -1; srv_stop; fi
if srv_start s4-off $M/spark-x2.5 $SK "" $SA; then gate g3 s4-off-g3 | tail -1; srv_stop; fi
for f in s4-on-g1 s4-on-g3 s4-on-g3l s4-off-g3; do cp $O/$f.json $O/$f.ids.json 2>/dev/null; done
tmaps $O/s4-on-g1.json $O/s4-on-g3.json $O/s4-on-g3l.json $O/s4-off-g3.json
cmpg g1 $O/s4-on-g1.json $G/out/refc-spark-g1.json; cmpg g1 $O/s4-on-g1.json $O/s-off-g1.json
cmpg g3 $O/s4-on-g3l.json $O/ref-spark-cpu-g3l.json; cmpg g3 $O/s4-on-g3l.json $O/s-off-g3l.json
tps $O/s4-on-g5.json $O/s-off-g5.json $O/s4-on-g3l.json $O/s-off-g3l.json

echo "== [L] llama.cpp GPU reference + spread: Spark 4k / 8k, qwen3-14b 16k ($(el))"
bash $I/lspread.sh $M/spark-x2.5/$SK $O/spread-spark 12288 13500 28000
bash $I/lspread.sh $M/qwen3-14b/Qwen3-14B-vanilla-Q5_K_M.gguf $O/spread-q14 18432 56000
SS=$O/spread-spark; for c in 13500 28000; do v=$SS/lub256-c$c.json,$SS/lub128-c$c.json,$SS/lb512-c$c.json
  case $c in 13500) t="$O/s4-on-g3.json $O/s4-off-g3.json";; *) t="$O/s4-on-g3l.json $O/s-off-g3l.json";; esac
  python3 $E/m4/g4acc/g3spread.py $SS/base-c$c.json $v -- $t; done

echo "== [Q] qwen3-14b paged ON (flash; MISTRALRS_CUDA_GRAPHS=0) / OFF ($(el))"
QP=$E/m4/devmap/prompts/qwen3-14b.json; QA="--max-seq-len 18432 --prefix-cache-n 0"
if PA=on srv_start q4-on $M/qwen3-14b Qwen3-14B-vanilla-Q5_K_M.gguf "TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64 MISTRALRS_CUDA_GRAPHS=0 TITAN_PATTN_TRACE=1" $QA; then
  echo "VRAM after load: $(smi)"; palines $SLOG 4
  gate g1 q4-on-g1 $QP | tail -1; gate g3 q4-on-g3 | tail -1; gate g3 q4-on-g3l 28000 | tail -1; gate g3 q4-on-g3x 56000 | tail -1
  echo "VRAM after 16k: $(smi)"; gate g5 q4-on-g5 8000 128 | tail -1; srv_stop
  grep "^pattn:" $SLOG | sort | uniq -c | sort -rn | head -6 | cut -c1-240
fi
if srv_start q4-off $M/qwen3-14b Qwen3-14B-vanilla-Q5_K_M.gguf "TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64" $QA; then
  gate g3 q4-off-g3x 56000 | tail -1; echo "VRAM after 16k: $(smi)"; srv_stop; fi
for f in q4-on-g1 q4-on-g3 q4-on-g3l q4-on-g3x q4-off-g3x; do cp $O/$f.json $O/$f.ids.json 2>/dev/null; done
tmapq $O/q4-on-g1.json $O/q4-on-g3.json $O/q4-on-g3l.json $O/q4-on-g3x.json $O/q4-off-g3x.json
same $O/q4-on-g1.ids.json $O/q-off-g1.ids.json; cmpg g1 $O/q4-on-g1.json $E/m4/devmap/out/lref-g1.json
DS=$E/m4/devmap/out; python3 $E/m4/g4acc/g3spread.py $DS/lref-g3.json $DS/spread/lub256-g3.json,$DS/spread/lub128-g3.json,$DS/spread/lb512-g3.json -- $O/q4-on-g3.json
python3 $E/m4/g4acc/g3spread.py $DS/lref-g3l.json $DS/spread/lub256-g3l.json,$DS/spread/lub128-g3l.json,$DS/spread/lb512-g3l.json -- $O/q4-on-g3l.json $O/q-off-g3l.json
QS=$O/spread-q14; python3 $E/m4/g4acc/g3spread.py $QS/base-c56000.json $QS/lub256-c56000.json,$QS/lub128-c56000.json,$QS/lb512-c56000.json -- $O/q4-on-g3x.json $O/q4-off-g3x.json
tps $O/q4-on-g5.json $O/q-off-g5.json $O/q4-on-g3l.json $O/q-off-g3l.json $O/q4-on-g3x.json $O/q4-off-g3x.json
free -m | head -2
echo "== pattn-a1 end $(el)"
