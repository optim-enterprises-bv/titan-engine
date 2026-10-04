#!/bin/bash
# pattn phase A window 3: [B] build (device map: activations for one paged prompt chunk); [Q] qwen3-14b paged ON with the
# default pool: VRAM / pool, 8k and ~16k G3, G5; with --max-num-batched-tokens 16384 (one chunk per prompt, like paged off):
# 8k / ~16k G3; [S] Spark paged ON --max-num-batched-tokens 16384 --prefix-cache-n 0: G1 / G3 4k / 8k vs paged off
# (identity) and llama.cpp's spread, G5; default chunks with --prefix-cache-n 0 too.
source $HOME/titan-engine/top-pattn/m4/pattn/lib.sh
restic_wait
exec > >(tee -a $I/window_a3-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin pattn-a3 58
echo "src: mistral.rs $(git -C $SRC log --oneline -1 | cut -c1-60) $(git -C $SRC diff --quiet && echo clean || echo DIRTY)"
build_bin2 1500
QM=$M/qwen3-14b; QF=Qwen3-14B-vanilla-Q5_K_M.gguf; QE="TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64 MISTRALRS_CUDA_GRAPHS=0"; QP=$E/m4/devmap/prompts/qwen3-14b.json
q() { # TAG ARGS...
  local t=$1; shift
  if PA=on srv_start $t $QM $QF "$QE" --max-seq-len 16384 --prefix-cache-n 0 "$@"; then
    echo "VRAM after load: $(smi)"; palines $SLOG 6
    gate g3 $t-g3l 28000 | tail -1; gate g3 $t-g3x 48000 | tail -1; echo "VRAM after ~16k: $(smi)"; gate g5 $t-g5 8000 128 | tail -1; srv_stop
    sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -E " WARN | ERROR " | head -3 | cut -c1-200
    for f in $t-g3l $t-g3x; do cp $O/$f.json $O/$f.ids.json 2>/dev/null; tmapq $O/$f.json; done
  fi
}
echo "== [Q] qwen3-14b paged ON, default pool ($(el))"
q q6-on
q q6-big --max-num-batched-tokens 16384
same $O/q6-big-g3l.ids.json $O/q-off-g3l.ids.json; same $O/q6-big-g3x.ids.json $O/q5-off-g3x.ids.json
DS=$E/m4/devmap/out; QS=$O/spread-q14
python3 $E/m4/g4acc/g3spread.py $DS/lref-g3l.json $DS/spread/lub256-g3l.json,$DS/spread/lub128-g3l.json,$DS/spread/lb512-g3l.json -- $O/q6-on-g3l.json $O/q6-big-g3l.json $O/q-off-g3l.json
python3 $E/m4/g4acc/g3spread.py $QS/base-c48000.json $QS/lub256-c48000.json,$QS/lub128-c48000.json,$QS/lb512-c48000.json -- $O/q6-on-g3x.json $O/q6-big-g3x.json $O/q5-off-g3x.json
tps $O/q6-on-g5.json $O/q6-big-g5.json $O/q-off-g5.json $O/q6-on-g3l.json $O/q6-big-g3l.json $O/q-off-g3l.json $O/q6-on-g3x.json $O/q6-big-g3x.json $O/q5-off-g3x.json

echo "== [S] Spark paged ON, prefix cache off ($(el))"
SK=Spark-X2.5-4B-Q4_K_M.gguf; SP=$G/prompts/spark.json; SA="--max-model-len 16384 --max-seq-len 16384 --prefix-cache-n 0"
s() { # TAG PA ARGS...
  local t=$1 pa=$2; shift 2
  if PA=$pa srv_start $t $M/spark-x2.5 $SK "" $SA "$@"; then
    echo "VRAM after load: $(smi)"; palines $SLOG 4
    gate g1 $t-g1 $SP | tail -1; gate g3 $t-g3 | tail -1; gate g3 $t-g3l 28000 | tail -1; echo "VRAM after 8k: $(smi)"
    gate g5 $t-g5 8000 128 | tail -1; srv_stop
    for f in $t-g1 $t-g3 $t-g3l; do cp $O/$f.json $O/$f.ids.json; done; tmaps $O/$t-g1.json $O/$t-g3.json $O/$t-g3l.json
  fi
}
s s6-big on --max-num-batched-tokens 16384
s s6-on on
s s6-off off
for t in s6-big s6-on; do for g in g1 g3 g3l; do same $O/$t-$g.ids.json $O/s6-off-$g.ids.json; done; done
SS=$O/spread-spark
python3 $E/m4/g4acc/g3spread.py $SS/base-c13500.json $SS/lub256-c13500.json,$SS/lub128-c13500.json,$SS/lb512-c13500.json -- $O/s6-big-g3.json $O/s6-on-g3.json $O/s6-off-g3.json
python3 $E/m4/g4acc/g3spread.py $SS/base-c28000.json $SS/lub256-c28000.json,$SS/lub128-c28000.json,$SS/lb512-c28000.json -- $O/s6-big-g3l.json $O/s6-on-g3l.json $O/s6-off-g3l.json
for t in s6-big s6-on s6-off; do cmpg g1 $O/$t-g1.json $G/out/refc-spark-g1.json | head -1; done
tps $O/s6-big-g5.json $O/s6-on-g5.json $O/s6-off-g5.json $O/s6-big-g3l.json $O/s6-on-g3l.json $O/s6-off-g3l.json
free -m | head -2
echo "== pattn-a3 end $(el)"
