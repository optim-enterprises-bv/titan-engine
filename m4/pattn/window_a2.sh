#!/bin/bash
# pattn phase A window 2: [B] build a03026bf9 (+ model_config admission planning, gemma2 fix); [S] Spark paged ON G1/G3
# 4k/8k/G5, and ON with --max-num-batched-tokens 16384 (whole prompt in one chunk, like paged off); [Q] qwen3-14b paged ON
# 8k (admission) and ~16k (48000 chars, --pa-context-len 16384), G5, OFF at 48000; [L] llama.cpp qwen3 reference + spread at
# 48000 chars; [G] gemma4-12b and REDCELL paged ON (G1, G3 4k, 8k); [C] clean concurrency, Spark --max-seqs 4
# --prefix-cache-n 0, ON and OFF, first-divergence logprob gaps.
source $HOME/titan-engine/top-pattn/m4/pattn/lib.sh
restic_wait
exec > >(tee -a $I/window_a2-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin pattn-a2 58
echo "src: mistral.rs $(git -C $SRC log --oneline -1 | cut -c1-60) $(git -C $SRC diff --quiet && echo clean || echo DIRTY)"
build_bin2 1500
SK=Spark-X2.5-4B-Q4_K_M.gguf; SP=$G/prompts/spark.json; SA="--max-model-len 16384 --max-seq-len 16384 --pa-context-len 16384"
spark() { # TAG ARGS...
  local t=$1; shift
  if PA=on srv_start $t $M/spark-x2.5 $SK "" $SA "$@"; then
    echo "VRAM after load: $(smi)"; gate g1 $t-g1 $SP | tail -1; gate g3 $t-g3 | tail -1; gate g3 $t-g3l 28000 | tail -1
    echo "VRAM after 8k: $(smi)"; gate g5 $t-g5 8000 128 | tail -1; srv_stop
    for f in $t-g1 $t-g3 $t-g3l; do cp $O/$f.json $O/$f.ids.json; done; tmaps $O/$t-g1.json $O/$t-g3.json $O/$t-g3l.json
  fi
}
echo "== [S] Spark paged ON, default chunks and one-chunk prompts ($(el))"
spark s5-on
spark s5-big --max-num-batched-tokens 16384
for t in s5-on s5-big; do same $O/$t-g3.ids.json $O/s4-off-g3.ids.json; same $O/$t-g3l.ids.json $O/s-off-g3l.ids.json; done
same $O/s5-on-g1.ids.json $O/s4-on-g1.ids.json
SS=$O/spread-spark
python3 $E/m4/g4acc/g3spread.py $SS/base-c13500.json $SS/lub256-c13500.json,$SS/lub128-c13500.json,$SS/lb512-c13500.json -- $O/s5-on-g3.json $O/s5-big-g3.json $O/s4-off-g3.json
python3 $E/m4/g4acc/g3spread.py $SS/base-c28000.json $SS/lub256-c28000.json,$SS/lub128-c28000.json,$SS/lb512-c28000.json -- $O/s5-on-g3l.json $O/s5-big-g3l.json $O/s-off-g3l.json
cmpg g1 $O/s5-on-g1.json $G/out/refc-spark-g1.json
tps $O/s5-on-g5.json $O/s5-big-g5.json $O/s-off-g5.json $O/s5-on-g3l.json $O/s5-big-g3l.json $O/s-off-g3l.json

echo "== [Q] qwen3-14b paged ON ($(el))"
QM=$M/qwen3-14b; QF=Qwen3-14B-vanilla-Q5_K_M.gguf; QE="TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64"; QP=$E/m4/devmap/prompts/qwen3-14b.json
if PA=on srv_start q5-on $QM $QF "$QE MISTRALRS_CUDA_GRAPHS=0" --max-seq-len 16384 --prefix-cache-n 0; then
  echo "VRAM after load: $(smi)"; palines $SLOG 4
  gate g1 q5-on-g1 $QP | tail -1; gate g3 q5-on-g3 | tail -1; gate g3 q5-on-g3l 28000 | tail -1; echo "VRAM after 8k: $(smi)"
  gate g5 q5-on-g5 8000 128 | tail -1; srv_stop
  sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -E "WARN|503" | head -3 | cut -c1-200
fi
if PA=on srv_start q5-on16 $QM $QF "$QE MISTRALRS_CUDA_GRAPHS=0" --max-seq-len 16384 --prefix-cache-n 0 --pa-context-len 16384; then
  echo "VRAM after load: $(smi)"; palines $SLOG 4
  gate g3 q5-on16-g3x 48000 | tail -1; echo "VRAM after ~16k: $(smi)"; srv_stop
  sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -E "WARN|503" | head -3 | cut -c1-200
fi
if srv_start q5-off $QM $QF "$QE" --max-seq-len 16384 --prefix-cache-n 0; then
  gate g3 q5-off-g3x 48000 | tail -1; echo "VRAM after ~16k: $(smi)"; srv_stop; fi
for f in q5-on-g1 q5-on-g3 q5-on-g3l q5-on16-g3x q5-off-g3x; do cp $O/$f.json $O/$f.ids.json 2>/dev/null; done
tmapq $O/q5-on-g1.json; tmapq $O/q5-on-g3.json; tmapq $O/q5-on-g3l.json; tmapq $O/q5-on16-g3x.json; tmapq $O/q5-off-g3x.json
echo "== [L] llama.cpp qwen3-14b reference + spread at 48000 chars ($(el))"
bash $I/lspread.sh $QM/$QF $O/spread-q14 16384 48000
DS=$E/m4/devmap/out
python3 $E/m4/g4acc/g3spread.py $DS/lref-g3.json $DS/spread/lub256-g3.json,$DS/spread/lub128-g3.json,$DS/spread/lb512-g3.json -- $O/q5-on-g3.json
python3 $E/m4/g4acc/g3spread.py $DS/lref-g3l.json $DS/spread/lub256-g3l.json,$DS/spread/lub128-g3l.json,$DS/spread/lb512-g3l.json -- $O/q5-on-g3l.json $O/q-off-g3l.json
QS=$O/spread-q14; python3 $E/m4/g4acc/g3spread.py $QS/base-c48000.json $QS/lub256-c48000.json,$QS/lub128-c48000.json,$QS/lb512-c48000.json -- $O/q5-on16-g3x.json $O/q5-off-g3x.json
cmpg g1 $O/q5-on-g1.json $DS/lref-g1.json; same $O/q5-on-g1.ids.json $O/q-off-g1.ids.json
tps $O/q5-on-g5.json $O/q-off-g5.json $O/q5-on-g3l.json $O/q-off-g3l.json $O/q5-on16-g3x.json $O/q5-off-g3x.json

echo "== [G] gemma4-12b and REDCELL paged ON ($(el))"
GA="--dtype f32 --max-model-len 11264 --max-seq-len 11264 --prefix-cache-n 0 -n 0:48"
RA="--dtype bf16 --max-model-len 8192 --max-seq-len 8192 --prefix-cache-n 0 -n 0:30"
if PA=on srv_start g5-on $M/gemma4-12b-qat gemma-4-12b-it-qat-q4_0.gguf "TITAN_PATTN_TRACE=1" $GA --pa-context-len 11264; then
  echo "VRAM after load: $(smi)"; palines $SLOG 4
  gate g1 g5-on-g1 $G/prompts/gemma4-12b.json | tail -1; gate g3 g5-on-g3 | tail -1; gate g3 g5-on-g3l 28000 | tail -1; gate g5 g5-on-g5 8000 128 | tail -1; srv_stop
  grep "^pattn:" $SLOG | sort | uniq -c | sort -rn | head -5 | cut -c1-220
fi
if PA=on srv_start r5-on $M/redcell-26b REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf "TITAN_PATTN_TRACE=1" $RA --pa-context-len 8192; then
  echo "VRAM after load: $(smi)"; palines $SLOG 4
  gate g1 r5-on-g1 $G/prompts/redcell.json | tail -1; gate g3 r5-on-g3 | tail -1; gate g5 r5-on-g5 8000 128 | tail -1; srv_stop
  grep "^pattn:" $SLOG | sort | uniq -c | sort -rn | head -5 | cut -c1-220
fi
for m in g5 r5; do [ -e $O/$m-on-g1.json ] && same $O/$m-on-g1.json $O/dep-$( [ $m = g5 ] && echo ig || echo ir )-g1.json; done

echo "== [C] clean concurrency: Spark --max-seqs 4 --prefix-cache-n 0 ($(el))"
for pa in on off; do
  extra=""; [ $pa = on ] && extra="--pa-context-len 16384"
  if PA=$pa srv_start c5-$pa $M/spark-x2.5 $SK "" --max-model-len 16384 --max-seq-len 16384 --max-seqs 4 --prefix-cache-n 0 $extra; then
    sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -E "Layers [0-9]|disabling PagedAttention|max_num_seqs" | sed 's/^.*INFO [a-z_:0-9]*: //' | head -3
    conc $SP $O/c5-$pa-conc.json 4 128 0; conc $SP $O/c5-$pa-conc2.json 4 256 8; srv_stop
  fi
done
free -m | head -2
echo "== pattn-a2 end $(el)"
