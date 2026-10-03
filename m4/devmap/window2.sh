#!/bin/bash
# devmap window 2: [B] build (window 1 stopped at a borrow error in the memlog helper; v1 prefill-activation formulas
# + half-device activation cap); [M] measure: per model an unpinned load (estimate line + mapping), then the pinned
# all-GPU server with prompts of increasing length (TITAN_DEVMAP_LOG pool used / peak per prompt step, nvidia-smi peak);
# [P] pause for the calibration commit (go.flag); [R] rebuild, unit tests in the background; [G1] the five models
# unpinned with a near-max prompt; [G2] qwen3-14b @65536 and DeepSeek-R1-Qwen3-8B Q8_0 @65536 with a ~30k prompt.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/top-devmap/m4/devmap/window2.sh
source $HOME/titan-engine/top-devmap/m4/devmap/lib.sh
exec > >(tee -a $I/window2-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin devmap-window2 59
rm -f $I/go.flag $I/stop.flag
echo "src: $(git -C $SRC log --oneline -1 | cut -c1-90) $(git -C $SRC diff --quiet && echo clean || echo DIRTY)"
Q=(qwen3-14b Qwen3-14B-vanilla-Q5_K_M.gguf); G=(gemma4-12b-qat gemma-4-12b-it-qat-q4_0.gguf)
R=(redcell-26b REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf); S=(spark-x2.5 Spark-X2.5-4B-Q4_K_M.gguf)
QE="TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64"
declare -A ARGS=(
  [q16]="--max-seq-len 16384" [q64]="--max-seq-len 65536" [d64]="--max-seq-len 65536"
  [gf]="--dtype f32 --max-model-len 11264 --max-seq-len 11264 --prefix-cache-n 0"
  [gb]="--dtype bf16 --max-model-len 12288 --max-seq-len 12288 --prefix-cache-n 0"
  [rc]="--dtype bf16 --max-model-len 8192 --max-seq-len 8192 --prefix-cache-n 0"
  [sp]="--max-model-len 16384 --max-seq-len 16384")
declare -A PIN=([q16]="-n 0:40" [gf]="-n 0:48" [gb]="-n 0:48" [rc]="-n 0:30" [sp]="-n 0:36")
declare -A LENS=([q16]="1000,4000,8000,12000,16000" [gf]="1000,4000,8000,11300" [gb]="1000,6000,12100" [rc]="1000,4000,8000" [sp]="1000,8000,16000")
declare -A G1N=([q16]=16000 [gf]=11300 [gb]=12100 [rc]=8100 [sp]=16100)
mdl() { case $1 in q*) echo "$M/${Q[0]} ${Q[1]}";; d64) echo "$M DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf";; g*) echo "$M/${G[0]} ${G[1]}";; rc) echo "$M/${R[0]} ${R[1]}";; sp) echo "$M/${S[0]} ${S[1]}";; esac; }
menv() { case $1 in q*|d64) echo "$QE";; *) echo "";; esac; }

echo "== [B] build ($(el))"
mbuild 1500 || exit 1
cp $MBIN $NEWBIN; BIN=$NEWBIN; echo "binary v1: sha256 $(sha256sum $BIN | cut -d' ' -f1)"

echo "== [M] measure ($(el))"
for k in q16 gf gb rc sp q64; do
  set -- $(mdl $k)
  if srv_start est-$k $1 $2 "$(menv $k) TITAN_DEVMAP_LOG=1" ${ARGS[$k]}; then
    [ $k = q64 ] && { PT=600 probe $O/est-q64.json 1000,4000 64; memlines $SLOG; }
    srv_stop
  fi
  [ $k = q64 ] && continue
  if srv_start pin-$k $1 $2 "$(menv $k) TITAN_DEVMAP_LOG=1" ${ARGS[$k]} ${PIN[$k]}; then
    probe $O/pin-$k.json ${LENS[$k]} 64; memlines $SLOG; srv_stop
  fi
done

echo "== [P] waiting for the calibration commit: touch $I/go.flag ($(el), $(left)s left)"
date -Is > $O/waiting.flag
while [ ! -e $I/go.flag ] && [ ! -e $I/stop.flag ] && [ $(left) -gt 1500 ]; do sleep 5; done
rm -f $O/waiting.flag
[ -e $I/go.flag ] || { echo "no go ($(el)): ending the window"; exit 0; }
echo "go: $(git -C $SRC log --oneline -1 | cut -c1-90) $(git -C $SRC diff --quiet && echo clean || echo DIRTY) ($(el))"

echo "== [R] rebuild ($(el))"
mbuild 900 || exit 1
cp $MBIN $NEWBIN; BIN=$NEWBIN; echo "binary v2: sha256 $(sha256sum $BIN | cut -d' ' -f1)"
( mtest unit 1500 -p mistralrs-core --lib devmap_ > $O/mtest.status 2>&1 ) &
TP=$!

echo "== [G1] unpinned, near-max prompt ($(el))"
for k in q16 gf gb rc sp; do
  set -- $(mdl $k)
  if MEM=14G srv_start g1-$k $1 $2 "$(menv $k) TITAN_DEVMAP_LOG=1" ${ARGS[$k]}; then
    maplines $SLOG | grep -q ": cpu" && echo "G1 $k: FAIL mapping (layers on the CPU)" || echo "G1 $k: mapping all GPU"
    probe $O/g1-$k.json 1000,${G1N[$k]} 64; memlines $SLOG | tail -2; srv_stop
  fi
done
echo "== [G2] does not fit: offload + ~30k prompt ($(el))"
for k in q64; do
  set -- $(mdl $k)
  if MEM=14G srv_start g2-$k $1 $2 "$(menv $k) TITAN_DEVMAP_LOG=1" ${ARGS[$k]}; then
    maplines $SLOG | grep -q ": cpu" && echo "G2 $k: mapping offloads" || echo "G2 $k: FAIL mapping (no layer on the CPU)"
    PT=1300 probe $O/g2-$k.json 1000,30000 32; memlines $SLOG | tail -2; srv_stop
  fi
done
echo "== [T] unit tests ($(el))"
wait $TP; cat $O/mtest.status
free -m | head -2
echo "== devmap-window2 end $(el)"
