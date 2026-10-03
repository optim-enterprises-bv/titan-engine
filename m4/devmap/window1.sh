#!/bin/bash
# devmap window 1: measure. [B] build the instrumented binary (GGUF tensor inventory for native-path weights, Gemma 4's
# F16 KV under f32, the estimate log line, TITAN_DEVMAP_LOG step VRAM log) in the background while [O] the deployed
# binary loads each native-path model unpinned (its mapping); then [N] the new binary: unpinned load (estimate +
# mapping), pinned all-GPU run with prompts of increasing length (pool used / peak per prompt step, nvidia-smi peak).
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/top-devmap/m4/devmap/window1.sh
source $HOME/titan-engine/top-devmap/m4/devmap/lib.sh
exec > >(tee -a $I/window1-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin devmap-window1 58
echo "src: $(git -C $SRC log --oneline -1 | cut -c1-70) + $(git -C $SRC diff --stat | tail -1)"
Q=(qwen3-14b Qwen3-14B-vanilla-Q5_K_M.gguf "TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64")
G=(gemma4-12b-qat gemma-4-12b-it-qat-q4_0.gguf ""); R=(redcell-26b REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf ""); S=(spark-x2.5 Spark-X2.5-4B-Q4_K_M.gguf "")
declare -A ARGS=(
  [q16]="--max-seq-len 16384" [q64]="--max-seq-len 65536"
  [gf]="--dtype f32 --max-model-len 11264 --max-seq-len 11264 --prefix-cache-n 0"
  [gb]="--dtype bf16 --max-model-len 12288 --max-seq-len 12288 --prefix-cache-n 0"
  [rc]="--dtype bf16 --max-model-len 8192 --max-seq-len 8192 --prefix-cache-n 0"
  [sp]="--max-model-len 16384 --max-seq-len 16384")
declare -A PIN=([q16]="-n 0:40" [q64]="" [gf]="-n 0:48" [gb]="-n 0:48" [rc]="-n 0:30" [sp]="-n 0:36")
declare -A LENS=([q16]="1000,4000,8000,12000,16000" [q64]="1000,16000,30000" [gf]="1000,4000,8000,11300" [gb]="1000,8000,12100"
  [rc]="1000,4000,8000" [sp]="1000,8000,16000")
mdl() { case $1 in q*) echo "${Q[@]:0:2}";; g*) echo "${G[@]:0:2}";; rc) echo "${R[@]:0:2}";; sp) echo "${S[@]:0:2}";; esac; }
menv() { case $1 in q*) echo "${Q[2]}";; *) echo "";; esac; }

echo "== [B] build, background ($(el))"
( mbuild 1800 > $O/mbuild.status 2>&1 ) &
BP=$!
echo "== [O] deployed binary, unpinned loads (MemoryMax 12G while the build runs) ($(el))"
BIN=$OLDBIN
for k in q16 q64 gf gb rc sp; do
  set -- $(mdl $k)
  if MEM=12G srv_start old-$k $M/$1 $2 "$(menv $k)" ${ARGS[$k]}; then
    [ $k = q16 ] && PT=300 probe $O/old-q16.json 300 128
    srv_stop
  fi
done
echo "== [B] waiting for the build ($(el))"
wait $BP; cat $O/mbuild.status
[ -x $MBIN ] && grep -q Finished $O/build-$WN.log || { echo "no binary"; exit 1; }
cp $MBIN $NEWBIN; BIN=$NEWBIN; echo "binary: $BIN sha256 $(sha256sum $BIN | cut -d' ' -f1)"
echo "== [N] new binary: unpinned load, then pinned prompts ($(el))"
for k in q16 gf gb rc sp q64; do
  set -- $(mdl $k)
  if srv_start new-$k $M/$1 $2 "$(menv $k) TITAN_DEVMAP_LOG=1" ${ARGS[$k]}; then
    if [ $k = q64 ] || ! maplines $SLOG | grep -q ": cpu"; then
      echo "-- all on the GPU (or q64): probing this server"; probe $O/new-$k.json ${LENS[$k]} 64; memlines $SLOG; srv_stop; continue
    fi
    srv_stop
  fi
  if srv_start pin-$k $M/$1 $2 "$(menv $k) TITAN_DEVMAP_LOG=1" ${ARGS[$k]} ${PIN[$k]}; then
    probe $O/pin-$k.json ${LENS[$k]} 64; memlines $SLOG; srv_stop
  fi
done
free -m | head -2
echo "== devmap-window1 end $(el)"
