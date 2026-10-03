#!/bin/bash
# pcache window 1: [B] nvcc-free build of top-pcache/mr-pcache; [T1] config unit tests (mistralrs-cli);
# [S] from-config swap gate on roster.toml: 35B shared-prefix TTFT -> gemma4-12b (prefix_cache_n 0) 6 x ~4k ->
# 35B TTFT again; then gemma4 default (16) / byte cap, REDCELL default / 0 / byte cap, 6 x ~4k each; nvidia-smi after
# every swap; [R] regression core; [T2] prefix_cacher CUDA unit test if time is left.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/pcache/window1.sh
source $HOME/titan-engine/m4/pcache/lib.sh
exec > >(tee -a $I/window1-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin pcache-window1 58
echo "src: $(git -C $SRC log --oneline -1 | cut -c1-80) $(git -C $SRC diff --quiet && echo clean || echo DIRTY)"
C=python3; P=$I/pcache.py; CL=$I/b2client.py

echo "== [B] mistral.rs build ($(el))"
rm -f $I/rebuild.flag
until mbuild 1500; do
  echo "build failed: waiting for $I/rebuild.flag ($(left)s left)"; date -Is > $O/build-failed.flag
  while [ ! -e $I/rebuild.flag ] && [ $(left) -gt 2100 ]; do sleep 10; done
  [ -e $I/rebuild.flag ] || { echo "no binary: ending the window"; exit 1; }
  rm -f $I/rebuild.flag $O/build-failed.flag; echo "rebuilding ($(el)): $(git -C $SRC log --oneline -1 | cut -c1-80)"
done
cp $MBIN $NEWBIN; BIN=$NEWBIN; echo "binary: $BIN sha256 $(sha256sum $BIN | cut -d' ' -f1) from $(git -C $SRC log --oneline -1 | cut -c1-60)"

echo "== [T1] config unit tests ($(el))"
mtest cli 900 -p mistralrs-cli --features oxide --bin mistralrs config::tests

echo "== [S] swap gate, roster.toml ($(el))"
if cfg_start roster $I/roster.toml; then
  echo "VRAM after 35B (default) load: $(smi)"
  timeout 600 $C $P pc $PORT qwen3.6-35b q35-before
  timeout 300 $C $P touch $PORT gemma4-12b
  timeout 600 $C $P long $PORT gemma4-12b g12-pc0 6
  timeout 600 $C $P touch $PORT qwen3.6-35b
  timeout 600 $C $P pc $PORT qwen3.6-35b q35-after
  timeout 300 $C $P touch $PORT gemma4-12b-def
  timeout 600 $C $P long $PORT gemma4-12b-def g12-def 6 13500 1
  timeout 300 $C $P touch $PORT gemma4-12b-cap
  timeout 600 $C $P long $PORT gemma4-12b-cap g12-cap 6 13500 1
  timeout 300 $C $P touch $PORT redcell-26b-def
  timeout 600 $C $P long $PORT redcell-26b-def red-def 6 13500 1
  timeout 300 $C $P touch $PORT redcell-26b-pc0
  timeout 600 $C $P long $PORT redcell-26b-pc0 red-pc0 6
  timeout 300 $C $P touch $PORT redcell-26b-cap
  timeout 600 $C $P long $PORT redcell-26b-cap red-cap 6 13500 1
  srv_stop
  echo "-- server log: swaps, per-model prefix cache, byte-cap evictions, OOMs"
  grep -E "titan swap: (loaded|unloaded|\[)|titan settings for the next|Prefix caching enabled|Prefix cache:|OUT_OF_MEMORY|out of memory|panicked" $SLOG | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-260
fi
$C $P same $O/g12-pc0.json $E/m4/integ/out/g12-g3.json
$C $P same $O/g12-def.json $O/g12-pc0.json; $C $P same $O/g12-cap.json $O/g12-pc0.json

echo "== [R1] 35B gate pair ($(el))"
coll pcache-m2 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2 && cmp40 $E/m6/out/h-off.json $E/m3/out/pcache-m2.json
coll pcache-mf0 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=0 && cmp40 $E/m6/out/mtpfile-off.json $E/m3/out/pcache-mf0.json
echo "== [R2] Bonsai-2 27B B2-off vs o-off ($(el))"
if srv_start b-off $M/bonsai2-27b Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf "TITAN_MTP=0 TITAN_REASONING_EFFORT=medium" --max-seq-len 65536; then
  $C $CL greedy titan $PORT b-off 8 300; srv_stop
fi
$C $CL cmp o-off b-off
echo "== [R3] IQ2_M, Bonsai-27B Q1_0 new vs integ, 8 x 300 ($(el))"
if srv_start i-new $M/q35-lowbit Qwen3.6-35B-A3B-UD-IQ2_M.gguf "TITAN_MTP=2 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64" --max-seq-len 65536; then
  $C $CL greedy titan $PORT i-new 8 300; srv_stop
fi
$C $CL cmp i-integ i-new; $C $CL cmp i-old-b2merge i-new
if srv_start q-new $M/bonsai/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf "TITAN_ATTN_FLASH_PREFILL=1" --max-seq-len 32768; then
  $C $CL greedy titan $PORT q-new 8 300; srv_stop
fi
$C $CL cmp q-integ q-new; $C $CL cmp q-old-b2merge q-new

echo "== [T2] prefix_cacher unit tests incl. the CUDA byte-limit test ($(el))"
if [ $(left) -gt 1080 ]; then mtest core $(( $(left) - 180 )) -p mistralrs-core --features oxide --lib prefix_cacher::; else echo "skip [T2]: $(left)s left"; fi

echo "== [O] old binary (bin/mistralrs-titan-integ) re-runs if time is left ($(el))"
BIN=$OLDBIN
if [ $(left) -gt 420 ] && srv_start i-old $M/q35-lowbit Qwen3.6-35B-A3B-UD-IQ2_M.gguf "TITAN_MTP=2 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64" --max-seq-len 65536; then
  $C $CL greedy titan $PORT i-old 8 300; srv_stop; $C $CL cmp i-old i-new
fi
if [ $(left) -gt 420 ] && srv_start q-old $M/bonsai/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf "TITAN_ATTN_FLASH_PREFILL=1" --max-seq-len 32768; then
  $C $CL greedy titan $PORT q-old 8 300; srv_stop; $C $CL cmp q-old q-new
fi
BIN=$NEWBIN
free -m | head -2
echo "== pcache-window1 end $(el)"
