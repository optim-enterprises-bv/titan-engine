#!/bin/bash
# devmap window 3 (last): [M] qwen3-14b prompt transients on the v2 binary, --prefix-cache-n 0 (all 40 layers at
# 8k / 16k; 30 layers at 12k / 24k); [R] regression core on v2 while the qwen3 activation term is calibrated
# (go.flag); [B] v3 build (the calibration touches only the qwen3 estimate) + unit tests; [G1] qwen3-14b @16384 and
# gemma4-12b f32 @11264 (11.2k prompt) unpinned; [G2] qwen3-14b @65536 and DeepSeek-R1-Qwen3-8B Q8_0 @65536, ~30k
# prompt; [D] from-config dry run of models-nopins.toml (deployed ExecStart / env / MemoryMax, port 18650); [R3] 35B
# MTP=2 gate on v3 if time is left.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/top-devmap/m4/devmap/window3.sh
source $HOME/titan-engine/top-devmap/m4/devmap/lib.sh
exec > >(tee -a $I/window3-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin devmap-window3 60
rm -f $I/go.flag $I/stop.flag
echo "src: $(git -C $SRC log --oneline -1 | cut -c1-90) $(git -C $SRC diff --quiet && echo clean || echo DIRTY)"
BIN=$NEWBIN; echo "binary v2: sha256 $(sha256sum $BIN | cut -d' ' -f1)"
QM="$M/qwen3-14b Qwen3-14B-vanilla-Q5_K_M.gguf"; QE="TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64"

echo "== [M] qwen3-14b transients, prefix cache off ($(el))"
if srv_start mq40 $QM "$QE TITAN_DEVMAP_LOG=1" --max-seq-len 16384 --prefix-cache-n 0 -n 0:40; then
  probe $O/mq40.json 8000,16000 16; memlines $SLOG | grep after; srv_stop
fi
if srv_start mq30 $QM "$QE TITAN_DEVMAP_LOG=1" --max-seq-len 32768 --prefix-cache-n 0 -n 0:30; then
  PT=500 probe $O/mq30.json 12000,24000 16; memlines $SLOG | grep after; srv_stop
fi

echo "== [R] regression core on v2 ($(el))"
coll devmap-m2 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2 && cmp40 $E/m6/out/h-off.json $E/m3/out/devmap-m2.json
coll devmap-mf0 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=0 && cmp40 $E/m6/out/mtpfile-off.json $E/m3/out/devmap-mf0.json
if srv_start b-off $M/bonsai2-27b Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf "TITAN_MTP=0 TITAN_REASONING_EFFORT=medium" --max-seq-len 65536; then
  $C $I/b2client.py greedy titan $PORT b-off 8 300; srv_stop
fi
$C $I/b2client.py cmp integ3-o-off b-off
if srv_start i-new $M/q35-lowbit Qwen3.6-35B-A3B-UD-IQ2_M.gguf "TITAN_MTP=2 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64" --max-seq-len 65536; then
  $C $I/b2client.py greedy titan $PORT i-new 8 300; srv_stop
fi
$C $I/b2client.py cmp integ3-i-new i-new
if srv_start q-new $M/bonsai/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf "TITAN_ATTN_FLASH_PREFILL=1" --max-seq-len 32768; then
  $C $I/b2client.py greedy titan $PORT q-new 8 300; srv_stop
fi
$C $I/b2client.py cmp integ3-q-new q-new

echo "== [P] calibration commit: touch $I/go.flag ($(el), $(left)s left)"
date -Is > $O/waiting.flag
while [ ! -e $I/go.flag ] && [ ! -e $I/stop.flag ] && [ $(left) -gt 1260 ]; do sleep 5; done
rm -f $O/waiting.flag
if [ -e $I/go.flag ]; then
  echo "go: $(git -C $SRC log --oneline -1 | cut -c1-90) $(git -C $SRC diff --quiet && echo clean || echo DIRTY); v2..HEAD: $(git -C $SRC diff --stat 38b8a6b7d..HEAD | tail -1)"
  git -C $SRC diff --stat c3e40cbae..HEAD
  echo "== [B] v3 build ($(el))"
  if mbuild 600; then cp $MBIN $NEWBIN; echo "binary v3: sha256 $(sha256sum $NEWBIN | cut -d' ' -f1)"; fi
  ( mtest unit 900 -p mistralrs-core --lib devmap_ > $O/mtest.status 2>&1 ) &
  TP=$!
fi
BIN=$NEWBIN

echo "== [G1] unpinned, near-max prompt ($(el))"
if MEM=14G srv_start g1-q16 $QM "$QE TITAN_DEVMAP_LOG=1" --max-seq-len 16384; then
  maplines $SLOG | grep -q ": cpu" && echo "G1 q16: mapping offloads" || echo "G1 q16: mapping all GPU"
  probe $O/g1b-q16.json 1000,16000 64; memlines $SLOG | grep after | tail -2; srv_stop
fi
if MEM=14G srv_start g1-gf $M/gemma4-12b-qat gemma-4-12b-it-qat-q4_0.gguf "TITAN_DEVMAP_LOG=1" --dtype f32 --max-model-len 11264 --max-seq-len 11264 --prefix-cache-n 0; then
  maplines $SLOG | grep -q ": cpu" && echo "G1 gf: FAIL mapping offloads" || echo "G1 gf: mapping all GPU"
  probe $O/g1b-gf.json 1000,11150 64; memlines $SLOG | grep after | tail -2; srv_stop
fi
echo "== [G2] does not fit: offload + ~30k prompt ($(el))"
for k in q64 d64; do
  [ $k = d64 ] && [ $(left) -lt 1080 ] && { echo "skip d64: $(left)s left"; continue; }
  if [ $k = q64 ]; then set -- $QM; else set -- $M DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf; fi
  if MEM=14G srv_start g2-$k $1 $2 "$QE TITAN_DEVMAP_LOG=1" --max-seq-len 65536; then
    maplines $SLOG | grep -q ": cpu" && echo "G2 $k: mapping offloads" || echo "G2 $k: FAIL mapping (no layer on the CPU)"
    PT=600 probe $O/g2b-$k.json 1000,30000 32; memlines $SLOG | grep after | tail -2; srv_stop
  fi
done
[ -n "${TP:-}" ] && { wait $TP; cat $O/mtest.status; }

echo "== [D] dry run, models-nopins.toml ($(el), $(left)s left)"
if cfg_start dry $I/models-nopins.toml; then
  timeout $(( $(left) - 60 )) python3 $I/dryrun.py $PORT $I/models-nopins.toml $SLOG $O/dryrun.json | grep -v "^    reasoning"
  srv_stop
  sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -E "titan swap: load|Layers [0-9]|Automatic device map estimate|titan tiered" | sed 's/^.*INFO [a-z_:]*: //' | cut -c1-300
fi
echo "== [R3] 35B MTP=2 on v3 ($(el), $(left)s left)"
coll devmap-m2v3 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2 && cmp40 $E/m6/out/h-off.json $E/m3/out/devmap-m2v3.json
free -m | head -2
echo "== devmap-window3 end $(el)"
