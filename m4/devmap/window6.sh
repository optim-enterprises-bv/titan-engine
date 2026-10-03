#!/bin/bash
# devmap2 window 3 (of 3), final binary bin/mistralrs-titan-devmap2 (no build): [Q] qwen3-14b as models-deploy.toml
# runs it (no pin, prefix cache off): mapping, short decode, ~16k prompt; qwen3 G1 / G3 / G3l vs llama.cpp;
# [I] gemma4-12b f32, spark-x2.5, REDCELL G1 deployed binary vs new (same args); [D] dry run of models-nopins2.toml
# (no pins, qwen3 prefix cache default) with the deployed service's ExecStart / env / MemoryMax; [G2] DeepSeek-R1-Qwen3-8B
# Q8_0 @65536 mapping and a 4k prompt.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/top-devmap/m4/devmap/window6.sh
source $HOME/titan-engine/top-devmap/m4/devmap/lib.sh
exec > >(tee -a $I/window6-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin devmap2-window6 58
BIN=$NEWBIN; echo "binary: $BIN sha256 $(sha256sum $BIN | cut -d' ' -f1); deployed: $(sha256sum $OLDBIN | cut -c1-16)"
QM="$M/qwen3-14b Qwen3-14B-vanilla-Q5_K_M.gguf"; QE="TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64"

echo "== [Q] qwen3-14b, deploy settings: no pin, 16384, prefix cache off ($(el))"
if srv_start fq $QM "$QE TITAN_DEVMAP_LOG=1" --max-seq-len 16384 --prefix-cache-n 0; then
  maplines $SLOG | grep -q ": cpu" && echo "Q mapping: FAIL offloads" || echo "Q mapping: 40/40 on the GPU"
  probe $O/fq-short.json 50 256; probe $O/fq-long.json 16200 64; probe $O/fq-short2.json 50 256
  memlines $SLOG | grep after | tail -3
  tgate g1 $O/fq-g1.json $I/prompts/qwen3-14b.json | tail -1; tgate g3 $O/fq-g3.json | tail -1; tgate g3 $O/fq-g3l.json 28000 | tail -1
  srv_stop
fi
tmapq $O/fq-g1.json $O/fq-g3.json $O/fq-g3l.json
cmpg g1 $O/fq-g1.json $O/lref-g1.json
for g in g3 g3l; do python3 $E/m4/g4acc/g3spread.py $O/lref-$g.json $O/spread/lub256-$g.json,$O/spread/lub128-$g.json,$O/spread/lb512-$g.json -- $O/fq-$g.json | tail -1; done

echo "== [I] G1 identity, deployed binary vs new ($(el))"
GA="--dtype f32 --max-model-len 11264 --max-seq-len 11264 --prefix-cache-n 0 -n 0:48"
RA="--dtype bf16 --max-model-len 8192 --max-seq-len 8192 --prefix-cache-n 0 -n 0:30"
SA="--max-model-len 16384 --max-seq-len 16384"
for t in old new; do
  [ $t = old ] && BIN=$OLDBIN || BIN=$NEWBIN
  if srv_start ig-$t $M/gemma4-12b-qat gemma-4-12b-it-qat-q4_0.gguf "" $GA; then tgate g1 $O/ig-$t-g1.json $G/prompts/gemma4-12b.json | tail -1; srv_stop; fi
  if srv_start is-$t $M/spark-x2.5 Spark-X2.5-4B-Q4_K_M.gguf "" $SA; then tgate g1 $O/is-$t-g1.json $G/prompts/spark.json | tail -1; srv_stop; fi
  if srv_start ir-$t $M/redcell-26b REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf "" $RA; then tgate g1 $O/ir-$t-g1.json $G/prompts/redcell.json | tail -1; srv_stop; fi
done
BIN=$NEWBIN
for m in ig is ir; do python3 $I/same.py $O/$m-new-g1.json $O/$m-old-g1.json; done

echo "== [D] dry run, models-nopins2.toml ($(el), $(left)s left)"
if cfg_start dry2 $I/models-nopins2.toml; then
  timeout $(( $(left) - 300 )) python3 $I/dryrun.py $PORT $I/models-nopins2.toml $SLOG $O/dryrun2.json | grep -v "^    reasoning"
  srv_stop
  sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -E "titan swap: loaded|Layers [0-9]|titan tiered auto" | sed -E 's/^.*INFO [a-z_:0-9]*: //; s/ in [0-9.]+ s.*//' | cut -c1-200
fi

echo "== [G2] DeepSeek-R1-Qwen3-8B Q8_0 @65536 ($(el), $(left)s left)"
if [ $(left) -gt 360 ] && MEM=14G srv_start g2-d64 $I/models/ds8b DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf "$QE TITAN_DEVMAP_LOG=1" --max-seq-len 65536; then
  maplines $SLOG | grep -q ": cpu" && echo "G2 d64: mapping offloads" || echo "G2 d64: mapping all GPU"
  PT=$(( $(left) - 120 )) probe $O/g2c-d64.json 50,4000 32; srv_stop
fi
free -m | head -2
echo "== devmap2-window6 end $(el)"
