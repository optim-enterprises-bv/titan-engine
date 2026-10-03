#!/bin/bash
# orca window 3 (of 3), final binary bin/mistralrs-titan-orca: [R] regression core vs bin/mistralrs-titan-swap (35B MTP=2 /
# MTP-off 40/40 incl. the tiered auto fraction and free VRAM, Bonsai-2 B2-off 8/8, IQ2_M and Bonsai-27B Q1_0 8 x 300 vs
# the deployed binary's recorded runs); [I] gemma4 / Spark / REDCELL G1 vs the deployed binary's recorded runs;
# [D] full roster dry run: m4/orca/models-deploy.toml (deploy/models.toml + orcasaq2-cyber-27b) on a scratch port with the
# service's ExecStart / env.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/top-orca/m4/orca/window3.sh
source $HOME/titan-engine/top-orca/m4/orca/lib.sh
exec > >(tee -a $I/window3-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin orca-window3 58
BIN=$NEWBIN; echo "binary: $BIN sha256 $(sha256sum $BIN | cut -d' ' -f1); deployed: $(sha256sum $OLDBIN | cut -c1-16); src $(git -C $SRC log --oneline -1 | cut -c1-50) $(git -C $SRC diff --quiet && echo clean || echo DIRTY)"

echo "== [R1] 35B gate pair ($(el))"
coll orca-m2 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2 && cmp40 $E/m6/out/h-off.json $E/m3/out/orca-m2.json
coll orca-mf0 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=0 && cmp40 $E/m6/out/mtpfile-off.json $E/m3/out/orca-mf0.json
for n in devmap2-m2 orca-m2 devmap2-mf0 orca-mf0; do
  echo "$n: $(sed 's/\x1b\[[0-9;]*m//g' $E/m3/out/$n.server.log | grep -E 'titan tiered auto' | head -1 | sed 's/^.*INFO [a-z_:0-9]*: //' | cut -c1-200)"
  echo "   $(sed 's/\x1b\[[0-9;]*m//g' $E/m3/out/$n.server.log | grep -E 'Recurrent state pool|recurrent' | head -1 | cut -c1-160)"
  echo "   VRAM after the run: $(sed 's/\x1b\[[0-9;]*m//g' $E/m3/out/$n.server.log | grep -E 'titan tiered experts:' | head -1 | sed 's/^.*INFO [a-z_:0-9]*: //' | cut -c1-160)"
done
echo "== [R2] Bonsai-2 B2-off, IQ2_M, Bonsai-27B Q1_0 vs the deployed binary's runs ($(el))"
if srv_start b-off $M/bonsai2-27b Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf "TITAN_MTP=0 TITAN_REASONING_EFFORT=medium" --max-seq-len 65536; then
  echo "VRAM after load: $(smi)"; $C $CL greedy titan $PORT b-off 8 300; srv_stop; fi
$C $CL cmp o-off b-off; $C $CL cmp dep-b-off b-off
if srv_start i-new $M/q35-lowbit Qwen3.6-35B-A3B-UD-IQ2_M.gguf "TITAN_MTP=2 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64" --max-seq-len 65536; then
  echo "VRAM after load: $(smi)"; $C $CL greedy titan $PORT i-new 8 300; srv_stop; fi
$C $CL cmp dep-i i-new
if srv_start q-new $M/bonsai/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf "TITAN_ATTN_FLASH_PREFILL=1" --max-seq-len 32768; then
  $C $CL greedy titan $PORT q-new 8 300; srv_stop; fi
$C $CL cmp dep-q q-new

echo "== [I] gemma4 / Spark / REDCELL G1 vs the deployed binary's recorded runs (m4/devmap window 6 args) ($(el))"
GA="--dtype f32 --max-model-len 11264 --max-seq-len 11264 --prefix-cache-n 0 -n 0:48"
RA="--dtype bf16 --max-model-len 8192 --max-seq-len 8192 --prefix-cache-n 0 -n 0:30"
SA="--max-model-len 16384 --max-seq-len 16384"
if srv_start ig $M/gemma4-12b-qat gemma-4-12b-it-qat-q4_0.gguf "" $GA; then gate g1 ig-g1 $G/prompts/gemma4-12b.json; srv_stop; fi
if srv_start is $M/spark-x2.5 Spark-X2.5-4B-Q4_K_M.gguf "" $SA; then gate g1 is-g1 $G/prompts/spark.json; srv_stop; fi
if srv_start ir $M/redcell-26b REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf "" $RA; then gate g1 ir-g1 $G/prompts/redcell.json; srv_stop; fi
for m in ig is ir; do same $O/$m-g1.json $O/dep-$m-g1.json; done

echo "== [D] roster dry run, models-deploy.toml on port $PORT ($(el), $(left)s left)"
sed "s/^port = 1234/port = $PORT/" $I/models-deploy.toml > $O/models-dry.toml; grep -n "^port" $O/models-dry.toml
if cfg_start dry $O/models-dry.toml; then
  timeout $(( $(left) - 200 )) python3 $I/dryrun.py $PORT $O/models-dry.toml $SLOG $O/dryrun.json | grep -v "^    reasoning"
  MODELS=orcasaq2-cyber-27b timeout 600 python3 $I/probe.py $PORT $O/dry-orca-ctx.json 2000,7900 64 orcasaq2-cyber-27b
  srv_stop
  sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -E "titan swap: loaded|Layers [0-9]|titan tiered auto|titan mtp stats" | sed -E 's/^.*INFO [a-z_:0-9]*: //' | cut -c1-200
fi
free -m | head -2
echo "== orca-window3 end $(el)"
