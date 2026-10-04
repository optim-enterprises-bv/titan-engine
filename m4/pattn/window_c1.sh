#!/bin/bash
# pattn3 window 1: [B] build (per-model paged_attn / pa_cache_type / pa_context_len / pa_memory_mb /
# max_num_batched_tokens / max_seqs in [models.titan]), stage bin/mistralrs-titan-pattn3; [U] CLI unit tests;
# [S] swap gate on one from-config server with models-deploy.toml (port 18670): 35B -> Spark -> qwen3-14b -> gemma4-12b
# -> REDCELL -> OrcaSAQ -> 35B, memory after each; Spark / qwen3 G1 / G3 / G5 / concurrency through the swap server;
# [R] REGRESSION CORE (paged off) vs bin/mistralrs-titan-swap; [U2] core unit tests (scheduler templates) if time allows.
source $HOME/titan-engine/top-pattn/m4/pattn/lib.sh
restic_wait
exec > >(tee -a $I/window_c1-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin pattn-c1 58
echo "src: mistral.rs $(git -C $SRC log --oneline -1 | cut -c1-50) $(git -C $SRC diff --quiet && echo clean || echo DIRTY)"
build_bin3 1500
utest 900 mistralrs-cli titan_

echo "== [S] swap gate: models-deploy.toml on port $PORT ($(el))"
sed "s/^port = 1234/port = $PORT/" $I/models-deploy.toml > $O/models-gate.toml; grep -n "^port" $O/models-gate.toml
SP=$E/m4/g4s/prompts/spark.json; QP=$E/m4/devmap/prompts/qwen3-14b.json
ln=1
step() { # print the swap lines since the last step, then VRAM
  echo "   -- swap log:"; swaplog $SLOG $ln | sed 's/^/      /'; ln=$(( $(wc -l < $SLOG) + 1 ))
  echo "   -- VRAM: $(smi)"
}
gm() { local m=$1 mode=$2 out=$3; shift 3; MODEL=$m timeout 900 python3 $I/gates_m.py $mode $PORT titan $O/$out.json "$@"; }
if cfg_start sw $O/models-gate.toml; then
  step
  echo "-- 35B before"; timeout 600 $C $P pc $PORT qwen3.6-35b c1-q35-before | tail -1; step
  echo "-- Spark"; timeout 300 $C $P touch $PORT spark-x2.5; step
  timeout 600 $C $P burst $PORT spark-x2.5 c1-spark-8k 4 28000 64 | tail -2; step
  MODEL=spark-x2.5 conc $SP $O/c1-spark-conc.json 4 128 0; MODEL=spark-x2.5 conc $SP $O/c1-spark-conc2.json 4 256 8
  gm spark-x2.5 g1 c1-s-g1 $SP | tail -1; gm spark-x2.5 g3 c1-s-g3 | tail -1; gm spark-x2.5 g3 c1-s-g3l 28000 | tail -1; gm spark-x2.5 g5 c1-s-g5 8000 128 | tail -1; step
  echo "-- qwen3-14b"; timeout 300 $C $P touch $PORT qwen3-14b; step
  timeout 900 $C $P burst $PORT qwen3-14b c1-q14-16k 4 48000 64 | tail -2; step
  MODEL=qwen3-14b conc $QP $O/c1-q14-conc.json 4 128 0
  gm qwen3-14b g1 c1-q-g1 $QP | tail -1; gm qwen3-14b g3 c1-q-g3l 28000 | tail -1; gm qwen3-14b g3 c1-q-g3x 48000 | tail -1; gm qwen3-14b g5 c1-q-g5 8000 128 | tail -1; step
  echo "-- gemma4-12b"; timeout 900 $C $P long $PORT gemma4-12b c1-g12-8k 2 28000 | tail -1; step
  echo "-- redcell-26b"; timeout 900 $C $P long $PORT redcell-26b c1-red-4k 3 | tail -1; step
  echo "-- orcasaq2-cyber-27b"; timeout 900 $C $P long $PORT orcasaq2-cyber-27b c1-orca-8k 1 27000 | tail -1; step
  echo "-- 35B after"; timeout 600 $C $P touch $PORT qwen3.6-35b; step
  timeout 600 $C $P pc $PORT qwen3.6-35b c1-q35-after | tail -1; timeout 600 $C $P pc $PORT qwen3.6-35b c1-q35-after2 | tail -1; step
  srv_stop
  echo "-- tiered fraction lines (35B loads):"; sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -E "titan tiered auto" | sed -E 's/^.*INFO [a-z_:0-9]*: //' | cut -c1-200
  echo "-- unload / load VRAM:"; sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -E "titan swap: (loaded|unloaded)|VRAM free before any model" | sed -E 's/^.*INFO [a-z_:0-9]*: //' | cut -c1-200
fi
echo "-- identity vs pattn2 (standalone, paged ON)"
same $O/c1-s-g1.json $O/s7-big-g1.ids.json; same $O/c1-s-g3.json $O/s7-big-g3.ids.json; same $O/c1-s-g3l.json $O/s7-big-g3l.ids.json
same $O/c1-q-g1.json $O/q7-on-g1.ids.json; same $O/c1-q-g3l.json $O/q7-on-g3l.ids.json; same $O/c1-q-g3x.json $O/q7-on-g3x.ids.json
tps $O/c1-s-g5.json $O/s7-big-g5.json $O/c1-q-g5.json $O/q7-on-g5.json
if ! python3 $I/same.py $O/c1-s-g3l.json $O/s7-big-g3l.ids.json | head -1 | grep -q "4/4"; then
  echo "-- Spark G3 with prefix_cache_n = 0 (the pattn2 runs had the prefix cache off) ($(el))"
  python3 - $O/models-gate.toml $O/models-gate-pc0.toml <<'PY'
import sys
s = open(sys.argv[1]).read()
old = 'paged_attn = "on"\nmax_seqs = 4\nmax_num_batched_tokens = 16384\n'
assert s.count(old) == 1
open(sys.argv[2], "w").write(s.replace(old, old + "prefix_cache_n = 0\n"))
PY
  if cfg_start sw0 $O/models-gate-pc0.toml; then
    gm spark-x2.5 g3 c1-s0-g3 | tail -1; gm spark-x2.5 g3 c1-s0-g3l 28000 | tail -1; srv_stop
    same $O/c1-s0-g3.json $O/s7-big-g3.ids.json; same $O/c1-s0-g3l.json $O/s7-big-g3l.ids.json
  fi
fi

echo "== [R1] REGRESSION CORE (paged attention OFF): 35B gate pair ($(el))"
coll pattn4-m2 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2 && cmp40 $E/m6/out/h-off.json $E/m3/out/pattn4-m2.json
coll pattn4-mf0 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=0 && cmp40 $E/m6/out/mtpfile-off.json $E/m3/out/pattn4-mf0.json
echo "== [R2] Bonsai-2 B2-off, IQ2_M, Bonsai-27B Q1_0 vs the deployed binary's runs ($(el))"
if srv_start b4-off $M/bonsai2-27b Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf "TITAN_MTP=0 TITAN_REASONING_EFFORT=medium" --max-seq-len 65536; then
  $C $CL greedy titan $PORT b4-off 8 300; srv_stop; fi
$C $CL cmp o-off b4-off; $C $CL cmp dep-b-off b4-off
if srv_start i4-new $M/q35-lowbit Qwen3.6-35B-A3B-UD-IQ2_M.gguf "TITAN_MTP=2 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64" --max-seq-len 65536; then
  $C $CL greedy titan $PORT i4-new 8 300; srv_stop; fi
$C $CL cmp dep-i-new i4-new
if srv_start q4-new $M/bonsai/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf "TITAN_ATTN_FLASH_PREFILL=1" --max-seq-len 32768; then
  $C $CL greedy titan $PORT q4-new 8 300; srv_stop; fi
$C $CL cmp dep-q-new q4-new
echo "== [R3] Spark / gemma4 / REDCELL G1 vs the deployed binary's runs ($(el))"
SK=Spark-X2.5-4B-Q4_K_M.gguf
GA="--dtype f32 --max-model-len 11264 --max-seq-len 11264 --prefix-cache-n 0 -n 0:48"
RA="--dtype bf16 --max-model-len 8192 --max-seq-len 8192 --prefix-cache-n 0 -n 0:30"
if srv_start is4 $M/spark-x2.5 $SK "" --max-model-len 16384 --max-seq-len 16384; then gate g1 is4-g1 $SP | tail -1; srv_stop; fi
if srv_start ig4 $M/gemma4-12b-qat gemma-4-12b-it-qat-q4_0.gguf "" $GA; then gate g1 ig4-g1 $G/prompts/gemma4-12b.json | tail -1; srv_stop; fi
if srv_start ir4 $M/redcell-26b REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf "" $RA; then gate g1 ir4-g1 $G/prompts/redcell.json | tail -1; srv_stop; fi
for m in is ig ir; do [ -e $O/${m}4-g1.json ] && same $O/${m}4-g1.json $O/dep-$m-g1.json; done
if [ $(left) -gt 1200 ]; then utest 1100 mistralrs-core titan_swap_template; fi
free -m | head -2
echo "== pattn-c1 end $(el)"
