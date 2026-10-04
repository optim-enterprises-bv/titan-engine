#!/bin/bash
# pattn phase B window 3: [B] clean rebuild of the committed sources (stage bin/mistralrs-titan-pattn2); [V] identity vs
# B2's runs; [P] REDCELL paged OFF G3/G5 (comparison for B2's paged ON run); gemma4-12b paged ON on the final binary;
# [C] qwen3-14b concurrency paged ON / OFF; [R] REGRESSION CORE on the final binary.
source $HOME/titan-engine/top-pattn/m4/pattn/lib.sh
restic_wait
exec > >(tee -a $I/window_b3-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin pattn-b3 58
echo "src: mistral.rs $(git -C $SRC log --oneline -1 | cut -c1-50) $(git -C $SRC diff --quiet && echo clean || echo DIRTY); oxide $(git -C $OX log --oneline -1 | cut -c1-50) $(git -C $OX diff --quiet && echo clean || echo DIRTY)"
build_bin2 1500
SK=Spark-X2.5-4B-Q4_K_M.gguf; SP=$G/prompts/spark.json; SA="--max-model-len 16384 --max-seq-len 16384 --prefix-cache-n 0"
QM=$M/qwen3-14b; QF=Qwen3-14B-vanilla-Q5_K_M.gguf; QP=$E/m4/devmap/prompts/qwen3-14b.json; QE="TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64"
RA="--dtype bf16 --max-model-len 8192 --max-seq-len 8192 --prefix-cache-n 0 -n 0:30"
GA="--dtype f32 --max-model-len 11264 --max-seq-len 11264 --prefix-cache-n 0 -n 0:48"
echo "== [V] identity vs B2 ($(el))"
if PA=on srv_start s8-on $M/spark-x2.5 $SK "" $SA; then gate g1 s8-on-g1 $SP | tail -1; gate g3 s8-on-g3l 28000 | tail -1; gate g5 s8-on-g5 8000 128 | tail -1; srv_stop; fi
if PA=on srv_start f8x $M/spark-x2.5 $SK "" $SA --pa-cache-type f8e4m3; then timeout 600 python3 $I/decodekl.py run $PORT $SP $O/f8x-dkl.json 16 128; srv_stop; fi
same $O/s8-on-g1.json $O/s7-on-g1.json; same $O/s8-on-g3l.json $O/s7-on-g3l.json; python3 $I/decodekl.py cmp $O/f7-on-dkl.json $O/f8x-dkl.json
echo "== [P] REDCELL paged OFF G3 / G5; gemma4-12b paged ON ($(el))"
if srv_start r8-off $M/redcell-26b REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf "" $RA; then
  echo "VRAM after load: $(smi)"; gate g3 r8-off-g3 | tail -1; gate g5 r8-off-g5 8000 128 | tail -1; srv_stop; fi
cmpg g3 $O/r7-on-g3.json $O/r8-off-g3.json | head -1; tps $O/r7-on-g5.json $O/r8-off-g5.json $O/r7-on-g3.json $O/r8-off-g3.json
if PA=on srv_start g8-on $M/gemma4-12b-qat gemma-4-12b-it-qat-q4_0.gguf "TITAN_PATTN_TRACE=1" $GA --pa-context-len 11264; then
  echo "VRAM after load: $(smi)"; palines $SLOG 3
  gate g1 g8-on-g1 $G/prompts/gemma4-12b.json | tail -1; gate g3 g8-on-g3 | tail -1; gate g5 g8-on-g5 8000 128 | tail -1; srv_stop
  sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -E " WARN | ERROR |panicked|out of memory|not supported" | grep -v "^pattn" | sort -u | head -5 | cut -c1-220
fi
[ -e $O/g8-on-g1.json ] && same $O/g8-on-g1.json $O/dep-ig-g1.json
echo "== [C] qwen3-14b concurrency --max-seqs 4 --prefix-cache-n 0, paged ON / OFF ($(el))"
for pa in on off; do
  if PA=$pa srv_start cq-$pa $QM $QF "$QE" --max-seq-len 16384 --max-seqs 4 --prefix-cache-n 0; then
    sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -E "Layers [0-9]|disabling PagedAttention|max_num_seqs" | sed 's/^.*INFO [a-z_:0-9]*: //' | head -2
    conc $QP $O/cq-$pa-conc.json 4 128 0; srv_stop; fi
done
echo "== [R1] REGRESSION CORE (paged attention OFF): 35B gate pair ($(el))"
coll pattn3-m2 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2 && cmp40 $E/m6/out/h-off.json $E/m3/out/pattn3-m2.json
coll pattn3-mf0 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=0 && cmp40 $E/m6/out/mtpfile-off.json $E/m3/out/pattn3-mf0.json
echo "== [R2] Bonsai-2 B2-off, IQ2_M, Bonsai-27B Q1_0 vs the deployed binary's runs ($(el))"
if srv_start b3-off $M/bonsai2-27b Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf "TITAN_MTP=0 TITAN_REASONING_EFFORT=medium" --max-seq-len 65536; then
  $C $CL greedy titan $PORT b3-off 8 300; srv_stop; fi
$C $CL cmp o-off b3-off; $C $CL cmp dep-b-off b3-off
if srv_start i3-new $M/q35-lowbit Qwen3.6-35B-A3B-UD-IQ2_M.gguf "TITAN_MTP=2 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64" --max-seq-len 65536; then
  $C $CL greedy titan $PORT i3-new 8 300; srv_stop; fi
$C $CL cmp dep-i-new i3-new
if srv_start q3-new $M/bonsai/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf "TITAN_ATTN_FLASH_PREFILL=1" --max-seq-len 32768; then
  $C $CL greedy titan $PORT q3-new 8 300; srv_stop; fi
$C $CL cmp dep-q-new q3-new
echo "== [R3] Spark / gemma4 / REDCELL G1 vs the deployed binary's runs ($(el))"
GA="--dtype f32 --max-model-len 11264 --max-seq-len 11264 --prefix-cache-n 0 -n 0:48"
SA0="--max-model-len 16384 --max-seq-len 16384"
if srv_start is3 $M/spark-x2.5 $SK "" $SA0; then gate g1 is3-g1 $SP | tail -1; srv_stop; fi
if srv_start ig3 $M/gemma4-12b-qat gemma-4-12b-it-qat-q4_0.gguf "" $GA; then gate g1 ig3-g1 $G/prompts/gemma4-12b.json | tail -1; srv_stop; fi
if srv_start ir3 $M/redcell-26b REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf "" $RA; then gate g1 ir3-g1 $G/prompts/redcell.json | tail -1; srv_stop; fi
for m in is ig ir; do [ -e $O/${m}3-g1.json ] && same $O/${m}3-g1.json $O/dep-$m-g1.json; done
free -m | head -2
echo "== pattn-b3 end $(el)"
