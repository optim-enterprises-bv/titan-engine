#!/bin/bash
# pattn window 3 (of 3): [B] rebuild with the gqa_grouped_sdpa fix (mistral.rs 0709064c1: paged gather V layout, the
# HTTP 500 on Spark ~8k prompts with paged attention on); [S] Spark ~8k prompts paged ON (the fix) + G1 / G5 again;
# [C] concurrency probe, Spark paged ON / OFF with --max-seqs 4 (window 2's --max-batch-size 4 moved 13 layers to the CPU
# and disabled paged attention: invalid); [R] REGRESSION CORE (paged OFF vs bin/mistralrs-titan-swap); [Q] qwen3-14b
# paged ON with MISTRALRS_CUDA_GRAPHS=0 (upstream decode graphs need the nvcc-only pad_decode_input_u32) + a
# TITAN_NVCC_ONLY=warn discovery run listing every unported launcher qwen3 + paged + upstream graphs reaches; [M] v_scale mutant.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/top-pattn/m4/pattn/window3.sh
source $HOME/titan-engine/top-pattn/m4/pattn/lib.sh
source $I/mut.sh
exec > >(tee -a $I/window3-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin pattn-window3 58
waitfix() { # TAG MIN_LEFT_S
  rm -f $I/fix.flag; date -Is > $O/$1-failed.flag; echo "!! $1 failed: waiting for $I/fix.flag ($(left)s left)"
  while [ ! -e $I/fix.flag ] && [ $(left) -gt $2 ]; do sleep 10; done
  rm -f $O/$1-failed.flag; [ -e $I/fix.flag ] && { rm -f $I/fix.flag; return 0; }; return 1
}
echo "src: mistral.rs $(git -C $SRC log --oneline -1 | cut -c1-60) $(git -C $SRC diff --quiet && echo clean || echo DIRTY); oxide $(git -C $OX log --oneline -1 | cut -c1-50) $(git -C $OX diff --quiet && echo clean || echo DIRTY)"
echo "== [B] mistral.rs build + [M] v_scale mutant meanwhile ($(el))"
( mbuild 1500 > $O/mbuild3.status 2>&1 ) &
BP=$!
mutants_b_vscale
wait $BP; cat $O/mbuild3.status
until grep -q "Finished" $O/build-$WN.log && [ $MBIN -nt $SRC/mistralrs-core/src/attention/mod.rs ]; do
  waitfix mbuild 2400 || { echo "no binary: ending the window"; exit 1; }; mbuild 1500
done
cp $MBIN $NEWBIN; BIN=$NEWBIN; sha256sum $BIN | tee $NEWBIN.sha256
echo "binary: $BIN sha256 $(sha256sum $BIN | cut -d' ' -f1) from $(git -C $SRC log --oneline -1 | cut -c1-50) $(git -C $SRC diff --quiet && echo clean || echo DIRTY); deployed: $(sha256sum $OLDBIN | cut -c1-16)"
tmaps() { timeout 300 systemd-run --user --unit=pattn-tmap --collect --wait --pipe -q -p MemoryMax=4G $HOME/ai/convert-env/bin/python3 $G/tokmap.py spark "$@"; }
SK=Spark-X2.5-4B-Q4_K_M.gguf; SP=$G/prompts/spark.json; SA="--max-model-len 16384 --max-seq-len 16384"

echo "== [S] Spark paged ON: ~8k prompts (the fix), G1, G5 ($(el))"
if PA=on srv_start s3-on $M/spark-x2.5 $SK "" $SA --pa-context-len 16384; then
  echo "VRAM after load: $(smi)"
  gate g3 s3-on-g3l 28000 | tail -1; echo "VRAM after the ~8k prompts: $(smi)"
  gate g1 s3-on-g1 $SP | tail -1
  gate g5 s3-on-g5 8000 128 | tail -1
  srv_stop
fi
for f in s3-on-g3l s3-on-g1; do cp $O/$f.json $O/$f.ids.json 2>/dev/null; done
same $O/s3-on-g1.ids.json $O/s-on-g1.ids.json; same $O/s3-on-g3l.ids.json $O/s-off-g3l.ids.json
tmaps $O/s3-on-g3l.json $O/s3-on-g1.json
cmpg g3 $O/s3-on-g3l.json $O/s-off-g3l.json
tps $O/s3-on-g3l.json $O/s-off-g3l.json $O/s3-on-g5.json $O/s-off-g5.json

echo "== [C] Spark concurrency: --max-seqs 4, 4 concurrent vs sequential ($(el))"
for pa in on off; do
  extra=""; [ $pa = on ] && extra="--pa-context-len 16384"
  if PA=$pa srv_start c3-$pa $M/spark-x2.5 $SK "" $SA --max-seqs 4 $extra; then
    echo "VRAM c3-$pa after load: $(smi)"; sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -E "Layers [0-9]|disabling PagedAttention|max_num_seqs" | sed 's/^.*INFO [a-z_:0-9]*: //' | head -3
    conc $SP $O/c3-$pa-conc.json 4 128 0
    conc $SP $O/c3-$pa-conc2.json 4 256 8
    echo "VRAM c3-$pa after: $(smi)"; srv_stop
  fi
done

echo "== [R1] REGRESSION CORE (paged attention OFF): 35B gate pair ($(el))"
coll pattn-m2 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2 && cmp40 $E/m6/out/h-off.json $E/m3/out/pattn-m2.json
coll pattn-mf0 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=0 && cmp40 $E/m6/out/mtpfile-off.json $E/m3/out/pattn-mf0.json
echo "== [R2] Bonsai-2 B2-off, IQ2_M, Bonsai-27B Q1_0 vs the deployed binary's runs ($(el))"
if srv_start b-off $M/bonsai2-27b Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf "TITAN_MTP=0 TITAN_REASONING_EFFORT=medium" --max-seq-len 65536; then
  $C $CL greedy titan $PORT b-off 8 300; srv_stop; fi
$C $CL cmp o-off b-off; $C $CL cmp dep-b-off b-off
if srv_start i-new $M/q35-lowbit Qwen3.6-35B-A3B-UD-IQ2_M.gguf "TITAN_MTP=2 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64" --max-seq-len 65536; then
  $C $CL greedy titan $PORT i-new 8 300; srv_stop; fi
$C $CL cmp dep-i-new i-new
if srv_start q-new $M/bonsai/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf "TITAN_ATTN_FLASH_PREFILL=1" --max-seq-len 32768; then
  $C $CL greedy titan $PORT q-new 8 300; srv_stop; fi
$C $CL cmp dep-q-new q-new
echo "== [R3] Spark / gemma4 / REDCELL G1 vs the deployed binary's runs ($(el))"
GA="--dtype f32 --max-model-len 11264 --max-seq-len 11264 --prefix-cache-n 0 -n 0:48"
RA="--dtype bf16 --max-model-len 8192 --max-seq-len 8192 --prefix-cache-n 0 -n 0:30"
if srv_start is $M/spark-x2.5 $SK "" $SA; then gate g1 is-g1 $SP | tail -1; srv_stop; fi
if srv_start ig $M/gemma4-12b-qat gemma-4-12b-it-qat-q4_0.gguf "" $GA; then gate g1 ig-g1 $G/prompts/gemma4-12b.json | tail -1; srv_stop; fi
if srv_start ir $M/redcell-26b REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf "" $RA; then gate g1 ir-g1 $G/prompts/redcell.json | tail -1; srv_stop; fi
for m in is ig ir; do [ -e $O/$m-g1.json ] && same $O/$m-g1.json $O/dep-$m-g1.json; done

echo "== [Q] qwen3-14b paged ON with MISTRALRS_CUDA_GRAPHS=0 ($(el))"
QP=$E/m4/devmap/prompts/qwen3-14b.json
if PA=on srv_start q3-on $M/qwen3-14b Qwen3-14B-vanilla-Q5_K_M.gguf "TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64 MISTRALRS_CUDA_GRAPHS=0" --max-seq-len 16384 --prefix-cache-n 0 --pa-context-len 16384; then
  echo "VRAM after load: $(smi)"; palines $SLOG 4
  gate g1 q3-on-g1 $QP | tail -1
  gate g3 q3-on-g3l 28000 | tail -1; echo "VRAM after the ~8k prompts: $(smi)"
  gate g5 q3-on-g5 8000 128 | tail -1
  srv_stop
fi
for f in q3-on-g1 q3-on-g3l; do cp $O/$f.json $O/$f.ids.json 2>/dev/null; done
same $O/q3-on-g1.ids.json $O/q-off-g1.ids.json
tmapq $O/q3-on-g1.json $O/q3-on-g3l.json
cmpg g1 $O/q3-on-g1.json $O/q-off-g1.json; cmpg g1 $O/q3-on-g1.json $E/m4/devmap/out/lref-g1.json
cmpg g3 $O/q3-on-g3l.json $O/q-off-g3l.json; cmpg g3 $O/q3-on-g3l.json $E/m4/devmap/out/lref-g3l.json
tps $O/q3-on-g5.json $O/q-off-g5.json $O/q3-on-g3l.json $O/q-off-g3l.json
echo "== [D] discovery: qwen3-14b paged ON + upstream CUDA graphs, TITAN_NVCC_ONLY=warn (outputs are garbage by design) ($(el))"
if PA=on srv_start q3-disc $M/qwen3-14b Qwen3-14B-vanilla-Q5_K_M.gguf "TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64 TITAN_NVCC_ONLY=warn" --max-seq-len 16384 --prefix-cache-n 0 --pa-context-len 16384; then
  PT=120 probe $O/q3-disc.json 200 16 > /dev/null 2>&1; srv_stop
fi
sed 's/\x1b\[[0-9;]*m//g' $O/q3-disc.server.log | grep "TITAN_NVCC_ONLY hit" | sort -u
free -m | head -2
echo "== pattn-window3 end $(el)"
