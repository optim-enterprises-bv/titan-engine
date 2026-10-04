#!/bin/bash
# pattn phase B window 2: [K] mistralrs-paged-attn-b rebuilt with short oxide names for the 864 decode entries (B1's 891
# mangled names overflowed opt's 128 KiB argument: "continuing with unoptimized IR", decode ran at half speed); PTX vs
# the committed one (names mapped back); full gate; mutants. [B] mistral.rs build. [Q] qwen3-14b paged ON, upstream CUDA
# graphs on vs off. [F] Spark FP8 KV vs bf16 KV (G1/G3/decode KL/VRAM at a fixed 16k pool/tok/s). [S] Spark paged ON
# table on the final binary. [P] REDCELL paged ON (pad_decode_input_u32 now ported). [C] Spark concurrency.
# [R] REGRESSION CORE (paged OFF vs bin/mistralrs-titan-swap's runs).
source $HOME/titan-engine/top-pattn/m4/pattn/lib.sh
source $I/mut.sh
restic_wait
exec > >(tee -a $I/window_b2-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin pattn-b2 58
echo "src: mistral.rs $(git -C $SRC log --oneline -1 | cut -c1-50) $(git -C $SRC diff --quiet && echo clean || echo DIRTY); oxide $(git -C $OX log --oneline -1 | cut -c1-50) $(git -C $OX diff --quiet && echo clean || echo DIRTY)"
c=mistralrs-paged-attn-b
echo "== [K] oxide build ($(el))"
until OXMEM=11G oxbuild $OX $c pattn-ox 1800; do waitfix ox-$c 1800 || exit 1; done
grep -c "failed to run opt" $O/oxbuild-oxide-pattn-$c.log | sed 's/^/opt skipped (count): /'
python3 $I/ptxrename.py $OX/$c/src/instances.rs $OX/$c/mistralrs_paged_attn_b.ptx $SC/b2-named.ptx
python3 $I/ptxkind.py $O/b1base-$c.ptx $SC/b2-named.ptx
(cd $OX/titan-oxide-ffi && python3 gen.py > /dev/null)
echo "== [B] mistral.rs build, background ($(el))"
( build_bin2 1500 > $O/mbuild-b2.status 2>&1 ) &
BP=$!
echo "== [K] gate ($(el))"
kgate $OX $c $O/gate-b-b2.log 2400
grep -E "family|coverage|not reachable|NOT LAUNCHED|FAIL|first diff|status diff|RACE|skipped|launcher calls" $O/gate-b-b2.log | head -30
echo "== [M] mutants ($(el))"
mutants_fp8dec; mutants_b_vscale
wait $BP; cat $O/mbuild-b2.status
BIN=$NEWBIN2; [ -x $BIN ] && [ $BIN -nt $OX/$c/mistralrs_paged_attn_b.ptx ] || { echo "no fresh binary"; exit 1; }

QM=$M/qwen3-14b; QF=Qwen3-14B-vanilla-Q5_K_M.gguf; QP=$E/m4/devmap/prompts/qwen3-14b.json; QE="TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64"
echo "== [Q] qwen3-14b paged ON, upstream CUDA decode graphs on / off ($(el))"
if PA=on srv_start q7-on $QM $QF "$QE" --max-seq-len 16384 --prefix-cache-n 0; then
  echo "VRAM after load: $(smi)"; palines $SLOG 4; sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -o "Captured .*" | head -1
  gate g1 q7-on-g1 $QP | tail -1; gate g3 q7-on-g3l 28000 | tail -1; gate g3 q7-on-g3x 48000 | tail -1; echo "VRAM after ~16k: $(smi)"
  gate g5 q7-on-g5 8000 128 | tail -1; srv_stop
  for f in q7-on-g1 q7-on-g3l q7-on-g3x; do cp $O/$f.json $O/$f.ids.json; done; tmapq $O/q7-on-g1.json $O/q7-on-g3l.json $O/q7-on-g3x.json
fi
if PA=on srv_start q7-ng $QM $QF "$QE MISTRALRS_CUDA_GRAPHS=0" --max-seq-len 16384 --prefix-cache-n 0; then
  gate g1 q7-ng-g1 $QP | tail -1; gate g5 q7-ng-g5 8000 128 | tail -1; srv_stop; fi
same $O/q7-on-g1.ids.json $O/q-off-g1.ids.json; same $O/q7-on-g1.json $O/q7-ng-g1.json
DS=$E/m4/devmap/out; QS=$O/spread-q14
cmpg g1 $O/q7-on-g1.json $DS/lref-g1.json | head -1
python3 $E/m4/g4acc/g3spread.py $DS/lref-g3l.json $DS/spread/lub256-g3l.json,$DS/spread/lub128-g3l.json,$DS/spread/lb512-g3l.json -- $O/q7-on-g3l.json $O/q-off-g3l.json
python3 $E/m4/g4acc/g3spread.py $QS/base-c48000.json $QS/lub256-c48000.json,$QS/lub128-c48000.json,$QS/lb512-c48000.json -- $O/q7-on-g3x.json $O/q5-off-g3x.json
tps $O/q7-on-g5.json $O/q7-ng-g5.json $O/q-off-g5.json $O/q7-on-g3l.json $O/q-off-g3l.json $O/q7-on-g3x.json $O/q5-off-g3x.json

SK=Spark-X2.5-4B-Q4_K_M.gguf; SP=$G/prompts/spark.json; SA="--max-model-len 16384 --max-seq-len 16384 --prefix-cache-n 0"
s() { # TAG ARGS...
  local t=$1; shift
  if PA=on srv_start $t $M/spark-x2.5 $SK "" $SA "$@"; then
    echo "VRAM after load: $(smi)"; palines $SLOG 3
    gate g1 $t-g1 $SP | tail -1; gate g3 $t-g3 | tail -1; gate g3 $t-g3l 28000 | tail -1; echo "VRAM after 8k: $(smi)"
    gate g5 $t-g5 8000 128 | tail -1; timeout 600 python3 $I/decodekl.py run $PORT $SP $O/$t-dkl.json 16 128; srv_stop
    sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -E " WARN | ERROR " | sort -u | head -3 | cut -c1-200
    for f in $t-g1 $t-g3 $t-g3l; do cp $O/$f.json $O/$f.ids.json; done; tmaps $O/$t-g1.json $O/$t-g3.json $O/$t-g3l.json
  fi
}
echo "== [S]+[F] Spark paged ON: bf16 KV (default chunks, 16384 chunks), FP8 E4M3 KV (both chunkings) ($(el))"
s s7-on
s s7-big --max-num-batched-tokens 16384
s f7-on --pa-cache-type f8e4m3
s f7-big --pa-cache-type f8e4m3 --max-num-batched-tokens 16384
for g in g1 g3 g3l; do same $O/s7-on-$g.ids.json $O/s6-on-$g.ids.json; same $O/s7-big-$g.ids.json $O/s6-off-$g.ids.json; done
SS=$O/spread-spark
python3 $E/m4/g4acc/g3spread.py $SS/base-c13500.json $SS/lub256-c13500.json,$SS/lub128-c13500.json,$SS/lb512-c13500.json -- $O/s7-on-g3.json $O/s7-big-g3.json $O/f7-on-g3.json $O/f7-big-g3.json $O/s6-off-g3.json
python3 $E/m4/g4acc/g3spread.py $SS/base-c28000.json $SS/lub256-c28000.json,$SS/lub128-c28000.json,$SS/lb512-c28000.json -- $O/s7-on-g3l.json $O/s7-big-g3l.json $O/f7-on-g3l.json $O/f7-big-g3l.json $O/s6-off-g3l.json
for t in s7-on s7-big f7-on f7-big; do cmpg g1 $O/$t-g1.json $G/out/refc-spark-g1.json | head -1; done
for p in "f7-on s7-on" "f7-big s7-big"; do set -- $p
  cmpg g1 $O/$1-g1.json $O/$2-g1.json | head -1; cmpg g3 $O/$1-g3.json $O/$2-g3.json | head -1; cmpg g3 $O/$1-g3l.json $O/$2-g3l.json | head -1
  python3 $I/decodekl.py cmp $O/$2-dkl.json $O/$1-dkl.json; done
python3 $I/decodekl.py cmp $O/s7-big-dkl.json $O/s7-on-dkl.json
tps $O/s7-on-g5.json $O/s7-big-g5.json $O/f7-on-g5.json $O/f7-big-g5.json $O/s6-off-g5.json $O/s7-on-g3l.json $O/s7-big-g3l.json $O/f7-on-g3l.json $O/s6-off-g3l.json
echo "-- VRAM at a fixed 16384-token pool, bf16 vs FP8 ($(el))"
for kv in bf16 f8e4m3; do
  ex=""; [ $kv = f8e4m3 ] && ex="--pa-cache-type f8e4m3"
  if PA=on srv_start v7-$kv $M/spark-x2.5 $SK "" $SA --pa-context-len 16384 $ex; then
    echo "$kv: VRAM after load: $(smi)"; palines $SLOG 3; gate g3 v7-$kv-g3l 28000 | tail -1; echo "$kv: VRAM after 8k: $(smi)"; srv_stop; fi
done

echo "== [P] REDCELL paged ON ($(el))"
RA="--dtype bf16 --max-model-len 8192 --max-seq-len 8192 --prefix-cache-n 0 -n 0:30"
if PA=on srv_start r7-on $M/redcell-26b REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf "TITAN_PATTN_TRACE=1" $RA --pa-context-len 8192; then
  echo "VRAM after load: $(smi)"; palines $SLOG 4
  gate g1 r7-on-g1 $G/prompts/redcell.json | tail -1; gate g3 r7-on-g3 | tail -1; gate g5 r7-on-g5 8000 128 | tail -1; srv_stop
  grep "^pattn:" $SLOG | sort | uniq -c | sort -rn | head -4 | cut -c1-220
  sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -E " WARN | ERROR |panicked|nvcc-only" | sort -u | head -4 | cut -c1-220
fi
[ -e $O/r7-on-g1.json ] && { same $O/r7-on-g1.json $O/dep-ir-g1.json; cmpg g1 $O/r7-on-g1.json $O/dep-ir-g1.json | head -1; }

echo "== [C] Spark concurrency, paged ON --max-seqs 4 --prefix-cache-n 0 ($(el))"
if PA=on srv_start c7-on $M/spark-x2.5 $SK "" --max-model-len 16384 --max-seq-len 16384 --max-seqs 4 --prefix-cache-n 0 --pa-context-len 16384; then
  conc $SP $O/c7-on-conc.json 4 128 0; conc $SP $O/c7-on-conc2.json 4 256 8; srv_stop; fi

echo "== [R1] REGRESSION CORE (paged attention OFF): 35B gate pair ($(el))"
coll pattn2-m2 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2 && cmp40 $E/m6/out/h-off.json $E/m3/out/pattn2-m2.json
coll pattn2-mf0 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=0 && cmp40 $E/m6/out/mtpfile-off.json $E/m3/out/pattn2-mf0.json
echo "== [R2] Bonsai-2 B2-off, IQ2_M, Bonsai-27B Q1_0 vs the deployed binary's runs ($(el))"
if srv_start b2-off $M/bonsai2-27b Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf "TITAN_MTP=0 TITAN_REASONING_EFFORT=medium" --max-seq-len 65536; then
  $C $CL greedy titan $PORT b2-off 8 300; srv_stop; fi
$C $CL cmp o-off b2-off; $C $CL cmp dep-b-off b2-off
if srv_start i2-new $M/q35-lowbit Qwen3.6-35B-A3B-UD-IQ2_M.gguf "TITAN_MTP=2 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64" --max-seq-len 65536; then
  $C $CL greedy titan $PORT i2-new 8 300; srv_stop; fi
$C $CL cmp dep-i-new i2-new
if srv_start q2-new $M/bonsai/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf "TITAN_ATTN_FLASH_PREFILL=1" --max-seq-len 32768; then
  $C $CL greedy titan $PORT q2-new 8 300; srv_stop; fi
$C $CL cmp dep-q-new q2-new
echo "== [R3] Spark / gemma4 / REDCELL G1 vs the deployed binary's runs ($(el))"
GA="--dtype f32 --max-model-len 11264 --max-seq-len 11264 --prefix-cache-n 0 -n 0:48"
SA0="--max-model-len 16384 --max-seq-len 16384"
if srv_start is2 $M/spark-x2.5 $SK "" $SA0; then gate g1 is2-g1 $SP | tail -1; srv_stop; fi
if srv_start ig2 $M/gemma4-12b-qat gemma-4-12b-it-qat-q4_0.gguf "" $GA; then gate g1 ig2-g1 $G/prompts/gemma4-12b.json | tail -1; srv_stop; fi
if srv_start ir2 $M/redcell-26b REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf "" $RA; then gate g1 ir2-g1 $G/prompts/redcell.json | tail -1; srv_stop; fi
for m in is ig ir; do [ -e $O/${m}2-g1.json ] && same $O/${m}2-g1.json $O/dep-$m-g1.json; done
free -m | head -2
echo "== pattn-b2 end $(el)"
