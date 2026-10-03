#!/bin/bash
# IQ4_XS dense window 3: rebuild with the candle CPU matmul fix (lhs scratch in VecDotType blocks: the i-quants'
# 256-value blocks dot against 32-value Q8_1 blocks; window 2's auto-mapped 27B panicked at iquant.rs:349 on its CPU
# layers), OrcaSAQ-2 27B all-GPU (-n 0:64) MTP 0 / 1 + context probe, the auto map (CPU layers) as the fix's gate,
# regression core again (candle-core changed), candle cuda_iq4_xs test.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/top-iq4xs/m4/iq4xs/window3.sh
source $HOME/titan-engine/top-iq4xs/m4/iq4xs/lib.sh
exec > >(tee -a $I/window3-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin iq4xs-window3 58
echo "src: mistral.rs $(git -C $SRC log --oneline -1 | cut -c1-60) $(git -C $SRC diff --quiet && echo clean || echo DIRTY); candle $(git -C $E/top-iq4xs log --oneline -1 | cut -c1-60) $(git -C $E/top-iq4xs diff --quiet -- candle && echo clean || echo DIRTY); TITAN_* in env: $(env | grep -c '^TITAN_')"
waitfix() { rm -f $I/fix.flag; date -Is > $O/$1-failed.flag; echo "!! $1 failed: waiting for $I/fix.flag ($(left)s left)";
  while [ ! -e $I/fix.flag ] && [ $(left) -gt $2 ]; do sleep 10; done; rm -f $O/$1-failed.flag; [ -e $I/fix.flag ] && rm -f $I/fix.flag; }
echo "== [B] mistral.rs build ($(el))"
until mbuild 1500; do waitfix mbuild 1800 || exit 1; done
cp $MBIN $NEWBIN; BIN=$NEWBIN; echo "binary: $BIN sha256 $(sha256sum $BIN | cut -d' ' -f1)"

OD=$M/orcasaq2-cyber-27b; OF=OrcaSAQ-2-27B-Uncensored.gguf; GP=$I/orca-g1.json
tm() { timeout 300 systemd-run --user --unit=iq4xs-tmap --collect --wait --pipe -q -p MemoryMax=4G python3 $I/tokmap_gguf.py $OD/$OF "$@"; }
probe() { # NAME: increasing ~4k..~55k-token prompts until one fails
  for c in 16000 40000 80000 120000 160000 220000; do
    [ $(left) -gt 1200 ] || { echo "probe stopped: $(left)s left"; break; }
    timeout 600 python3 $SP/gates.py g5 $PORT titan $O/$1-ctx$c.json $c 32 > $O/$1-ctx$c.out 2>&1; rc=$?
    echo "ctx probe $c chars: rc=$rc GPU $(smi) OOM lines $(grep -c OUT_OF_MEMORY $SLOG) panics $(grep -c panicked $SLOG) $(tail -1 $O/$1-ctx$c.out | cut -c1-150)"
    [ $rc = 0 ] && g5s $O/$1-ctx$c.json || break
    systemctl --user -q is-active iq4xs-srv || { echo "server died"; break; }
  done
}
orca() { # TAG ENV ARGS...: load, VRAM, G1, speed, greedy 8x300 (MTP stats)
  local t=$1 env=$2; shift 2
  echo "== [O] $t: env '$env' args '$*' ($(el))"
  srv_start $t $OD $OF "$env" "$@" || return 1
  echo "VRAM after load: $(smi)"; sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -E "Layers |nextn|MTP|mtp" | grep -v DEBUG | head -6 | cut -c30-230
  gate g1 $t-g1 $GP; echo "VRAM after G1: $(smi)"
  gate g5 $t-g5a 2000 128; gate g5 $t-g5b 16000 128
  $C $CL greedy titan $PORT $t-g 8 300
  [ "${PROBE:-}" = 1 ] && probe $t
  srv_stop
  sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -E "titan mtp stats|accept" | tail -2 | cut -c30-260
  tm $O/$t-g1.json > /dev/null; cmpg g1 $O/$t-g1.json $O/ref-orca-g1.json | head -4
  g5s $O/$t-g5a.json $O/$t-g5b.json $O/ref-orca-g5a.json $O/ref-orca-g5b.json
}
PROBE=1 orca o64-m0 "TITAN_MTP=0" -n 0:64 --max-seq-len 65536 || orca a16-m0 "TITAN_MTP=0" --max-seq-len 16384
orca o64-m1 "TITAN_MTP=1" -n 0:64 --max-seq-len 32768 || orca a16-m1 "TITAN_MTP=1" --max-seq-len 16384
[ -e $O/o64-m0-g1.json ] && [ -e $O/o64-m1-g1.json ] && same $O/o64-m0-g1.json $O/o64-m1-g1.json
echo "== [A] fix gate: the window-2 configuration (auto map at 65536 -> layers 53-63 on CPU) G1 ($(el))"
if srv_start a64-m0 $OD $OF "TITAN_MTP=0" --max-seq-len 65536; then
  sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -E "Layers " | cut -c30-200; gate g1 a64-m0-g1 $GP; srv_stop
  echo "panics: $(grep -c panicked $SLOG)"; tm $O/a64-m0-g1.json > /dev/null; cmpg g1 $O/a64-m0-g1.json $O/ref-orca-g1.json | head -2
fi

echo "== [R1] 35B gate pair ($(el))"
coll iq4xs3-m2 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2 && cmp40 $E/m6/out/h-off.json $E/m3/out/iq4xs3-m2.json
coll iq4xs3-mf0 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=0 && cmp40 $E/m6/out/mtpfile-off.json $E/m3/out/iq4xs3-mf0.json
echo "== [R2] Bonsai-2 B2-off, IQ2_M, Bonsai-27B Q1_0, REDCELL / gemma4 G1 vs the deployed binary's runs ($(el))"
if srv_start b-off3 $M/bonsai2-27b Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf "TITAN_MTP=0 TITAN_REASONING_EFFORT=medium" --max-seq-len 65536; then
  $C $CL greedy titan $PORT b-off3 8 300; srv_stop; fi
$C $CL cmp o-off b-off3
if srv_start i-new3 $M/q35-lowbit Qwen3.6-35B-A3B-UD-IQ2_M.gguf "TITAN_MTP=2 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64" --max-seq-len 65536; then
  $C $CL greedy titan $PORT i-new3 8 300; srv_stop; fi
$C $CL cmp i-dep i-new3
if srv_start q-new3 $M/bonsai/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf "TITAN_ATTN_FLASH_PREFILL=1" --max-seq-len 32768; then
  $C $CL greedy titan $PORT q-new3 8 300; srv_stop; fi
$C $CL cmp q-dep q-new3
IO3=$E/m4/integ3/out
if srv_start rz3-g1 $M/redcell-26b REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf "" --max-model-len 8192 --max-seq-len 8192 -n 0:30 --dtype bf16 --prefix-cache-n 0; then
  gate g1 rz3-g1 $G4/prompts/redcell.json; srv_stop; fi
if srv_start g12-3-g1 $M/gemma4-12b-qat gemma-4-12b-it-qat-q4_0.gguf "" --max-model-len 8192 --max-seq-len 8192 -n 0:48 --dtype bf16; then
  gate g1 g12-3-g1 $G4/prompts/gemma4-12b.json; srv_stop; fi
tmap gemma4 $O/rz3-g1.json $O/g12-3-g1.json > /dev/null
same $O/rz3-g1.json $IO3/rz-a-g1.json; same $O/g12-3-g1.json $IO3/g12-g1.json
echo "== [E] 9B IQ4_XS e2e mistral side again (llama side from window 1) ($(el))"
if [ $(left) -gt 400 ] && srv_start e2e3-iq4xs $M/iq4xs-e2e Qwen3.5-9B-IQ4_XS.gguf "" --max-seq-len 4096; then
  cp $O/iq4xs_llama.json $O/iq4xs3_llama.json; (cd $I && timeout 600 python3 e2e.py mistral $PORT iq4xs3 > /dev/null); srv_stop
  (cd $I && python3 e2e.py cmp iq4xs3 $M/iq4xs-e2e/Qwen3.5-9B-IQ4_XS.gguf | tail -4)
fi
if [ $(left) -gt 700 ]; then
  echo "== [C] candle cuda_iq4_xs test ($(el))"
  mtest candle-iq4xs $(( $(left) - 120 )) -p candle-core -p mistralrs-quant --features mistralrs-quant/cuda --lib cuda_iq4_xs -- --nocapture
fi
free -m | head -2
echo "== iq4xs-window3 end $(el)"
# --- appended while window 3 ran: finer 27B context probe (all-GPU -n 0:64; the default probe failed at ~3.9k tokens)
fprobe() { # TAG ENV CHARS...
  local t=$1 env=$2; shift 2
  echo "== [P] $t: env '$env' ($(el))"
  srv_start $t $OD $OF "$env" -n 0:64 --max-seq-len 32768 || return 1
  for c in "$@"; do
    [ $(left) -gt 150 ] || break
    timeout 300 python3 $SP/gates.py g5 $PORT titan $O/$t-ctx$c.json $c 64 > $O/$t-ctx$c.out 2>&1; rc=$?
    echo "ctx probe $t $c chars: rc=$rc GPU $(smi) OOM lines $(grep -c OUT_OF_MEMORY $SLOG)"; [ $rc = 0 ] && g5s $O/$t-ctx$c.json
  done
  srv_stop
}
fprobe p-m0 "TITAN_MTP=0" 6000 9000 12000 14000
fprobe p-m0b "TITAN_MTP=0 TITAN_PREFILL_BIG_CHUNK=0" 9000 12000 14000 16000
fprobe p-m1 "TITAN_MTP=1 TITAN_PREFILL_BIG_CHUNK=0" 1000 2000 4000 6000
echo "== iq4xs-window3 appended end $(el)"
