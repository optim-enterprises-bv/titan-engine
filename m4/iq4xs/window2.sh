#!/bin/bash
# IQ4_XS dense window 2: OrcaSAQ-2 27B (qwen35 dense, IQ4_XS 439 tensors): llama.cpp reference (G1, speed),
# titan TITAN_MTP=0 and 1: VRAM after load, G1 vs llama.cpp, decode / prefill tok/s, context probe, MTP acceptance.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/top-iq4xs/m4/iq4xs/window2.sh [stages]
source $HOME/titan-engine/top-iq4xs/m4/iq4xs/lib.sh
STAGES=${1:-"neg llama t0 t1 reg ctest"}
exec > >(tee -a $I/window2-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin iq4xs-window2 58
BIN=$NEWBIN; echo "binary: $BIN sha256 $(sha256sum $BIN | cut -d' ' -f1); stages: $STAGES; TITAN_* in env: $(env | grep -c '^TITAN_')"
OD=$M/orcasaq2-cyber-27b; OF=OrcaSAQ-2-27B-Uncensored.gguf; GP=$I/orca-g1.json
tm() { timeout 300 systemd-run --user --unit=iq4xs-tmap --collect --wait --pipe -q -p MemoryMax=4G python3 $I/tokmap_gguf.py $OD/$OF "$@"; }
lgate() { timeout 1200 python3 $SP/gates.py "$1" $LP llama $O/$2.json "${@:3}"; }

if [[ " $STAGES " == *" neg "* ]]; then
  echo "== [N] negative control: the deployed binary on the 9B IQ4_XS file (dense IQ4_XS must fail there) ($(el))"
  BIN=$OLDBIN
  if srv_start neg-dep $M/iq4xs-e2e Qwen3.5-9B-IQ4_XS.gguf "" --max-seq-len 4096; then
    curl -s -m 120 localhost:$PORT/v1/completions -H 'Content-Type: application/json' -d '{"model":"default","prompt":"The capital of France is","max_tokens":8,"temperature":0}' | cut -c1-300; echo
    srv_stop
  fi
  grep -E "unsupported dtype|panicked|Error" $SLOG | head -3 | cut -c1-250
  BIN=$NEWBIN
fi

if [[ " $STAGES " == *" llama "* ]]; then
  echo "== [L] llama.cpp reference, -ngl 99 -c 8192, F16 KV ($(el))"
  KV="f16"
  for kvargs in "" "-fa on -ctk q8_0 -ctv q8_0"; do
    systemctl --user reset-failed iq4xs-llama 2>/dev/null
    systemd-run --user --unit=iq4xs-llama --collect -q -p MemoryMax=20G -p MemorySwapMax=0 -p StandardOutput=truncate:$O/orca_llama.log -p StandardError=truncate:$O/orca_llama.log \
      $HOME/ai/llama.cpp/build/bin/llama-server -m $OD/$OF -ngl 99 -c 8192 --port $LP --temp 0 --top-k 1 -np 1 $kvargs
    up=0; for i in $(seq 1 200); do curl -sf -m 2 -o /dev/null localhost:$LP/health && { up=1; break; }; systemctl --user -q is-active iq4xs-llama || break; sleep 2; done
    [ $up = 1 ] && { KV="${kvargs:-f16}"; break; }
    echo "llama.cpp did not come up with KV args '${kvargs:-f16}':"; grep -iE "error|fail|out of memory" $O/orca_llama.log | tail -5
    systemctl --user stop iq4xs-llama; sleep 2
  done
  echo "llama.cpp up with KV: $KV; GPU $(smi)"; grep -E "offloaded|CUDA0 model buffer|KV buffer|compute buffer" $O/orca_llama.log | head -6
  lgate g1 ref-orca-g1 $GP
  lgate g5 ref-orca-g5a 2000 128; lgate g5 ref-orca-g5b 16000 128
  systemctl --user stop iq4xs-llama; sleep 2
  g5s $O/ref-orca-g5a.json $O/ref-orca-g5b.json
fi

probe() { # NAME: increasing ~4k..~56k-token prompts until one fails
  for c in 16000 40000 80000 120000 160000 220000; do
    [ $(left) -gt 300 ] || { echo "probe stopped: $(left)s left"; break; }
    timeout 600 python3 $SP/gates.py g5 $PORT titan $O/$1-ctx$c.json $c 32 > $O/$1-ctx$c.out 2>&1; rc=$?
    echo "ctx probe $c chars: rc=$rc GPU $(smi) OOM lines $(grep -c OUT_OF_MEMORY $SLOG) $(tail -1 $O/$1-ctx$c.out | cut -c1-150)"
    g5s $O/$1-ctx$c.json
    [ $rc = 0 ] && grep -q prompt_tokens $O/$1-ctx$c.json 2>/dev/null || break
    systemctl --user -q is-active iq4xs-srv || { echo "server died"; break; }
  done
}
for mtp in 0 1; do
  [[ " $STAGES " == *" t$mtp "* ]] || continue
  echo "== [T$mtp] titan TITAN_MTP=$mtp, --max-seq-len 65536 ($(el))"
  if srv_start orca-m$mtp $OD $OF "TITAN_MTP=$mtp" --max-seq-len 65536; then
    echo "VRAM after load: $(smi)"; grep -E "titan|device map|layers|GPU|CPU" $SLOG | grep -vE " DEBUG |INFO.*Request" | head -12 | cut -c1-220
    gate g1 orca-m$mtp-g1 $GP
    gate g5 orca-m$mtp-g5a 2000 128; gate g5 orca-m$mtp-g5b 16000 128
    $C $CL greedy titan $PORT orca-m$mtp-g 8 300
    [ $mtp = 0 ] && probe orca-m0
    srv_stop
    grep -E "titan mtp stats|accept" $SLOG | tail -3 | cut -c1-250
    tm $O/orca-m$mtp-g1.json > /dev/null
    [ -e $O/ref-orca-g1.json ] && cmpg g1 $O/orca-m$mtp-g1.json $O/ref-orca-g1.json | head -6
    g5s $O/orca-m$mtp-g5a.json $O/orca-m$mtp-g5b.json
  fi
done
[ -e $O/orca-m0-g1.json ] && [ -e $O/orca-m1-g1.json ] && same $O/orca-m0-g1.json $O/orca-m1-g1.json
if [[ " $STAGES " == *" reg "* ]]; then
  echo "== [RG] REDCELL (pc 0) and gemma4-12b G1 vs the deployed binary's recorded runs (m4/integ3 rz-a-g1, g12-g1) ($(el))"
  IO3=$E/m4/integ3/out
  if srv_start rz-g1 $M/redcell-26b REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf "" --max-model-len 8192 --max-seq-len 8192 -n 0:30 --dtype bf16 --prefix-cache-n 0; then
    gate g1 rz-g1 $G4/prompts/redcell.json; srv_stop
  fi
  if srv_start g12-g1 $M/gemma4-12b-qat gemma-4-12b-it-qat-q4_0.gguf "" --max-model-len 8192 --max-seq-len 8192 -n 0:48 --dtype bf16; then
    gate g1 g12-g1 $G4/prompts/gemma4-12b.json; srv_stop
  fi
  tmap gemma4 $O/rz-g1.json $O/g12-g1.json > /dev/null
  same $O/rz-g1.json $IO3/rz-a-g1.json; same $O/g12-g1.json $IO3/g12-g1.json
fi
if [[ " $STAGES " == *" ctest "* ]] && [ $(left) -gt 900 ]; then
  echo "== [C] candle cuda_iq4_xs test (CPU dequant vs llama.cpp oracle, GPU dequant f32/f16, mmvq / dequant+matmul) ($(el))"
  mtest candle-iq4xs 840 -p candle-core --features cuda --lib -- quantized::cuda::test::cuda_iq4_xs --nocapture
fi
free -m | head -2
echo "== iq4xs-window2 end $(el)"
