#!/bin/bash
# Window 6: rebuild the swap/service binary with the ported typed GLU quantize kernels
# (launch_mmq_quantize_glu_q8_1_{D4,DS4,D2S6}, bit-identical to the C++ per the gate), then
# test the DENSE model (qwen3-14b Q5_K_M) that used to SIGABRT on prefill via the missing kernel.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/swap/window6.sh
trap 'systemctl --user stop swap-srv 2>/dev/null; systemctl --user start titan-mistral' EXIT
trap 'exit 130' INT TERM HUP
T0=$(date +%s); DEADLINE=$((T0 + 55 * 60)); source $HOME/titan-engine/m4/swap/lib.sh
exec > >(tee -a $S/window6-$(date +%Y%m%d-%H%M).log) 2>&1
echo "== window6 start $(date -Is)"; killra; sleep 2; echo "GPU before: $(smi)"

systemctl --user stop titan-mistral
echo "== (A) build the service binary with the ported kernels ($(el))"
mbuild || { echo "!! build failed ($(el))"; exit 1; }
ls -la $MBIN

echo "== (B) start the swap server on the live roster ($(el))"
sed 's/^port = 1234/port = 18590/' $E/deploy/models.toml > $S/live-test.toml
BIN=${MBIN}
srv_start w6 $S/live-test.toml || { echo "!! server did not come up"; exit 1; }
echo "GPU after load: $(smi)"

echo "== (C) THE REGRESSION: dense qwen3-14b (Q5_K_M) prefill - used to abort() here ($(el))"
O=$S/out/w6-dense.jsonl; rm -f $O
timeout 900 python3 seq.py $PORT qwen3-14b $O dense 13k 23k 28k
echo "-- dense done: GPU $(smi)"
echo "== abort / nvcc-only markers in the server log (want NONE) =="
grep -cE "nvcc-only|SIGABRT|panicked|abort" $SLOG || echo "0 (clean)"
grep -oE "titan oxide build: CUDA launcher .* is nvcc-only.*" $SLOG | head -3

echo "== (D) control: the MoE default still works ($(el))"
timeout 600 python3 seq.py $PORT qwen3.6-35b $O moe-ref 13k 23k
echo "GPU: $(smi)"

echo "== summary"
python3 -c "
import json
for l in open('$O'):
    r=json.loads(l); print(r['tag'], r['size'], 'ok=' + str(r['ok']), r.get('prompt_tokens'), round(r.get('prompt_tok_s') or 0), round(r['wall'],1))" 2>/dev/null
srv_stop
echo "== window6 end $(el)"
