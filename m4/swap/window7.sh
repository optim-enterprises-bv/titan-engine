#!/bin/bash
# Window 7: dense qwen3-14b with device_layers = ["0:40"] (all layers on GPU, no CPU offload), to
# confirm the ported GLU quantize kernels give a usable dense model end to end.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/swap/window7.sh
trap 'systemctl --user stop swap-srv 2>/dev/null; systemctl --user start titan-mistral' EXIT
trap 'exit 130' INT TERM HUP
T0=$(date +%s); DEADLINE=$((T0 + 40 * 60)); source $HOME/titan-engine/m4/swap/lib.sh
exec > >(tee -a $S/window7-$(date +%Y%m%d-%H%M).log) 2>&1
echo "== window7 start $(date -Is)"; killra; sleep 2; echo "GPU before: $(smi)"

systemctl --user stop titan-mistral
sed 's/^port = 1234/port = 18590/' $E/deploy/models.toml > $S/live-test7.toml
grep -A3 'name = "qwen3-14b"' $S/live-test7.toml | head -8
BIN=$MBIN
srv_start w7 $S/live-test7.toml || { echo "!! server did not come up"; exit 1; }
echo "GPU after server start: $(smi)"

echo "== (A) the device map that was chosen ($(el))"
grep -E "Layers .* cuda|Layers .* cpu" $SLOG | tail -4

echo "== (B) dense qwen3-14b, all layers on GPU ($(el))"
O=$S/out/w7-dense.jsonl; rm -f $O
timeout 1200 python3 seq.py $PORT qwen3-14b $O dense 13k 23k 28k 36k
echo "-- dense done: GPU $(smi)"
echo "== abort markers (want 0) =="
grep -cE "nvcc-only|SIGABRT|panicked" $SLOG || echo "0 (clean)"

echo "== (C) MoE control ($(el))"
timeout 600 python3 seq.py $PORT qwen3.6-35b $O moe-ref 13k
srv_stop
echo "== summary"
cat $O 2>/dev/null
echo "== window7 end $(el)"
