#!/bin/bash
# calib + eval traces with everything GPU-resident (fastest, trace is routing-only),
# then the AVX2 twin at half and zero GPU fraction on the eval set (correctness vs the eval trace run).
cd "$(dirname "$0")"
./collect.sh calib prompts-calib.txt 256 TITAN_TIERED=1 TITAN_TIERED_TRACE=$PWD/out/calib.trace || exit 1
./collect.sh eval  prompts-eval.txt  256 TITAN_TIERED=1 TITAN_TIERED_TRACE=$PWD/out/eval.trace || exit 1
./collect.sh half  prompts-eval.txt  256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=0.5 || exit 1
python3 - <<'PY'
import json
a, b = json.load(open("out/eval.json")), json.load(open("out/half.json"))
print(f"eval vs half (AVX2 twin): {sum(x == y for x, y in zip(a, b))}/{len(a)} identical")
PY
