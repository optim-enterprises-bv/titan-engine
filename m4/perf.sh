#!/bin/bash
# M4 perf: routing trace on calib prompts -> profile -> eval with lowest-id vs profiled placement.
cd "$(dirname "$0")"; E=$HOME/titan-engine
export DIR=$HOME/ai/models FILE=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
C=$E/m3/collect.sh; FR=${FR:-0.55}
cp $E/m3/prompts-calib.txt $E/m3/prompts-eval.txt .
cd $E/m3
rm -f $E/m4/out/q35-calib.trace
$C q35-calib $E/m4/prompts-calib.txt 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=$FR TITAN_TIERED_TRACE=$E/m4/out/q35-calib.trace || exit 1
python3 profile.py $E/m4/out/q35-calib.trace > $E/m4/profile-q35.txt
$C q35-lowid $E/m4/prompts-eval.txt 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=$FR || exit 1
$C q35-prof $E/m4/prompts-eval.txt 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=$FR TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt || exit 1
python3 - <<'PY'
import json
a, b = json.load(open("out/q35-lowid.json")), json.load(open("out/q35-prof.json"))
print(f"q35 lowid vs prof: {sum(x == y for x, y in zip(a, b))}/{len(a)} identical")
PY
for n in q35-lowid q35-prof; do grep -o 'decode hit rate.*' out/$n.server.log | tail -1; done
