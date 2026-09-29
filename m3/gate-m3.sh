#!/bin/bash
# M3 runtime gate: 50% of experts on GPU, eval prompts, output must equal the all-GPU eval run;
# compares lowest-id vs profiled static vs profiled+lru placement (tok/s and decode hit rate).
cd "$(dirname "$0")"
P=$PWD/profile-calib.txt
run() { ./collect.sh "$@" && python3 - "$1" <<'PY'
import json, sys
a, b = json.load(open("out/eval.json")), json.load(open(f"out/{sys.argv[1]}.json"))
print(f"eval vs {sys.argv[1]}: {sum(x == y for x, y in zip(a, b))}/{len(a)} identical")
PY
grep -o 'decode hit rate.*' out/$1.server.log | tail -1; grep -m1 -o 'MiB on host.*' out/$1.server.log; }
run lowid50  prompts-eval.txt 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=0.5
run prof50   prompts-eval.txt 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=0.5 TITAN_TIERED_PROFILE=$P
run proflru50 prompts-eval.txt 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=0.5 TITAN_TIERED_PROFILE=$P TITAN_TIERED_POLICY=lru
run prof25   prompts-eval.txt 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=0.25 TITAN_TIERED_PROFILE=$P
