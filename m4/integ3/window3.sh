#!/bin/bash
# integ3-20261003 window 3: [A] dry-run decode A/B: qwen3-14b and gpt-oss-120b (18.5 / 5.0 tok/s on the first, cold
# request in window 2), two requests each, integ3 vs integ2, same roster-dry.toml; [X] bisect: REDCELL G3 at pc 0 differs
# from redcell2's dz-a on the 3547 / 3532-token prompts (G1 and G2-with-flash-decode-off identical): build integ3 with
# the g4acc merge reverted (branch integ3-bisect-nog4acc, worktree top-integ3/mr-bis) and run the dz procedure on it.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/top-integ3/m4/integ3/window3.sh
source $HOME/titan-engine/top-integ3/m4/integ3/lib.sh
exec > >(tee -a $I/window3-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin integ3-window3 50
RED=REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf; RP=$G4/prompts/redcell.json; RA=(--max-model-len 8192 --max-seq-len 8192 -n 0:30 --dtype bf16)
RC2=$E/top-integ3/m4/redcell2/out

echo "== [A] decode A/B on the deploy roster ($(el))"
for t in new old; do
  [ $t = old ] && BIN=$OLDBIN || BIN=$NEWBIN
  if CWD=$HOME cfg_start ab-$t $I/roster-dry.toml; then
    MODELS=qwen3-14b,gpt-oss-120b REPEAT=2 timeout 900 python3 $I/dryrun.py $PORT $I/roster-dry.toml $SLOG $O/ab-$t.json | grep DRY
    srv_stop
  fi
done
BIN=$NEWBIN

echo "== [X] bisect build: integ3 minus the g4acc merge ($(el))"
SRC=$E/top-integ3/mr-bis; TGT=$E/target-integ3bis-oxide; MBIN=$TGT/release/mistralrs
echo "src: $(git -C $SRC log --oneline -1 | cut -c1-80) $(git -C $SRC diff --quiet && echo clean || echo DIRTY)"
[ -d $TGT ] || cp -a --reflink=auto $E/target-integ3-oxide $TGT
if mbuild 1500; then
  cp $MBIN $O/mistralrs-bis; BIN=$O/mistralrs-bis; echo "bisect binary sha256 $(sha256sum $BIN | cut -c1-16)"
  if srv_start bz $M/redcell-26b $RED "" "${RA[@]}" --prefix-cache-n 0; then
    gate g1 bz-g1 $RP; gate g3 bz-g3 2>&1 | tail -1; gate g2 bz-g2 $RP 256 2>&1 | tail -1; srv_stop
  fi
  python3 - $O $RC2 <<'EOF'
import json, sys
O, R = sys.argv[1], sys.argv[2]
lp = lambda r: [[v for _, v in s] for s in r["top"]]
for g in ("g1", "g3"):
    for t, f in (("bisect (no g4acc)", f"{O}/bz-{g}.json"), ("integ3", f"{O}/rz-a-{g}.json")):
        a, b = json.load(open(f)), json.load(open(f"{R}/dz-a-{g}.json"))
        print(f"BISECT {g} {t} vs redcell2 dz-a: top-20 logprob values identical {sum(lp(x) == lp(y) for x, y in zip(a, b))}/{len(a)}")
a, b = json.load(open(f"{O}/bz-g2.json")), json.load(open(f"{R}/dz-a-g2.json"))
print(f"BISECT g2 bisect vs redcell2 dz-a: tokens identical {sum(x['tokens'] == y['tokens'] for x, y in zip(a, b))}/{len(a)}")
EOF
fi
free -m | head -2
echo "== integ3-window3 end $(el)"
