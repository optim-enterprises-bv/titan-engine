# gate (f): decode tok/s A/B, MTP=2 40 x 256 (the gate-a harness), old / new alternating. Arg: tag
W=$HOME/titan-engine/m4/pfs; T=${1:-ab}
for r in 1 2; do bash $W/gatea.sh $T-old$r ""; bash $W/gatea.sh $T-new$r "TITAN_PFS=1"; done 2>&1 | grep -E "completions|GATE"
