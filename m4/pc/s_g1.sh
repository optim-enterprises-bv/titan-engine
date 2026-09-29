# gate1 MTP=${MTPV}: warm opencode-shaped turns vs cold (no prefix cache) vs cold split at the resume point; short tool loops
source $HOME/titan-engine/m4/pc/lib.sh
T=${TAG:-m2}; MENV=${MENV:-TITAN_MTP=2}
PT="python3 $P/pc_test.py"
srv w$T 18501 "$MENV" "$PT warm 18501 out/w$T.server.log out/w$T.json"
srv c$T 18502 "$MENV TITAN_PREFIX_CACHE=0" "$PT cold 18502 out/c$T.server.log out/w$T.json out/c$T.json"
$PT cmp out/w$T.json out/c$T.json
# the resume chain of the mid-chunk case (turn 1 may itself resume at a grid point of an earlier case)
S=$(python3 -c "import json;c=[x for x in json.load(open('out/w$T.json')) if x['case']=='ptool'][0];print(','.join(str(t['resume'][0][0]) for t in c['turns'] if t['resume']))" 2>/dev/null)
echo "ptool resume chain: $S"
if [ -n "$S" ]; then
  srv s$T 18503 "$MENV TITAN_PREFIX_CACHE=0 TITAN_PREFILL_SPLIT_AT=$S" "$PT cold 18503 out/s$T.server.log out/w$T.json out/s$T.json ptool"
  $PT cmp out/w$T.json out/s$T.json
fi
srv ws$T 18504 "$MENV" "$PT short 18504 out/ws$T.server.log out/ws$T.json"
srv cs$T 18505 "$MENV TITAN_PREFIX_CACHE=0" "$PT short 18505 out/cs$T.server.log out/cs$T.json out/ws$T.json"
$PT cmp out/ws$T.json out/cs$T.json
