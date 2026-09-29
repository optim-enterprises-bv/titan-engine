# gate1 MTP=2 cold reference (rerun after a system OOM kill)
source $HOME/titan-engine/m4/pc/lib.sh
PT="python3 $P/pc_test.py"
free -m | head -3
srv cm2 18502 "TITAN_MTP=2 TITAN_PREFIX_CACHE=0" "$PT cold 18502 out/cm2.server.log out/wm2.json out/cm2.json"
$PT cmp out/wm2.json out/cm2.json
$PT cmp out/wm2.json out/sm2.json
