# window 2: rebuild (boundary points kept), gate1 MTP=2 (+loop case), MTP off warm/cold, gate3, gate4
source $HOME/titan-engine/m4/pc/lib.sh
build || exit 1
PT="python3 $P/pc_test.py"
srv w2m2 18501 "TITAN_MTP=2" "$PT warm 18501 out/w2m2.server.log out/w2m2.json"
srv c2m2 18502 "TITAN_MTP=2 TITAN_PREFIX_CACHE=0" "$PT cold 18502 out/c2m2.server.log out/w2m2.json out/c2m2.json nothink think loop"
$PT cmp out/w2m2.json out/c2m2.json | tee out/w2-gate1.txt
srv w2m0 18503 "TITAN_MTP=0" "$PT warm 18503 out/w2m0.server.log out/w2m0.json"
srv c2m0 18504 "TITAN_MTP=0 TITAN_PREFIX_CACHE=0" "$PT cold 18504 out/c2m0.server.log out/w2m0.json out/c2m0.json nothink think loop"
$PT cmp out/w2m0.json out/c2m0.json | tee -a out/w2-gate1.txt
bash $P/s_g3.sh
srv big3 18507 "TITAN_MTP=2" "for k in 1 2 3; do python3 $E/m4/bigprompt.py 18507 13000 || echo FAIL; done; python3 $P/pc_test.py big 18507 out/big3.server.log out/big28c.json $P/sys28k.txt 1 || echo FAIL"
grep -E "Hybrid prefix cache" out/big3.server.log | sed 's/.*INFO //'
echo '== w2 tests done'
