# gate4b: 13k then ~28k (service config), then deploy
source $HOME/titan-engine/m4/pc/lib.sh
srv big2 18507 "TITAN_MTP=2" "python3 $E/m4/bigprompt.py 18507 13000 || echo FAIL; python3 $P/pc_test.py big 18507 out/big2.server.log out/big28b.json $P/sys28k.txt 1 || echo FAIL; python3 $E/m4/bigprompt.py 18507 13000 || echo FAIL"
grep -E "Hybrid prefix cache" out/big2.server.log | sed 's/.*INFO //'
bash $P/s_deploy.sh
