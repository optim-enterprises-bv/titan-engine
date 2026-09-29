# gate4: service config, 3 back-to-back ~13k (bigprompt.py, identical requests) then a ~28k prompt; OOM check
source $HOME/titan-engine/m4/pc/lib.sh
srv big 18506 "TITAN_MTP=2 TITAN_PREFILL_MEMLOG=0" "for k in 1 2 3; do python3 $E/m4/bigprompt.py 18506 13000 || echo FAIL; done; python3 $P/pc_test.py big 18506 out/big.server.log out/big28.json $P/sys28k.txt 1 || echo FAIL; python3 $P/pc_test.py big 18506 out/big.server.log out/big13.json $P/sys13k.txt 2 || echo FAIL"
grep -E "Hybrid prefix cache (hit|:)" out/big.server.log | sed 's/.*INFO //' | head -20
