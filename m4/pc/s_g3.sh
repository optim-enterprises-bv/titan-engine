# gate3: 35B 40 x 256 vs q35-prof.json, MTP=2 40 x 256 vs h-off.json (as window4, service defaults incl. prefix cache)
source $HOME/titan-engine/m4/pc/lib.sh
PR=$E/m4/prompts-eval.txt
coll() {
  local n=$1 dir=$2; shift 2
  [ $(left) -gt 330 ] || { echo "skip $n: $(left)s left"; return 1; }
  timeout 600 systemd-run --user --unit=pc-$n --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=580 \
    --setenv=BIN=$BIN --setenv=DIR=$dir --setenv=FILE=$F \
    bash $E/m3/collect.sh $n $PR 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt "$@" 2>&1 | grep -v " INFO "
  grep -c "Hybrid prefix cache hit" $E/m3/out/$n.server.log | sed 's/^/   prefix hits: /'
}
cmp() { python3 -c "import json,sys;a,b=json.load(open(sys.argv[1])),json.load(open(sys.argv[2]));print('GATE',sys.argv[2].split('/')[-1],'vs',sys.argv[1].split('/')[-1],':',sum(x==y for x,y in zip(a,b)),'/',len(a),'len',len(b))" $1 $2; }
coll pc-m1 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2; cmp $E/m6/out/h-off.json $E/m3/out/pc-m1.json
coll pc-a1 $M; cmp $E/m3/out/q35-prof.json $E/m3/out/pc-a1.json
