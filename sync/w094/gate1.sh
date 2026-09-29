# gate 1: 35B 40 x 256 vs m3/out/q35-prof.json (MTP off), MTP=2 40 x 256 vs m6/out/h-off.json. Args: BIN tag
E=$HOME/titan-engine; M=$HOME/ai/models; F=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf; BIN=${1:-$E/target-094-oxide/release/mistralrs}; T=${2:-g1}
PR=$E/m4/prompts-eval.txt
coll() {
  local n=$1 dir=$2; shift 2
  timeout 600 systemd-run --user --unit=t094-$n --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=580 \
    --setenv=BIN=$BIN --setenv=DIR=$dir --setenv=FILE=$F \
    bash $E/sync/w094/collect094.sh $n $PR 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt "$@" 2>&1 | grep -v " INFO "
  grep -o "TITAN_NVCC_ONLY hit: [a-z_0-9]*" $E/m3/out/$n.server.log | sort -u
}
cmp() { python3 -c "import json,sys;a,b=json.load(open(sys.argv[1])),json.load(open(sys.argv[2]));print('GATE',sys.argv[2].split('/')[-1],'vs',sys.argv[1].split('/')[-1],':',sum(x==y for x,y in zip(a,b)),'/',len(a),'len',len(b))" $1 $2; }
coll $T-m2 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2 TITAN_NVCC_ONLY=warn; cmp $E/m6/out/h-off.json $E/m3/out/$T-m2.json
coll $T-m0 $E/sync/w094/q35root TITAN_NVCC_ONLY=warn; cmp $E/m3/out/q35-prof.json $E/m3/out/$T-m0.json
