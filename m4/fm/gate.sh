#!/bin/bash
# fm identity gates: 35B 40x256 vs m3/out/q35-prof.json (MTP off) and MTP=2 vs m6/out/h-off.json, service
# env (TITAN_TIERED=1, TITAN_TIERED_GPU_FRACTION=auto, TITAN_TIERED_PROFILE, TITAN_PFS=1, TITAN_CUDA_GRAPHS=1
# reserve 64, TITAN_TIERED_RESERVE_MIB=1536). TITAN_ATTN_FLASH / TITAN_MMVQ_MOE left at their defaults (on).
# Args: TAG BIN
set -u
E=$HOME/titan-engine; M=$HOME/ai/models; F=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
T=$1; BIN=$2
PR=$E/m3/prompts-eval.txt
OUT=$E/m4/fm/out; mkdir -p $OUT
coll() {
  local n=$1 dir=$2; shift 2
  timeout 600 systemd-run --user --unit=fm-$n --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=580 \
    --setenv=BIN=$BIN --setenv=DIR=$dir --setenv=FILE=$F \
    bash $E/sync/w094/collect094.sh $n $PR 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt TITAN_PFS=1 \
      TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64 TITAN_TIERED_RESERVE_MIB=1536 "$@" 2>&1 | grep -v " INFO "
  grep -E "titan tiered|titan graphs|panicked|CUDA_ERROR" $E/m3/out/$n.server.log | sed 's/.*INFO //' | cut -c1-220 | head -4
}
cmp() {
  python3 -c "
import json,sys
a,b=json.load(open(sys.argv[1])),json.load(open(sys.argv[2]))
n=sum(x==y for x,y in zip(a,b))
print('GATE',sys.argv[2].split('/')[-1],'vs',sys.argv[1].split('/')[-1],':',n,'/',len(a),'len',len(b))
json.dump({'pass':n,'total':len(a),'lenb':len(b)}, open(sys.argv[3],'w'))
" "$1" "$2" "$3"
}
coll fm-$T-m2 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2; cmp $E/m6/out/h-off.json $E/m3/out/fm-$T-m2.json $OUT/gate-m2.json
coll fm-$T-m0 $E/sync/w094/q35root;               cmp $E/m3/out/q35-prof.json $E/m3/out/fm-$T-m0.json $OUT/gate-m0.json
true
