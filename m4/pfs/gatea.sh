# gate (a)+(f): 35B 40 x 256 vs m3/out/q35-prof.json (MTP off) and MTP=2 vs m6/out/h-off.json, with ENV (e.g. TITAN_PFS=1);
# prints decode tok/s. Args: TAG "ENV" [m0]
[ -f $HOME/titan-engine/m4/pfs/ctl/build.ok ] || { echo "no fresh build (ctl/build.ok): skipped"; exit 3; }
E=$HOME/titan-engine; M=$HOME/ai/models; F=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf; BIN=${BIN:-$E/target-pfs-oxide/release/mistralrs}; T=$1; X=$2
PR=$E/m4/prompts-eval.txt
coll() {
  local n=$1 dir=$2; shift 2
  timeout 600 systemd-run --user --unit=pfs-$n --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=580 \
    --setenv=BIN=$BIN --setenv=DIR=$dir --setenv=FILE=$F \
    bash $E/sync/w094/collect094.sh $n $PR 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt "$@" 2>&1 | grep -v " INFO "
  grep -E "titan pfs: pass|panicked|CUDA_ERROR" $E/m3/out/$n.server.log | sed 's/.*INFO //' | cut -c1-200 | head -2
}
cmp() { python3 -c "import json,sys;a,b=json.load(open(sys.argv[1])),json.load(open(sys.argv[2]));print('GATE',sys.argv[2].split('/')[-1],'vs',sys.argv[1].split('/')[-1],':',sum(x==y for x,y in zip(a,b)),'/',len(a),'len',len(b))" $1 $2; }
coll pfs-$T-m2 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2 $X; cmp $E/m6/out/h-off.json $E/m3/out/pfs-$T-m2.json
[ "${3:-}" = m0 ] && { coll pfs-$T-m0 $E/sync/w094/q35root $X; cmp $E/m3/out/q35-prof.json $E/m3/out/pfs-$T-m0.json; }
true
