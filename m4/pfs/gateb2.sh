# gate (b) for the MMQ kernel: the 24-layer model's streamed prefill at GPU fraction 0.5 and 0.3 vs fraction 1
# (all resident, the same grouped MMQ without staging), 6 long + 40 short prompts (MIN_ROWS=16).
[ -f $HOME/titan-engine/m4/pfs/ctl/build.ok ] || { echo "no fresh build (ctl/build.ok): skipped"; exit 3; }
E=$HOME/titan-engine; W=$E/m4/pfs; BIN=${BIN:-$E/target-pfs-oxide/release/mistralrs}; T=${1:-b2}
coll() {
  local n=$1 pf=$2 mt=$3; shift 3
  timeout 600 systemd-run --user --unit=pfs-$n --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=580 \
    --setenv=BIN=$BIN --setenv=DIR=$E/m1/data --setenv=FILE=qwen3coder30b-first24.gguf \
    bash $E/sync/w094/collect094.sh $n $pf $mt TITAN_TIERED=1 TITAN_PFS=1 TITAN_PFS_MIN_ROWS=16 "$@" 2>&1 | grep -v " INFO "
  grep -E "titan pfs: ring|panicked|CUDA_ERROR" $E/m3/out/$n.server.log | sed 's/.*INFO //' | cut -c1-200 | sort | uniq -c | head -2
}
cmp() { python3 -c "import json,sys;a,b=json.load(open(sys.argv[1])),json.load(open(sys.argv[2]));print('GATE',sys.argv[2].split('/')[-1],'vs',sys.argv[1].split('/')[-1],':',sum(x==y for x,y in zip(a,b)),'/',len(a),'len',len(b))" $1 $2; }
O=$E/m3/out
for f in 1 0.5 0.3; do coll pfs-$T-f$f-l $W/long24.txt 64 TITAN_TIERED_GPU_FRACTION=$f; done
cmp $O/pfs-$T-f1-l.json $O/pfs-$T-f0.5-l.json; cmp $O/pfs-$T-f1-l.json $O/pfs-$T-f0.3-l.json
for f in 1 0.5; do coll pfs-$T-f$f-s $E/m4/prompts-eval.txt 64 TITAN_TIERED_GPU_FRACTION=$f; done
cmp $O/pfs-$T-f1-s.json $O/pfs-$T-f0.5-s.json
