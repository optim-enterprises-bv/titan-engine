# gate (b): the 24-layer Qwen3-Coder-30B test model (fits entirely): streaming prefill at GPU fraction 0.5 vs the
[ -f $HOME/titan-engine/m4/pfs/ctl/build.ok ] || { echo "no fresh build (ctl/build.ok): skipped"; exit 3; }
# all-GPU run (fraction 1), 40 short prompts (TITAN_PFS_MIN_ROWS=16 so their prompt passes stream) + 6 long prompts.
E=$HOME/titan-engine; W=$E/m4/pfs; BIN=${BIN:-$E/target-pfs-oxide/release/mistralrs}; T=${1:-b}
coll() {
  local n=$1 pf=$2 mt=$3; shift 3
  timeout 600 systemd-run --user --unit=pfs-$n --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=580 \
    --setenv=BIN=$BIN --setenv=DIR=$E/m1/data --setenv=FILE=qwen3coder30b-first24.gguf \
    bash $E/sync/w094/collect094.sh $n $pf $mt TITAN_TIERED=1 "$@" 2>&1 | grep -v " INFO "
  grep -E "titan pfs: pass|titan pfs: ring|panicked|CUDA_ERROR" $E/m3/out/$n.server.log | sed 's/.*INFO //' | cut -c1-260 | sort | uniq -c | sort -rn | head -4
}
cmp() { python3 -c "import json,sys;a,b=json.load(open(sys.argv[1])),json.load(open(sys.argv[2]));print('GATE',sys.argv[2].split('/')[-1],'vs',sys.argv[1].split('/')[-1],':',sum(x==y for x,y in zip(a,b)),'/',len(a),'len',len(b))" $1 $2; }
O=$E/m3/out
coll pfs-$T-ref-l $W/long24.txt 64 TITAN_TIERED_GPU_FRACTION=1
coll pfs-$T-str-l $W/long24.txt 64 TITAN_TIERED_GPU_FRACTION=0.5 TITAN_PFS=1 TITAN_PFS_TIMING=1
cmp $O/pfs-$T-ref-l.json $O/pfs-$T-str-l.json
coll pfs-$T-str-s $E/m4/prompts-eval.txt 64 TITAN_TIERED_GPU_FRACTION=0.5 TITAN_PFS=1 TITAN_PFS_MIN_ROWS=16
coll pfs-$T-ref-s $E/m4/prompts-eval.txt 64 TITAN_TIERED_GPU_FRACTION=1
cmp $O/pfs-$T-ref-s.json $O/pfs-$T-str-s.json
[ "${2:-}" = full ] || exit 0
coll pfs-$T-cpu-l $W/long24.txt 64 TITAN_TIERED_GPU_FRACTION=0.5
cmp $O/pfs-$T-ref-l.json $O/pfs-$T-cpu-l.json
coll pfs-$T-sh-l $W/long24.txt 64 TITAN_TIERED_GPU_FRACTION=0.5 TITAN_PFS=1 TITAN_PFS_SHARE=0.6 TITAN_PFS_RING_MIB=150
cmp $O/pfs-$T-ref-l.json $O/pfs-$T-sh-l.json
coll pfs-$T-mmq-l $W/long24.txt 64 TITAN_TIERED_GPU_FRACTION=0.5 TITAN_PFS=1 TITAN_PFS_KERNEL=mmq
cmp $O/pfs-$T-ref-l.json $O/pfs-$T-mmq-l.json
