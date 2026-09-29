# 80B (qwen3-next, mmap host experts, the gate-4 config) prefill speed + 8-prompt identity. Args: TAG "ENV"
[ -f $HOME/titan-engine/m4/pfs/ctl/build.ok ] || { echo "no fresh build (ctl/build.ok): skipped"; exit 3; }
E=$HOME/titan-engine; W=$E/m4/pfs; BIN=${BIN:-$E/target-pfs-oxide/release/mistralrs}; T=$1; X=$2
F80=$HOME/ai/models/qwen3-next-80b/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf
FIN="TITAN_TIERED_PROFILE=$E/m4/next/profile-next.txt TITAN_TIERED_LOOKAHEAD=2 TITAN_TIERED_LOOKAHEAD_POPULATE=1 TITAN_TIERED_PAGEOUT=1"
cd $E/m4/next
python3 -c "import os;fd=os.open('$F80',os.O_RDONLY);os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)"
BIN=$BIN timeout 900 ./pair.sh $F80 37 "pfs-$T-next8 TITAN_TIERED=1 TITAN_TIERED_MMAP=1 TITAN_TIERED_GPU_FRACTION=auto $FIN $X"
python3 -c "
import json
r=json.load(open('out/w2-8-final.json')); n_=json.load(open('out/pfs-$T-next8.json'))
print(f'GATE 80B: {sum(a==b for a,b in zip(r,n_))}/{len(r)} identical vs w2-8-final (len {len(n_)})')"
grep -E "titan pfs|panicked|CUDA_ERROR" out/pfs-$T-next8.log | sed 's/.*INFO //;s/.*WARN //' | cut -c1-250 | head -4
# prefill speed: ~4k-token prompt, cold (prefix cache off), 8k context
L=$W/logs/$T-80b.server.log; O=$W/out/$T-80b.jsonl; rm -f $O
timeout 900 systemd-run --user --unit=pfs-$T-80b --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=880 bash -c "
  env LD_LIBRARY_PATH=$E/lib TITAN_PREFIX_CACHE=0 TITAN_TIERED=1 TITAN_TIERED_MMAP=1 TITAN_TIERED_GPU_FRACTION=auto $FIN $X \
    $BIN serve -p 18498 --no-ui --paged-attn off --max-seq-len 8192 --format gguf -m $(dirname $F80) -f $(basename $F80) > $L 2>&1 &
  pid=\$!
  for i in \$(seq 1 450); do curl -sf -m 2 -o /dev/null localhost:18498/v1/models && break; kill -0 \$pid 2>/dev/null || break; sleep 2; done
  python3 $W/prompt.py 18498 $E/m4/pc/sys13k.txt 13600 $O 4k || true
  python3 $W/prompt.py 18498 $E/m4/pc/sys13k.txt 20000 $O 6k || true
  kill -TERM \$pid; sleep 5; kill -KILL \$pid 2>/dev/null; true"
grep -E "titan pfs|titan tiered auto|panicked|CUDA_ERROR|out of memory" $L | sed 's/.*INFO //;s/.*WARN //' | cut -c1-250 | head -6
