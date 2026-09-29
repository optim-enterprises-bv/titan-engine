# serve.sh TAG "ENV ..." CMD...: 35B service config (tiered auto, MTP=2, PFS, graphs, 64k ctx) plus ENV on port 18497
# under a 20G cap; runs each CMD (a shell line; $P is the port, $O the out/TAG.jsonl record file, $SESS the nsys
# session when NSYS=1) and stops the server. BIN (default: the deployed binary) picks the binary.
E=$HOME/titan-engine; W=$E/m4/fa; M=$HOME/ai/models; BIN=${BIN:-$E/bin/mistralrs-titan-kv}
NSYSB=/opt/nvidia/nsight-compute/2026.2.1/host/target-linux-x64/nsys
tag=$1; envs=$2; shift 2
L=$W/logs/$tag.server.log; O=$W/out/$tag.jsonl; rm -f $O
pre=""; SESS=fa-$tag-$$
[ "${NSYS:-0}" = 1 ] && pre="$NSYSB launch --session-new=$SESS -t cuda --cuda-graph-trace=node"
cmds=""; for c in "$@"; do cmds+="$c || echo CMD-FAILED; "; done
timeout 1500 systemd-run --user --unit=fa-$tag --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=1480 \
  --setenv=P=18497 --setenv=O=$O --setenv=W=$W --setenv=E=$E --setenv=SESS=$SESS --setenv=NSYSB=$NSYSB bash -c "
  cd $W/out; env LD_LIBRARY_PATH=$E/lib TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt TITAN_MTP=2 \
    TITAN_PFS=1 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64 TITAN_TIERED_RESERVE_MIB=1536 $envs \
    $pre $BIN serve -p 18497 --no-ui --paged-attn off --max-seq-len 65536 --format gguf -m $M/Qwen3.6-35B-A3B-MTP -f Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf > $L 2>&1 &
  pid=\$!; t0=\$(date +%s)
  for i in \$(seq 1 300); do curl -sf -m 2 -o /dev/null localhost:18497/v1/models && break; kill -0 \$pid 2>/dev/null || break; sleep 1; done
  echo \"$tag: up after \$(( \$(date +%s) - t0 ))s ($envs)\"
  $cmds
  kill -TERM \$pid; for i in \$(seq 1 30); do kill -0 \$pid 2>/dev/null || break; sleep 0.5; done; kill -KILL \$pid 2>/dev/null; true"
grep -E "panicked|ILLEGAL|out of memory|OUT_OF_MEMORY|CUDA_ERROR|error:" $L | head -5 | cut -c1-300
grep -E "titan attention|titan tiered auto" $L | sed 's/.*INFO //;s/.*WARN //' | cut -c1-200 | head -6
grep -o "titan mtp stats.*" $L | tail -1 | cut -c1-200
true
