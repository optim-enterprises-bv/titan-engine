# serve.sh TAG "ENV ..." CMD...: 35B service config (tiered auto, MTP=2, 64k ctx) plus ENV on port 18497 under a 20G cap;
[ -f $HOME/titan-engine/m4/pfs/ctl/build.ok ] || { echo "no fresh build (ctl/build.ok): skipped"; exit 3; }
# runs each CMD (a shell line; $P is the port, $O the out/TAG.jsonl record file) and stops the server.
E=$HOME/titan-engine; W=$E/m4/pfs; M=$HOME/ai/models; BIN=${BIN:-$E/target-pfs-oxide/release/mistralrs}
tag=$1; envs=$2; shift 2
L=$W/logs/$tag.server.log; O=$W/out/$tag.jsonl; rm -f $O
cmds=""; for c in "$@"; do cmds+="$c || echo CMD-FAILED; "; done
timeout 1500 systemd-run --user --unit=pfs-$tag --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=1480 \
  --setenv=P=18497 --setenv=O=$O --setenv=W=$W --setenv=E=$E bash -c "
  env LD_LIBRARY_PATH=$E/lib TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt TITAN_MTP=2 $envs \
    $BIN serve -p 18497 --no-ui --paged-attn off --max-seq-len 65536 --format gguf -m $M/Qwen3.6-35B-A3B-MTP -f Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf > $L 2>&1 &
  pid=\$!; t0=\$(date +%s)
  for i in \$(seq 1 300); do curl -sf -m 2 -o /dev/null localhost:18497/v1/models && break; kill -0 \$pid 2>/dev/null || break; sleep 1; done
  echo \"$tag: up after \$(( \$(date +%s) - t0 ))s ($envs)\"
  $cmds
  kill -TERM \$pid; sleep 5; kill -KILL \$pid 2>/dev/null; true"
grep -E "panicked|ILLEGAL|out of memory|OUT_OF_MEMORY|CUDA_ERROR|titan pfs: expert copy failed" $L | head -5 | cut -c1-300
grep -E "titan pfs|titan tiered auto" $L | sed 's/.*INFO //;s/.*WARN //' | cut -c1-330 | head -12
grep -E "Hybrid prefix cache hit" $L | sed 's/.*INFO //' | cut -c1-160 | head -6
