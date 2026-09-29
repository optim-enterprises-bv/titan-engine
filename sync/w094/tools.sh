# chat + tool calls + Anthropic Messages on the service config. Args: BIN tag
E=$HOME/titan-engine; W=$E/sync/w094; M=$HOME/ai/models; BIN=${1:-$E/target-094-oxide/release/mistralrs}; T=${2:-tools}; L=$W/logs/$T.server.log
timeout 600 systemd-run --user --unit=t094-$T --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=580 bash -c "
  env LD_LIBRARY_PATH=$E/lib TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt TITAN_MTP=2 \
    $BIN serve -p 18496 --no-ui --paged-attn off --max-seq-len 65536 --format gguf -m $M/Qwen3.6-35B-A3B-MTP -f Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf > $L 2>&1 &
  pid=\$!
  for i in \$(seq 1 200); do curl -sf -m 2 -o /dev/null localhost:18496/v1/models && break; kill -0 \$pid 2>/dev/null || break; sleep 2; done
  python3 $W/tools.py 18496
  kill -TERM \$pid; sleep 6; kill -KILL \$pid 2>/dev/null; true"
grep -E "panicked|ILLEGAL|TITAN_NVCC_ONLY|nvcc-only" $L | head -5
