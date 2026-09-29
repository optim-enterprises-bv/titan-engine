# titan-094 smoke on the service config (35B tiered + MTP=2) with TITAN_NVCC_ONLY=warn: lists
# every nvcc-only launcher the model path reaches. Args: BIN tag
E=$HOME/titan-engine; BIN=${1:-$E/target-094-oxide/release/mistralrs}; tag=${2:-smoke}
L=$E/sync/w094/logs/$tag.server.log; M=$HOME/ai/models
timeout 700 systemd-run --user --unit=t094-$tag --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=680 bash -c "
  env LD_LIBRARY_PATH=$E/lib TITAN_NVCC_ONLY=warn TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt TITAN_MTP=2 \
    $BIN serve -p 18494 --no-ui --paged-attn off --max-seq-len 65536 --format gguf -m $M/Qwen3.6-35B-A3B-MTP -f Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf > $L 2>&1 &
  pid=\$!
  for i in \$(seq 1 200); do curl -sf -m 2 -o /dev/null localhost:18494/v1/models && break; kill -0 \$pid 2>/dev/null || break; sleep 2; done
  echo loaded after \$((i*2))s
  for t in 0 0.7; do
    curl -s -m 300 localhost:18494/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{\"model\":\"default\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a haiku about GPUs.\"}],\"max_tokens\":64,\"temperature\":'\$t'}' | head -c 1200; echo
  done
  python3 $E/m4/bigprompt.py 18494 4000
  kill -TERM \$pid; sleep 5; kill -KILL \$pid 2>/dev/null; true"
echo "-- nvcc-only launchers reached:"; grep -o "TITAN_NVCC_ONLY hit: [a-z_0-9A-Z]*" $L | sort -u
grep -m3 -i "panicked\|error\|abort" $L | cut -c1-300
grep -m2 "titan tiered experts\|titan fork-local loader" $L | cut -c1-200
