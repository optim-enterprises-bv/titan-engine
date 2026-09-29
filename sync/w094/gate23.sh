# gates 2+3 on the service config (35B tiered, MTP=2, 64k ctx): 3-turn prefix-cache live check (turn 2 tool
# step, turn 3 new user message) with usage.cached_tokens, then 3 x 13k back-to-back + 27.7k (no OOM). Args: BIN tag
E=$HOME/titan-engine; W=$E/sync/w094; M=$HOME/ai/models; BIN=${1:-$E/target-094-oxide/release/mistralrs}; T=${2:-g23}
L=$W/logs/$T.server.log
timeout 1200 systemd-run --user --unit=t094-$T --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=1180 bash -c "
  env LD_LIBRARY_PATH=$E/lib TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt TITAN_MTP=2 \
    $BIN serve -p 18495 --no-ui --paged-attn off --max-seq-len 65536 --format gguf -m $M/Qwen3.6-35B-A3B-MTP -f Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf > $L 2>&1 &
  pid=\$!
  for i in \$(seq 1 200); do curl -sf -m 2 -o /dev/null localhost:18495/v1/models && break; kill -0 \$pid 2>/dev/null || break; sleep 2; done
  echo '== gate 2: live 3-turn'
  python3 $W/live094.py 18495 $T-live.json
  python3 -c \"import json;r=json.load(open('$E/m4/pc/out/$T-live.json'));print([(x['prompt_tokens'],round(x['wall'],2)) for x in r])\"
  echo '== cached_tokens in usage (repeat of a 13k prompt)'
  for k in 1 2; do curl -s -m 600 localhost:18495/v1/chat/completions -H 'Content-Type: application/json' -d @$W/req13k.json | python3 -c 'import json,sys;u=json.load(sys.stdin)[\"usage\"];print(u.get(\"prompt_tokens\"),u.get(\"prompt_tokens_details\"),u.get(\"total_prompt_time_sec\"))'; done
  echo '== gate 3: 3 x 13k back-to-back + 27.7k'
  for k in 1 2 3; do python3 $E/m4/bigprompt.py 18495 13000 || echo FAIL; done
  python3 $E/m4/pc/pc_test.py big 18495 $L $W/logs/$T-big28.json $E/m4/pc/sys28k.txt 1 || echo FAIL
  kill -TERM \$pid; sleep 6; kill -KILL \$pid 2>/dev/null; true"
grep -E "panicked|ILLEGAL|out of memory|OOM|TITAN_NVCC_ONLY" $L | head -5
grep -E "Hybrid prefix cache hit" $L | sed 's/.*INFO //' | cut -c1-200
