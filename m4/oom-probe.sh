#!/bin/bash
# For each reserve: start the service config, send ~4k, ~8k, ~15k-token prompts, record ok/OOM.
E=$HOME/titan-engine; cd $E/m4
export LD_LIBRARY_PATH=$E/lib TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt TITAN_MTP=2
for r in ${RESERVES:-1024 2048 3072}; do
  TITAN_TIERED_RESERVE_MIB=$r $E/bin/mistralrs-titan serve --host 127.0.0.1 -p 18495 --paged-attn off --max-seq-len 16384 --format gguf -m $HOME/ai/models/Qwen3.6-35B-A3B-MTP -f Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf > out/oom-$r.log 2>&1 & p=$!
  for i in $(seq 1 150); do curl -sf -m 2 -o /dev/null localhost:18495/v1/models && break; sleep 3; done
  echo "reserve $r: $(grep -o 'GPU fraction [0-9.]*' out/oom-$r.log | head -1)"
  for n in 4000 8000 15000; do echo "  ~$n tokens: $(python3 bigprompt.py 18495 $n)"; grep -c "OUT_OF_MEMORY" out/oom-$r.log | sed 's/^/    OOM lines so far: /'; done
  kill $p; wait $p 2>/dev/null
done
