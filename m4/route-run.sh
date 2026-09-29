#!/bin/bash
E=$HOME/titan-engine; cd $E/m4
export LD_LIBRARY_PATH=$E/lib TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt TITAN_TIERED_TRACE=$E/m4/out/route-35b.trace
$E/target-oxide/release/mistralrs --seed 0 serve -p 18491 --no-ui --paged-attn off --max-seq-len 4096 --format gguf -m $HOME/ai/models -f Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf > out/route.server.log 2>&1 & pid=$!
for i in $(seq 1 300); do curl -sf -m 2 -o /dev/null localhost:18491/v1/models && break; sleep 2; done
curl -s localhost:18491/v1/completions -H 'Content-Type: application/json' -d '{"model":"default","prompt":"The capital of France is","max_tokens":1,"temperature":0}' > /dev/null
kill $pid; wait $pid 2>/dev/null
