#!/bin/bash
# nsys trace (CUDA API + kernels, no GPU counters) of the 35B serving 8 prompts x 128 tokens, for one binary.
set -u; name=$1 bin=$2; E=$HOME/titan-engine; N=/opt/nvidia/nsight-compute/2026.2.1/host/target-linux-x64/nsys
cd $E/m4/prof
export LD_LIBRARY_PATH=$E/lib:/usr/local/cuda/lib64 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt
$N profile -t cuda,osrt --sample=none --cpuctxsw=none -o $name -f true --duration=0 \
  $bin --seed 0 serve -p 18480 --no-ui --paged-attn off --max-seq-len 4096 --format gguf -m $HOME/ai/models -f Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf > $name.server.log 2>&1 &
pid=$!
for i in $(seq 1 300); do curl -sf -m 2 -o /dev/null localhost:18480/v1/models && break; sleep 2; done
python3 - <<'PY'
import json, time, urllib.request
ps = [l.strip() for l in open("prompts8.txt") if l.strip()]
t0 = time.time(); toks = 0
for p in ps:
    body = json.dumps({"model": "default", "messages": [{"role": "user", "content": p}], "temperature": 0.0, "max_tokens": 128, "seed": 0}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18480/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=900))
    toks += r["usage"]["completion_tokens"]
print(f"{toks} tokens in {time.time()-t0:.1f}s")
PY
kill -INT $pid; wait $pid 2>/dev/null
$N stats --report cuda_gpu_kern_sum --format csv -o ${name}_kern $name.nsys-rep >/dev/null 2>&1
$N stats --report cuda_api_sum --format csv -o ${name}_api $name.nsys-rep >/dev/null 2>&1
ls -la ${name}*
