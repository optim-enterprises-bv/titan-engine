#!/bin/bash
cd "$(dirname "$0")"; M=$HOME/ai/models; F=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
export LD_LIBRARY_PATH=$HOME/titan-engine/lib:/usr/local/cuda/lib64
sw() { for i in $(seq 1 300); do curl -sf -m 2 -o /dev/null localhost:$1/v1/models && return 0; kill -0 $2 2>/dev/null || return 1; sleep 2; done; return 1; }
TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=0.55 $HOME/titan-engine/mistral.rs/target/release/mistralrs --seed 0 serve -p 18472 --no-ui --paged-attn off --max-seq-len 4096 --format gguf -m $M -f $F > out/det.log 2>&1 & p=$!
sw 18472 $p || exit 1
python3 client.py 18472 det1 64; python3 client.py 18472 det2 64
sed -i 's/, "top_k": 1//' client.py; python3 client.py 18472 det_notopk 64; git checkout -q client.py 2>/dev/null || sed -i 's/"temperature": 0.0,/"temperature": 0.0, "top_k": 1,/' client.py
kill $p; wait $p 2>/dev/null
python3 compare.py tiered55-run0 det1; python3 compare.py det1 det2; python3 compare.py det1 det_notopk
