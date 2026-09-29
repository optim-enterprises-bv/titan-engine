#!/bin/bash
# bench.sh NAME BIN PROMPTFILE NPROMPTS MAXTOK [ENV...] : start the 35B service config on port 18490 (optionally under
# nsys with NSYS=1), send NPROMPTS chat prompts greedy x MAXTOK, write out/NAME.json + a tok/s line. MODEL_DIR/MODEL_FILE/SEQ override.
set -u
E=$HOME/titan-engine; G=$E/m4/graph; cd $G; mkdir -p out
name=$1 bin=$2 pf=$3 np=$4 mt=$5; shift 5
N=/opt/nvidia/nsight-compute/2026.2.1/host/target-linux-x64/nsys
MD=${MODEL_DIR:-$HOME/ai/models/Qwen3.6-35B-A3B-MTP}; MF=${MODEL_FILE:-Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
PORT=18490
pre=(); [ "${NSYS:-0}" = 1 ] && pre=($N profile -t cuda,osrt --sample=none --cpuctxsw=none -o out/$name -f true)
env LD_LIBRARY_PATH=$E/lib:/usr/local/cuda/lib64 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt "$@" \
  "${pre[@]}" $bin --seed 0 serve -p $PORT --no-ui --paged-attn off --max-seq-len ${SEQ:-65536} --format gguf -m $MD -f $MF > out/$name.server.log 2>&1 &
pid=$!
for i in $(seq 1 200); do curl -sf -m 2 -o /dev/null localhost:$PORT/v1/models && break; kill -0 $pid 2>/dev/null || { echo "$name: server died"; tail -5 out/$name.server.log; exit 1; }; sleep 2; done
python3 - $name $PORT $pf $np $mt <<'PYY'
import json, sys, time, urllib.request
name, port, pf, np_, mt = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
ps = [l.strip() for l in open(pf) if l.strip()][:np_]
warm = json.dumps({"model": "default", "messages": [{"role": "user", "content": "Say hi."}], "temperature": 0.0, "max_tokens": 16}).encode()
urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", warm, {"Content-Type": "application/json"}), timeout=900).read()
res, dec, toks, t0 = [], [], 0, time.time()
for p in ps:
    body = json.dumps({"model": "default", "messages": [{"role": "user", "content": p}], "temperature": 0.0, "max_tokens": mt, "seed": 0}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=1800))
    res.append(r["choices"][0]["message"]["content"]); u = r["usage"]; toks += u["completion_tokens"]
    dec.append(u.get("avg_compl_tok_per_sec") or 0)
dt = time.time() - t0
json.dump(res, open(f"out/{name}.json", "w"), ensure_ascii=False, indent=1)
dec = sorted(dec)
print(f"{name}: {len(res)} x {mt}: {toks} tokens {dt:.1f}s wall {toks/dt:.1f} tok/s; decode tok/s median {dec[len(dec)//2]:.1f} mean {sum(dec)/len(dec):.1f}")
PYY
rc=$?
grep -o 'titan tiered auto.*' out/$name.server.log | head -1
kill -INT $pid; for i in $(seq 1 $([ "${NSYS:-0}" = 1 ] && echo 400 || echo 40)); do kill -0 $pid 2>/dev/null || break; sleep 0.5; done; kill -TERM $pid 2>/dev/null; sleep 3; kill -KILL $pid 2>/dev/null; wait $pid 2>/dev/null
exit $rc
