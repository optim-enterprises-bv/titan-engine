#!/bin/bash
# M1 gate: stock vs TITAN_TIERED=1 vs TITAN_TIERED=1 + TITAN_TIERED_PROBE=1 (permuted slots),
# same binary, same 24-layer Qwen3-Coder-30B GGUF, greedy. Pass = all three byte-identical.
# Runs as a transient systemd unit whose ExecStopPost restores titan's alive.
set -u
cd "$(dirname "$0")"
BIN=${BIN:-$HOME/titan-engine/mistral.rs/target/release/mistralrs}
export LD_LIBRARY_PATH=$HOME/titan-engine/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
DIR=$HOME/titan-engine/m1/data
FILE=qwen3coder30b-first24.gguf
PORT=18461
MAX_TOKENS=${MAX_TOKENS:-256}
OUT=out
mkdir -p $OUT


run_mode() {  # name, env...
    local name=$1; shift
    echo "=== $name: $*"
    env "$@" "$BIN" --seed 0 serve -p $PORT --no-ui --paged-attn off --max-seq-len 2048 \
        --format gguf -m "$DIR" -f "$FILE" > $OUT/$name.server.log 2>&1 &
    local pid=$!
    for i in $(seq 1 180); do
        curl -s -m 2 -o /dev/null localhost:$PORT/v1/models && break
        kill -0 $pid 2>/dev/null || { echo "server died"; tail -20 $OUT/$name.server.log; return 1; }
        sleep 2
    done
    python3 - "$name" "$PORT" "$MAX_TOKENS" <<'EOF'
import json, sys, time, urllib.request
name, port, max_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])
prompts = [
    "Write a Rust function that reverses a linked list.", "Explain what a mixture-of-experts layer does.",
    "List five prime numbers greater than 100.", "What is the capital of Vietnam?",
    "Translate 'good morning' into French, German and Thai.", "Write a haiku about GPUs.",
    "Summarize the plot of Hamlet in two sentences.", "def fib(n):", "SELECT * FROM users WHERE",
    "Why is the sky blue?", "Give a bash one-liner to count lines in all .rs files.",
    "Explain TCP slow start.", "Write a JSON object describing a cat.", "1, 1, 2, 3, 5, 8,",
    "What does `git rebase -i` do?", "Describe the Rust borrow checker to a Python programmer.",
    "Name three uses of WireGuard.", "Write a limerick about Linux kernels.",
    "How do I set up a systemd user timer?", "Continue: The quick brown fox",
]
res, t0, toks = [], time.time(), 0
for p in prompts:
    body = json.dumps({"model": "default", "messages": [{"role": "user", "content": p}],
                       "temperature": 0.0, "max_tokens": max_tokens, "seed": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    res.append(r["choices"][0]["message"]["content"])
    toks += r.get("usage", {}).get("completion_tokens", 0)
dt = time.time() - t0
json.dump(res, open(f"out/{name}.json", "w"), ensure_ascii=False, indent=1)
print(f"{name}: {len(res)} completions, {toks} tokens, {dt:.1f}s, {toks/dt:.1f} tok/s")
EOF
    local rc=$?
    grep -m3 -E 'titan tiered experts' $OUT/$name.server.log | head -3
    echo "tiered load lines: $(grep -c 'titan tiered experts' $OUT/$name.server.log)"
    kill $pid; wait $pid 2>/dev/null
    return $rc
}

run_mode stock TITAN_TIERED=0 || exit 1
run_mode tiered TITAN_TIERED=1 || exit 1
run_mode probe TITAN_TIERED=1 TITAN_TIERED_PROBE=1 || exit 1
run_mode half TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=0.5 || exit 1
[ "${SKIP_CPU:-0}" = 1 ] || run_mode cpu TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=0 || exit 1

python3 - <<'EOF'
import json, os
s = json.load(open("out/stock.json"))
chars = sum(len(a) for a in s)
ok = chars > 0
for name in ("tiered", "probe", "half", "cpu"):
    if not os.path.exists(f"out/{name}.json"):
        continue
    t = json.load(open(f"out/{name}.json"))
    same = sum(a == b for a, b in zip(s, t))
    print(f"stock vs {name}: {same}/{len(s)} identical")
    ok &= same == len(s)
    for i, (a, b) in enumerate(zip(s, t)):
        if a != b:
            k = next(j for j in range(min(len(a), len(b)) + 1) if j == min(len(a), len(b)) or a[j] != b[j])
            print(f"  prompt {i} diverges at char {k}: stock={a[k:k+40]!r} {name}={b[k:k+40]!r}")
print(f"{chars} chars of stock output")
print("GATE (M1 + M2 correctness):", "PASS" if ok else "FAIL")
EOF
