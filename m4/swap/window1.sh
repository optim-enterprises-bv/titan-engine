trap 'systemctl --user stop swap-srv 2>/dev/null; systemctl --user start titan-mistral' EXIT
systemctl --user stop titan-mistral
# m4/swap window 1: nvcc-free build of mr-swap (model-swap), then on the test config (port 18590):
# (a) opencode against the default model, (b)+(c) the swap cycle 35B -> 80B -> 35B -> 20b -> 35B -> 120b -> 35B with
# the memory after each unload, (e) concurrent requests during swaps, the model lists, the idle TTL.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/swap/window1.sh
trap 'exit 130' INT TERM HUP
T0=$(date +%s); DEADLINE=$((T0 + 32 * 60)); source $HOME/titan-engine/m4/swap/lib.sh
exec > >(tee -a $S/window1-$(date +%Y%m%d-%H%M).log) 2>&1
echo "== window1 start $(date -Is), titan-mistral: $(systemctl --user is-active titan-mistral)"
killra; free -m | head -2; sleep 3; echo "GPU with no server: $(smi)"
echo "== (f) build ($(el))"
n=0
until mbuild; do n=$((n + 1)); [ $n -ge 4 ] && { echo "no binary"; exit 1; }; retry_wait mbuild 540 || exit 1; done
REV=$(git -C $SRC rev-parse --short HEAD)$(git -C $SRC diff --quiet || echo -dirty)
cp $MBIN $S/mistralrs-swap-w1; BIN=$S/mistralrs-swap-w1; echo "binary from $REV: $(sha256sum $BIN | cut -c1-16)"
ldd $BIN | grep -iE "cuda|nvrtc|cublas" | head -5
echo "== server ($(el))"
srv_start w1 $S/test.toml || { retry_wait srv 300 && srv_start w1 $S/test.toml; } || exit 1
echo "GPU with the 35B loaded: $(smi)"
python3 swap.py models $PORT
echo "== (a) opencode run, model default ($(el))"
( cd /tmp && OPENCODE_CONFIG=$S/opencode-test.json timeout 400 opencode run -m titan-swap-test/default "Reply with exactly: ok" 2>&1 | tail -5 )
echo "opencode rc=$? ($(el))"
echo "== (b) swap cycle ($(el))"
timeout 900 python3 swap.py cycle $PORT out/w1-cycle1.json qwen3.6-35b qwen3-next-80b qwen3.6-35b
echo "== (e) 4 concurrent requests for gpt-oss-20b while the 35B is resident ($(el))"
timeout 600 python3 swap.py queue $PORT out/w1-queue20.json gpt-oss-20b 4
timeout 900 python3 swap.py cycle $PORT out/w1-cycle2.json qwen3.6-35b gpt-oss-120b qwen3.6-35b
echo "== (e) mixed: 2 x bonsai-27b and 2 x qwen3.6-35b at once ($(el))"
timeout 600 python3 swap.py mixed $PORT out/w1-mixed.json bonsai-27b qwen3.6-35b
echo "== TTL: bonsai-27b (ttl 20 s), then wait 40 s ($(el))"
timeout 300 python3 swap.py cycle $PORT out/w1-ttl.json bonsai-27b
sleep 40; python3 -c "import json,urllib.request;print({m['id']:m.get('status') for m in json.load(urllib.request.urlopen('http://127.0.0.1:$PORT/v1/models'))['data']})"
echo "GPU after the TTL unload: $(smi)"
timeout 300 python3 swap.py cycle $PORT out/w1-last.json qwen3.6-35b
echo "== swap log ($(el))"
swaplog
srv_stop
free -m | head -2
echo "== window1 end $(el)"
