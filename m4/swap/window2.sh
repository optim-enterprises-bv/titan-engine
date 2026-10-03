trap 'systemctl --user stop swap-srv 2>/dev/null; systemctl --user start titan-mistral' EXIT
systemctl --user stop titan-mistral
# m4/swap window 2: (d) 35B speed and identity fresh vs after a swap cycle (bench client, service plan = MTP=2 vs
# h-off; mtpoff plan on the MTP-off twin = MTP off vs h-off), then the OOM recovery test (13k, 60k, 13k).
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/swap/window2.sh [BIN]
trap 'exit 130' INT TERM HUP
T0=$(date +%s); DEADLINE=$((T0 + 32 * 60)); source $HOME/titan-engine/m4/swap/lib.sh
exec > >(tee -a $S/window2-$(date +%Y%m%d-%H%M).log) 2>&1
echo "== window2 start $(date -Is), titan-mistral: $(systemctl --user is-active titan-mistral)"
killra; free -m | head -2; sleep 3; echo "GPU with no server: $(smi)"
echo "== (f) build ($(el))"
n=0
until mbuild; do n=$((n + 1)); [ $n -ge 4 ] && { echo "no binary"; exit 1; }; retry_wait mbuild 540 || exit 1; done
REV=$(git -C $SRC rev-parse --short HEAD)$(git -C $SRC diff --quiet || echo -dirty)
cp $MBIN $S/mistralrs-swap-w2; BIN=$S/mistralrs-swap-w2; echo "binary from $REV: $(sha256sum $BIN | cut -c1-16)"
srv_start w2 $S/test.toml || exit 1
bc() { BENCH_MODEL=$1 timeout 600 python3 bclient.py $2 $PORT out/w2-$3.json 2>&1 | grep -v "^$"; }
echo "== (d) fresh: service plan on qwen3.6-35b ($(el))"
bc qwen3.6-35b service fresh
echo "== (a) opencode run, model default ($(el))"
( cd /tmp && OPENCODE_CONFIG=$S/opencode-test.json timeout 400 opencode run -m titan-swap-test/default "Reply with exactly: ok" 2>&1 | tail -5 )
echo "== (d) MTP off twin ($(el))"
bc qwen3.6-35b-mtp0 mtpoff mtp0
echo "== swap cycle ($(el))"
timeout 900 python3 swap.py cycle $PORT out/w2-cycle.json qwen3-next-80b qwen3.6-35b gpt-oss-120b qwen3.6-35b-mxfp4 qwen3.6-35b
echo "== (d) after the cycle: service plan on qwen3.6-35b ($(el))"
bc qwen3.6-35b service after
echo "== OOM recovery: 13k, 60k, 13k, short ($(el))"
timeout 900 python3 swap.py oom $PORT out/w2-oom.json qwen3.6-35b
echo "== gpt-oss-20b unload diagnosis: 20b, 35B, 20b x4 concurrent, 35B ($(el))"
timeout 600 python3 swap.py cycle $PORT out/w2-20b.json gpt-oss-20b qwen3.6-35b
timeout 600 python3 swap.py queue $PORT out/w2-20bq.json gpt-oss-20b 4
timeout 600 python3 swap.py cycle $PORT out/w2-20b2.json qwen3.6-35b
echo "== swap log ($(el))"
swaplog
srv_stop
free -m | head -2
echo "== window2 end $(el)"
