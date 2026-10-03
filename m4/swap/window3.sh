trap 'systemctl --user stop swap-srv 2>/dev/null; systemctl --user daemon-reload; systemctl --user start titan-mistral' EXIT
systemctl --user stop titan-mistral
# m4/swap window 3: build the fixed tree, run every gate on the test config, and if all pass deploy: bin/mistralrs-titan-swap
# and the swap unit (deploy/models.toml, port 1234); the EXIT trap starts it. Otherwise the pf3 unit comes back.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/swap/window3.sh
trap 'exit 130' INT TERM HUP
T0=$(date +%s); DEADLINE=$((T0 + 32 * 60)); source $HOME/titan-engine/m4/swap/lib.sh
exec > >(tee -a $S/window3-$(date +%Y%m%d-%H%M).log) 2>&1
echo "== window3 start $(date -Is), titan-mistral: $(systemctl --user is-active titan-mistral)"
killra; free -m | head -2; sleep 3; echo "GPU with no server: $(smi)"
echo "== (f) build ($(el))"
n=0
until mbuild; do n=$((n + 1)); [ $n -ge 4 ] && { echo "no binary"; exit 1; }; retry_wait mbuild 540 || exit 1; done
REV=$(git -C $SRC rev-parse --short HEAD)$(git -C $SRC diff --quiet || echo -dirty)
cp $MBIN $S/mistralrs-swap-w3; BIN=$S/mistralrs-swap-w3; echo "binary from $REV: $(sha256sum $BIN | cut -c1-16)"
srv_start w3 $S/test.toml || exit 1
bc() { BENCH_MODEL=$1 timeout 600 python3 bclient.py $2 $PORT out/w3-$3.json 2>&1 | grep -v "^$"; }
echo "== (d) fresh: service plan on qwen3.6-35b ($(el))"
bc qwen3.6-35b service fresh
echo "== (a) opencode run, model default ($(el))"
( cd /tmp && OPENCODE_CONFIG=$S/opencode-test.json timeout 400 opencode run -m titan-swap-test/default "Reply with exactly: ok" 2>&1 | tail -3 )
echo "== (d) MTP off twin ($(el))"
bc qwen3.6-35b-mtp0 mtpoff mtp0
echo "== (b) 35B -> 80B -> 35B -> gpt-oss-20b (4 queued) -> 35B -> gpt-oss-120b -> 35B ($(el))"
timeout 900 python3 swap.py cycle $PORT out/w3-cycle1.json qwen3.6-35b qwen3-next-80b qwen3.6-35b
timeout 600 python3 swap.py queue $PORT out/w3-queue20.json gpt-oss-20b 4
timeout 900 python3 swap.py cycle $PORT out/w3-cycle2.json qwen3.6-35b gpt-oss-120b qwen3.6-35b
echo "== mxfp4, bonsai, and a mixed burst ($(el))"
timeout 600 python3 swap.py cycle $PORT out/w3-mx.json qwen3.6-35b-mxfp4 bonsai-27b qwen3.6-35b
timeout 600 python3 swap.py mixed $PORT out/w3-mixed.json gpt-oss-20b qwen3.6-35b
timeout 300 python3 swap.py cycle $PORT out/w3-back.json qwen3.6-35b gpt-4o default
echo "== (d) after the cycle: service plan on qwen3.6-35b ($(el))"
bc qwen3.6-35b service after
echo "== OOM recovery: 13k, 60k, 13k, short ($(el))"
timeout 900 python3 swap.py oom $PORT out/w3-oom.json qwen3.6-35b
python3 swap.py models $PORT | grep -c '"status"'
echo "== swap log ($(el))"
swaplog | grep -v "settings for"
srv_stop
echo "== gates ($(el))"
if python3 check.py w3 $SLOG && [ ! -f $S/nodeploy ]; then
  echo "== deploy ($(el))"
  cp $BIN $E/bin/mistralrs-titan-swap
  cp $E/deploy/titan-mistral.service $E/deploy/titan-mistral.service.pf3-rollback
  cp $E/deploy/titan-mistral.service.swap $E/deploy/titan-mistral.service
  cp $E/deploy/titan-mistral.service $HOME/.config/systemd/user/titan-mistral.service
  echo "deployed $(sha256sum $E/bin/mistralrs-titan-swap | cut -c1-16); rollback unit: deploy/titan-mistral.service.pf3-rollback"
else
  echo "== NOT deployed ($(el))"
fi
free -m | head -2
echo "== window3 end $(el)"
