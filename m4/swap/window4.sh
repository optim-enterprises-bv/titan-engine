trap 'systemctl --user stop swap-srv 2>/dev/null; systemctl --user daemon-reload; systemctl --user start titan-mistral' EXIT
systemctl --user stop titan-mistral
# m4/swap window 4 (short): rebuild with unknown model names routed to the default model, check that and the basics
# (opencode, one swap and back, 35B identity), then deploy bin/mistralrs-titan-swap + the swap unit; the EXIT trap starts it.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/swap/window4.sh
trap 'exit 130' INT TERM HUP
T0=$(date +%s); DEADLINE=$((T0 + 20 * 60)); source $HOME/titan-engine/m4/swap/lib.sh
exec > >(tee -a $S/window4-$(date +%Y%m%d-%H%M).log) 2>&1
echo "== window4 start $(date -Is), titan-mistral: $(systemctl --user is-active titan-mistral)"
killra
n=0
until mbuild; do n=$((n + 1)); [ $n -ge 3 ] && { echo "no binary"; exit 1; }; retry_wait mbuild 300 || exit 1; done
REV=$(git -C $SRC rev-parse --short HEAD)$(git -C $SRC diff --quiet || echo -dirty)
cp $MBIN $S/mistralrs-swap-w4; BIN=$S/mistralrs-swap-w4; echo "binary from $REV: $(sha256sum $BIN | cut -c1-16)"
srv_start w4 $S/test.toml || exit 1
ok=1
timeout 300 python3 swap.py cycle $PORT out/w4-names.json default gpt-4o qwen3.6-35b || ok=0
python3 -c "import json,sys; r=json.load(open('out/w4-names.json')); sys.exit(0 if all(x.get('correct') for x in r) else 1)" || ok=0
( cd /tmp && OPENCODE_CONFIG=$S/opencode-test.json timeout 400 opencode run -m titan-swap-test/default "Reply with exactly: ok" 2>&1 | tail -2 ) | tee out/w4-opencode.txt
grep -qx "ok" <(sed 's/\x1b\[[0-9;]*m//g' out/w4-opencode.txt | tr -d ' \r') || ok=0
timeout 300 python3 swap.py cycle $PORT out/w4-swap.json bonsai-27b qwen3.6-35b
python3 -c "import json,sys; r=json.load(open('out/w4-swap.json')); sys.exit(0 if all(x.get('correct') for x in r) else 1)" || ok=0
BENCH_MODEL=qwen3.6-35b timeout 400 python3 bclient.py mtpoff $PORT out/w4-ident.json | grep -v "^$"
python3 -c "import json,sys; r=json.load(open('out/w4-ident.json')); sys.exit(0 if r['eight_1']['identity']=='8/8' else 1)" || ok=0
swaplog | grep -v "settings for" | tail -12
srv_stop
if [ $ok = 1 ] && [ ! -f $S/nodeploy ]; then
  cp $BIN $E/bin/mistralrs-titan-swap
  [ -f $E/deploy/titan-mistral.service.pf3-rollback ] || cp $E/deploy/titan-mistral.service $E/deploy/titan-mistral.service.pf3-rollback
  cp $E/deploy/titan-mistral.service.swap $E/deploy/titan-mistral.service
  cp $E/deploy/titan-mistral.service $HOME/.config/systemd/user/titan-mistral.service
  echo "== deployed $(sha256sum $E/bin/mistralrs-titan-swap | cut -c1-16) ($(el)); rollback: deploy/titan-mistral.service.pf3-rollback"
else
  echo "== NOT deployed (checks ok=$ok) ($(el))"
fi
echo "== window4 end $(el)"
