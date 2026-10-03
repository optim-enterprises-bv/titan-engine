trap 'systemctl --user stop swap-srv pf3-srv 2>/dev/null; systemctl --user start titan-mistral' EXIT
systemctl --user stop titan-mistral
# m4/swap window 5: the live regression. Same cold-prompt sequence (13k 23k 28k 36k) on (A) pf3 (the rollback unit's
# binary and env), (B) the swap binary with the live models.toml, fresh, then (C) after a swap away and back; then the
# 64.7k prompt the live opencode sent, on pf3 and on the swap build, each followed by a 13k. pfs / plan / VRAM lines.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/swap/window5.sh
trap 'exit 130' INT TERM HUP
T0=$(date +%s); DEADLINE=$((T0 + 30 * 60)); source $HOME/titan-engine/m4/swap/lib.sh
exec > >(tee -a $S/window5-$(date +%Y%m%d-%H%M).log) 2>&1
echo "== window5 start $(date -Is)"; killra; sleep 3; echo "GPU with no server: $(smi)"
O=$S/out/w5-seq.jsonl; rm -f $O
pfslines() { grep -oE "titan tiered auto:.*|titan pfs: (ring|no room).*|titan swap: (loaded|unloaded).*|OUT_OF_MEMORY|pools trimmed.*" $1 | cut -c1-200; }
echo "== (A) pf3 ($(el))"
PENV=$(python3 - <<'PY'
import os
print(" ".join(l.strip()[len("Environment="):].replace("%h",os.path.expanduser("~")) for l in open(os.path.expanduser("~/titan-engine/deploy/titan-mistral.service.pf3-rollback")) if l.startswith("Environment=")))
PY
)
echo "pf3 env: $PENV"
systemd-run --user --unit=pf3-srv --collect -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=900 -p StandardOutput=truncate:$S/out/w5-pf3.log -p StandardError=truncate:$S/out/w5-pf3.log \
  env $PENV $E/bin/mistralrs-titan-pf3 serve --host 127.0.0.1 -p $PORT --paged-attn off --max-seq-len 65536 --format gguf -m $M/Qwen3.6-35B-A3B-MTP -f Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
until curl -sf -m 2 -o /dev/null localhost:$PORT/v1/models; do sleep 2; systemctl --user -q is-active pf3-srv || break; done
echo "pf3 up ($(el)), GPU $(smi)"
timeout 600 python3 seq.py $PORT default $O pf3 13k 23k 28k 36k
echo "pf3 after the sequence: GPU $(smi)"
timeout 300 python3 seq.py $PORT default $O pf3 64k 13k
systemctl --user stop pf3-srv; pfslines $S/out/w5-pf3.log
echo "== (B) swap binary, live models.toml, fresh ($(el))"
sed 's/^port = 1234/port = 18590/' $E/deploy/models.toml > $S/live-test.toml
BIN=$E/bin/mistralrs-titan-swap
srv_start w5 $S/live-test.toml || exit 1
echo "GPU $(smi)"
timeout 600 python3 seq.py $PORT qwen3.6-35b $O swap-fresh 13k 23k 28k 36k
echo "== (C) swap away (bonsai) and back ($(el))"
timeout 300 python3 swap.py cycle $PORT out/w5-swap.json bonsai-27b qwen3.6-35b | cut -c1-120
timeout 600 python3 seq.py $PORT qwen3.6-35b $O swap-after 13k 23k 28k 36k
echo "after the sequence: GPU $(smi)"
timeout 300 python3 seq.py $PORT qwen3.6-35b $O swap-after 64k 13k
srv_stop; pfslines $SLOG
echo "== summary"; python3 -c "
import json
for l in open('$O'): r=json.loads(l); print(r['tag'], r['size'], r['ok'], r.get('prompt_tokens'), round(r.get('prompt_tok_s') or 0), round(r['wall'],1))"
echo "== window5 end $(el)"
