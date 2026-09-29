#!/bin/bash
# titan-094 integration window: build (oxide), bench x2 (candidate + clean baseline), monitor check, deploy gate.
# Run as: flock ~/titan-engine/.gpu.lock bash <this script>
#
# NOTE on the 30-minute service gap: RULES-agents.md and ~/titan-engine/.nogap were modified on disk at 23:14:41
# today, about a minute after this task started, adding a "NO service gap" section labeled "23:30" -- a
# timestamp then in the future relative to the file's actual mtime, which read as a possible injected change.
# The coordinator (main session) confirmed at 23:20 that it wrote the change itself at 23:14 on the user's
# verbal instruction ("stop wasting time with this 30 minute service gap") and fixed the stray "23:30" label
# to "23:14" in RULES-agents.md -- a correction that matches exactly the inconsistency that was flagged, which
# is treated as sufficient corroboration. This run (relaunched at ~23:20 after an earlier instance was killed
# while still in the gap-wait loop, before anything was built or the service was touched) does NOT wait for
# the 30-minute gap: it stops titan-mistral immediately after acquiring the lock, per that instruction.
set -u
E=$HOME/titan-engine; MR=$E/mr-094; MON=$E/m4/mon; O=$MON/out
mkdir -p "$O"
LOGDIR=$E/m4/094-int; mkdir -p "$LOGDIR"
exec > >(tee -a "$LOGDIR/window-$(date +%Y%m%d-%H%M).log") 2>&1
DECISION=$LOGDIR/decision.json
echo "== lock acquired $(date -Is)"
STOP_TS=$(date -Is)
echo "== no service gap (per coordinator/user instruction at 23:14/23:20) -- stopping now $STOP_TS"

RESTART_TS=""
trap 'RESTART_TS=$(date -Is); systemctl --user start titan-mistral; echo "== titan-mistral restart requested $RESTART_TS"' EXIT
trap 'exit 130' INT TERM HUP
systemctl --user stop titan-mistral

killra() { for p in $(ps -eo pid=,ppid=,comm= | awk '$2 == 878468 && $3 ~ /rust-analyzer/ {print $1}'); do echo "killing rust-analyzer $p"; kill "$p"; done; }
killra

T0=$(date +%s)
el() { echo "$(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s"; }

# ---------------- 1. build (nvcc-free, oxide) ----------------
echo "== [1/5] build $(el) $(date -Is)"
BLOG=$LOGDIR/build-$(date +%Y%m%d-%H%M).log
timeout 1500 systemd-run --user --unit=b094int-build --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 -p RuntimeMaxSec=1480 \
  -p WorkingDirectory="$MR" --setenv=CARGO_TARGET_DIR="$E/target-094-oxide" --setenv=TITAN_OXIDE_DIR="$E/oxide-kernels" --setenv=CUDA_HOME="$E/nocuda-bin" \
  --setenv=CUDA_PATH="$E/nocuda-bin" --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDARC_CUDA_VERSION=13030 --setenv=CARGO_BUILD_JOBS=2 \
  --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 "--setenv=RUSTFLAGS=-L $E/lib" --setenv=PATH="$HOME/.cargo/bin:/usr/bin:/bin" \
  -p StandardOutput=truncate:"$BLOG" -p StandardError=truncate:"$BLOG" nice -n 10 cargo build --release -p mistralrs-cli --features oxide
BUILD_RC=$?
systemctl --user stop b094int-build 2>/dev/null
grep -q "^error" "$BLOG" && BUILD_RC=1
echo "build rc=$BUILD_RC ($(el))"
tail -40 "$BLOG"
BIN=$E/target-094-oxide/release/mistralrs
if [ "$BUILD_RC" -ne 0 ] || [ ! -x "$BIN" ]; then
  echo "BUILD FAILED - aborting, no bench/deploy"
  python3 -c "import json; json.dump({'stage':'build','ok':False}, open('$DECISION','w'))"
  exit 1
fi
ls -la "$BIN"
CAND=$E/bin/mistralrs-titan-094-pfsmon-candidate
cp -a "$BIN" "$CAND"
sha256sum "$CAND"

# ---------------- 2. bench: candidate (094-pfs-mon) ----------------
echo "== [2/5] bench 094-pfs-mon $(el) $(date -Is)"
killra
timeout 1200 bash "$E/bench/run.sh" "$CAND" 094-pfs-mon
echo "bench 094-pfs-mon rc=$? ($(el))"

# ---------------- 3. bench: clean baseline ----------------
echo "== [3/5] bench baseline-094-clean $(el) $(date -Is)"
killra
timeout 1200 bash "$E/bench/run.sh" "$E/bin/mistralrs-titan-094" baseline-094-clean
echo "bench baseline-094-clean rc=$? ($(el))"

# ---------------- 4. monitor check (13k prompt + long decode, TITAN_PFS=1, service config) ----------------
echo "== [4/5] monitor check $(el) $(date -Is)"
killra
P=18492
cat > "$LOGDIR/mon-inner.sh" <<'INNER'
set -u
E=$HOME/titan-engine; MON=$E/m4/mon; O=$MON/out; P=18492
mapfile -t SENV < <(python3 "$E/bench/lib/svcconf.py" env)
mapfile -t SARGS < <(python3 "$E/bench/lib/svcconf.py" args "$P")
cd "$MON"
env "${SENV[@]}" "$BIN" --seed 0 "${SARGS[@]}" > "$O/094-live.server.log" 2>&1 &
pid=$!
for i in $(seq 1 200); do curl -sf -m 2 -o /dev/null localhost:$P/v1/models && break; kill -0 $pid 2>/dev/null || break; sleep 2; done
echo "server up after $((i*2))s"
curl -s localhost:$P/monitor | head -c 200; echo
curl -s localhost:$P/v1/titan/stats > "$O/094-idle-first.json"
python3 "$MON/poll.py" $P "$O/094-live-poll.jsonl" & ppid=$!
echo "-- 13k prompt (cold) $(date +%T)"
python3 "$E/m4/bigprompt.py" $P 13000 > "$O/094-big1.txt" 2>&1
cat "$O/094-big1.txt"
echo "-- long decode MTP=2 $(date +%T)"
python3 "$MON/longgen.py" $P 1500 > "$O/094-decode.txt" 2>&1
cat "$O/094-decode.txt"
curl -s localhost:$P/v1/titan/stats > "$O/094-final.json"
kill $ppid 2>/dev/null
kill -TERM $pid 2>/dev/null; sleep 5; kill -KILL $pid 2>/dev/null; true
INNER
rm -f "$O"/094-*.json "$O"/094-*.txt "$O/094-live-poll.jsonl" "$O/094-live.server.log"
timeout 900 systemd-run --user --unit=mon094-live --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=880 \
  --setenv=BIN="$CAND" bash "$LOGDIR/mon-inner.sh"
echo "monitor check rc=$? ($(el))"
grep -E "panicked|ILLEGAL|ERROR|error:" "$O/094-live.server.log" | head -5

# ---------------- 5. decide + (maybe) deploy ----------------
echo "== [5/5] decision $(el) $(date -Is)"
python3 "$LOGDIR/decide.py" "$DECISION" "$CAND"
DEPLOY=$(python3 -c "import json;print(json.load(open('$DECISION'))['deploy'])")
echo "deploy=$DEPLOY"
if [ "$DEPLOY" = "True" ]; then
  DEST=$E/bin/mistralrs-titan-094b
  cp -a "$CAND" "$DEST"
  sha256sum "$DEST"
  for f in "$E/deploy/titan-mistral.service" "$HOME/.config/systemd/user/titan-mistral.service"; do
    cp "$f" "$f.bak-094int"
    sed -i 's#bin/mistralrs-titan-pfs#bin/mistralrs-titan-094b#' "$f"
    grep -n "ExecStart=\|^Environment=" "$f"
  done
  systemctl --user daemon-reload
  echo "deployed $DEST, unit files updated, daemon-reload done; trap will restart the service on exit"
else
  echo "NOT deploying: $(python3 -c "import json;print(json.load(open('$DECISION'))['reasons'])")"
fi
echo "== window done $(el) $(date -Is)"
