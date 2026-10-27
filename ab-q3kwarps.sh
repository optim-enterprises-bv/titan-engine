#!/bin/bash
# A/B the single-row Q3_K warp geometry: TITAN_Q3K_WARPS=4 / 8 / 16 (see gguf/mmvq_rows.rs).
#   ./ab-q3kwarps.sh [max_tokens, default 700]
# Each variant is a separate process on its own temp unit, like the other ab-* harnesses; prod
# (titan-swap) is stopped for the sweep and restored on exit. The sha from decode-once.py is what
# tells you whether a variant changed the output or only the speed.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO/bench-lib.sh"
TOK="${1:-700}"
run() {
  local label=$1; shift
  bench_stop_prod; sleep 2
  pkill -f "mistralrs ser[v]e" 2>/dev/null; sleep 4
  systemctl --user reset-failed titan-dense.service 2>/dev/null
  systemd-run --user --unit=titan-dense --property=WorkingDirectory="$REPO" "$@" \
    "$REPO/serve-dense.sh" 65536 >/dev/null 2>&1
  for i in $(seq 1 40); do
    sleep 3
    ss -tln | grep -q ':1234' && break
  done
  echo "=== $label ==="
  "$REPO/decode-once.py" "$TOK"
}

run "8 warps (default)"
run "4 warps"  --setenv=TITAN_Q3K_WARPS=4
run "16 warps" --setenv=TITAN_Q3K_WARPS=16
