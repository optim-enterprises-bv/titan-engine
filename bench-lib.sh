#!/bin/bash
# Shared by the ab-*/prof-* harnesses: prod (titan-swap) owns :1234 and the whole GPU, so a harness
# that starts its own server must stop it first and put it back afterwards -- including when the
# harness dies half way, which is what the EXIT trap is for.
#
#   source "$REPO/bench-lib.sh"
#   bench_stop_prod        # stop titan-swap + any temp unit, arm the restore trap
#   ... start the variant, measure with "$REPO/decode-once.py" ...
#   bench_restore_prod     # optional: the trap does it on exit anyway

BENCH_SERVED=0
BENCH_TRAP_ARMED=0

bench_stop_prod() {
  systemctl --user stop titan-swap 2>/dev/null
  BENCH_SERVED=1
  if [ "$BENCH_TRAP_ARMED" = 0 ]; then
    trap bench_restore_prod EXIT
    BENCH_TRAP_ARMED=1
  fi
  systemctl --user stop titan-dense.service 2>/dev/null
  systemctl --user reset-failed titan-dense.service 2>/dev/null
}

bench_restore_prod() {
  systemctl --user stop titan-dense.service 2>/dev/null
  systemctl --user reset-failed titan-dense.service 2>/dev/null
  pkill -f "mistralrs ser[v]e" 2>/dev/null
  sleep 3
  if [ "$BENCH_SERVED" = 1 ]; then
    systemctl --user start titan-swap
    for _ in $(seq 1 40); do
      sleep 3
      systemctl --user is-active --quiet titan-swap && break
    done
  fi
}
