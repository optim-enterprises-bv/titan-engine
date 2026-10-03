#!/bin/bash
# integ2-20261003 window 3 (contingency): window 2's 35B repeat TTFT after the gemma4 swap (median 1.20 s; pcache branch
# 0.87 s, last three ~0.5 s) - noise or merge? A/B, interleaved, same roster-draft.toml: pcache branch binary
# (m4/pcache/out/mistralrs-pcache) vs bin/mistralrs-titan-integ2, each: 35B pc -> gemma4 6 x ~4k -> 35B pc x2.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/top-integ2/m4/integ2/window3.sh
source $HOME/titan-engine/top-integ2/m4/integ2/lib.sh
exec > >(tee -a $I/window3-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin integ2-window3 40
P=$I/pcache.py; PCB=$E/m4/pcache/out/mistralrs-pcache
echo "pcache bin sha256 $(sha256sum $PCB | cut -c1-16); integ2 $(sha256sum $NEWBIN | cut -c1-16)"
for run in pa:$PCB ia:$NEWBIN pb:$PCB ib:$NEWBIN; do
  t=${run%%:*}; BIN=${run#*:}
  echo "== [$t] $BIN ($(el))"
  if [ $(left) -gt 400 ] && cfg_start roster-$t $I/roster-draft.toml; then
    timeout 600 $C $P pc $PORT qwen3.6-35b $t-q35-before
    timeout 300 $C $P touch $PORT gemma4-12b
    timeout 600 $C $P long $PORT gemma4-12b $t-g12-pc0 6
    timeout 600 $C $P touch $PORT qwen3.6-35b
    timeout 600 $C $P pc $PORT qwen3.6-35b $t-q35-after
    timeout 600 $C $P pc $PORT qwen3.6-35b $t-q35-after2
    srv_stop
  fi
done
BIN=$NEWBIN
echo "== integ2-window3 end $(el)"
