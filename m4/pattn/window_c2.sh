#!/bin/bash
# pattn3 window 2: [U] unit tests (CLI config: per-model settings; core scheduler); [T] 35B repeat TTFT after a full
# swap chain, A/B: bin/mistralrs-titan-swap with deploy/models.toml vs bin/mistralrs-titan-pattn3 with
# models-deploy.toml (35B pc -> Spark -> qwen3-14b -> gemma4 -> REDCELL -> OrcaSAQ -> 35B pc x3), same port and order.
source $HOME/titan-engine/top-pattn/m4/pattn/lib.sh
restic_wait
exec > >(tee -a $I/window_c2-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin pattn-c2 45
echo "src: mistral.rs $(git -C $SRC log --oneline -1 | cut -c1-50) $(git -C $SRC diff --quiet && echo clean || echo DIRTY)"
UTARGET="--bin mistralrs" utest 900 mistralrs-cli titan_
utest 900 mistralrs-core scheduler::tests
sha256sum -c $NEWBIN3.sha256
sed "s/^port = 1234/port = $PORT/" $E/deploy/models.toml > $O/models-dep-gate.toml
sed "s/^port = 1234/port = $PORT/" $I/models-deploy.toml > $O/models-gate.toml
chain() { # TAG
  local t=$1
  timeout 600 $C $P pc $PORT qwen3.6-35b $t-q35-before | tail -1
  for m in spark-x2.5 qwen3-14b gemma4-12b redcell-26b orcasaq2-cyber-27b; do timeout 600 $C $P long $PORT $m $t-$m 1 | tail -1; done
  timeout 600 $C $P touch $PORT qwen3.6-35b
  for k in 1 2 3; do timeout 600 $C $P pc $PORT qwen3.6-35b $t-q35-after$k | tail -1; done
  echo "   VRAM: $(smi)"
}
for ab in old new; do
  if [ $ab = old ]; then BIN=$OLDBIN; T=$O/models-dep-gate.toml; else BIN=$NEWBIN3; T=$O/models-gate.toml; fi
  echo "== [T] $ab: $BIN ($(el))"
  if cfg_start c2-$ab $T; then chain c2-$ab; srv_stop
    sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -E "titan swap: (loaded|unloaded)|titan tiered auto:.*fraction" | sed -E 's/^.*INFO [a-z_:0-9]*: //' | cut -c1-160
  fi
done
free -m | head -2
echo "== pattn-c2 end $(el)"
