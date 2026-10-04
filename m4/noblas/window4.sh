#!/bin/bash
# noblas window 4 (phase 2): the staged noblas binary (bin/mistralrs-titan-noblas) through one from-config server with
# the deployed roster (deploy/models.toml, port 18690): per model G1 / G3 / G5 (gates_m.py), concurrency-4 bursts on
# Spark and qwen3-14b; the 35B 40 x 256 greedy gates (MTP=2 vs m6/out/h-off.json, MTP-off file vs mtpfile-off.json);
# then the same roster gates on the deployed binary while time allows.
source $HOME/titan-engine/top-noblas/m4/noblas/lib.sh
restic_wait
exec > >(tee -a $I/window4-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin noblas-w4 58
sha256sum -c $E/bin/mistralrs-titan-noblas.sha256 2>&1 | sed "s#^#$E/bin/#"
BIN=$NEWBIN roster_gates new
BIN=$NEWBIN
coll noblas-m2 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2 && cmp40 $E/m6/out/h-off.json $E/m3/out/noblas-m2.json
coll noblas-mf0 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=0 && cmp40 $E/m6/out/mtpfile-off.json $E/m3/out/noblas-mf0.json
BIN=$OLDBIN roster_gates old
echo "== noblas-w4 end $(el)"
