#!/bin/bash
# usage: run.sh CONFIG.json  -> out/<name>.log ; CPU only, capped at 4 GB
cd $(dirname $0)
n=$(basename $1 .json)
systemd-run --user --collect --wait -q -p MemoryMax=4G -p MemorySwapMax=0 -p WorkingDirectory=$PWD \
  -p StandardOutput=truncate:$PWD/out/$n.log -p StandardError=inherit nice -n 5 python3 sim.py $1
