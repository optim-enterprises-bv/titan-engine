#!/bin/bash
# wait for >= 12 GiB available, then run the M3/M2 set under a 10.5G cap
while [ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -lt 12 ]; do sleep 20; done
exec systemd-run --user --unit=titan-m3 --collect -p MemoryMax=10500M -p MemorySwapMax=0 -p WorkingDirectory=$PWD \
  -p StandardOutput=truncate:$PWD/run-all.log -p StandardError=truncate:$PWD/run-all.log ./run-all.sh
