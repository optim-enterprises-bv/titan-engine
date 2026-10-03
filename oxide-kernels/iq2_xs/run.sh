#!/bin/bash
# Build (serialised), regenerate the sm_120a cubin from the fresh PTX, then run the gate.
# `cargo oxide build` cleans the crate dir, so the cubin MUST be produced after each build.
cd $HOME/titan-engine/oxide-kernels/iq2_xs || exit 1
flock -w 900 /tmp/iq2s_build.lock bash -c 'bash b.sh > /dev/null 2>&1'
grep -cE '^error' build.log | sed 's/^/build_errors=/'
/usr/local/cuda-13.3/bin/ptxas --gpu-name=sm_120a -o iq2_xs.cubin iq2_xs.ptx || exit 1
systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 env "$@" ./target/release/iq2_xs
