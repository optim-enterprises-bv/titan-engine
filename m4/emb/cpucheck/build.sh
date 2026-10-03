#!/bin/bash
# build.sh MODE: "stub" links the cudarc stand-in (the unfixed iquant.rs needs it), "plain" does not
# (a plain build only compiles once the cudarc impls are behind #[cfg(feature = "cuda")]).
set -e; cd "$(dirname "$0")"
D=$HOME/titan-engine/target-integ-oxide/release/deps
HALF=$(ls $D/libhalf-*.rlib | head -1)
X=()
if [ "$1" = stub ]; then
  rustc --edition 2021 --crate-type rlib --crate-name cudarc -O -o libcudarc.rlib cudarc_stub.rs
  X=(--extern cudarc=libcudarc.rlib)
fi
rustc --edition 2021 -O -C debug-assertions=on -o iqcheck-$1 iqcheck.rs --extern half=$HALF -L dependency=$D -L . "${X[@]}"
