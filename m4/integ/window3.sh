#!/bin/bash
# integ-20261002 window 3 (contingency): REDCELL G1/G2 and gemma4-12b G1 have no valid branch-level reference:
# bin/mistralrs-g4s-w6 was built at 19:47 from the uncommitted tree that had TITAN_G4_ATTN_F32 on by default (the commit
# a7ea678f9 at 19:55 made it opt-in; w6 G1 median |dlogprob| 0.0019 = the commit message's ATTN_F32 number), and the
# baseline-redcell-g2.json binary (w5) predates the unaligned-K routing fix. (1) build the gemma4-spark head a7ea678f9 as
# committed (mr-g4s, clean) -> bin/mistralrs-g4s-a7ea (scratch, not deployed); (2) for integ and that binary: REDCELL
# --prefix-cache-n 0 G1 + G2, a second fresh server G1 + G2 (determinism), then the baseline procedure (default cache,
# G1 -> G3 -> G2); gemma4-12b G1 (8192, 0:48, default cache) on the clean branch binary.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/integ/window3.sh
source $HOME/titan-engine/m4/integ/lib.sh
exec > >(tee -a $I/window3-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin integ-window3 58
G4=$E/m4/g4s; R=$G4/out; SP=$E/m4/sparkpf; W4=$E/mr-g4s; T4=$E/target-g4sclean-oxide; B4=$E/bin/mistralrs-g4s-a7ea
gate() { [ $(left) -gt 240 ] || { echo "skip $1 $2: $(left)s left"; return 1; }; local mode=$1 out=$2; shift 2; timeout 900 python3 $SP/gates.py $mode $PORT titan $O/$out.json "$@"; }
cmpg() { python3 $SP/cmp.py "$@" 2>&1; }
same() { python3 $I/same.py "$@"; }
tmap() { timeout 300 systemd-run --user --unit=integ-tmap --collect --wait --pipe -q -p MemoryMax=4G $HOME/ai/convert-env/bin/python3 $G4/tokmap.py "$@"; }
echo "g4s src: $(git -C $W4 log --oneline -1 | cut -c1-80) $(git -C $W4 diff --quiet && git -C $W4 diff --cached --quiet && echo clean || echo DIRTY)"

echo "== [1] build gemma4-spark a7ea678f9 ($(el))"
L=$O/build-g4sclean.log; t=$(date +%s)
systemctl --user reset-failed integ-build 2>/dev/null
timeout 1530 systemd-run --user --unit=integ-build --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 -p RuntimeMaxSec=1500 \
  -p WorkingDirectory=$W4 --setenv=CARGO_TARGET_DIR=$T4 --setenv=TITAN_OXIDE_DIR=$E/oxide-kernels --setenv=CUDA_HOME=$E/nocuda-bin \
  --setenv=CUDA_PATH=$E/nocuda-bin --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDARC_CUDA_VERSION=13030 --setenv=CARGO_BUILD_JOBS=2 \
  --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 "--setenv=RUSTFLAGS=-L $E/lib" --setenv=PATH=$HOME/.cargo/bin:/usr/bin:/bin \
  -p StandardOutput=truncate:$L -p StandardError=truncate:$L nice -n 10 cargo build --release -p mistralrs-cli --features oxide
rc=$?; grep -q "^error" $L && rc=1; echo "build rc=$rc in $(( $(date +%s) - t ))s ($(el))"; grep -E "^error" -A10 $L | head -30; grep Finished $L
HAVE4=0; [ $rc = 0 ] && [ $T4/release/mistralrs -nt $W4/mistralrs-core/src/moe/experts/backends.rs ] && cp $T4/release/mistralrs $B4 && HAVE4=1 && echo "branch binary sha256 $(sha256sum $B4 | cut -c1-16)"

RED=REDCELL-26B-A4B-OSINT-Cyber-APEX-Mini.gguf; RP=$G4/prompts/redcell.json; RA=(--max-model-len 8192 --max-seq-len 8192 -n 0:30 --dtype bf16)
pc0() { # TAG
  for rep in a b; do
    if srv_start $1-$rep $M/redcell-26b $RED "" "${RA[@]}" --prefix-cache-n 0; then gate g1 $1-g1$rep $RP; gate g2 $1-g2$rep $RP 256; srv_stop; fi
  done
}
proc() { # TAG
  if srv_start $1 $M/redcell-26b $RED "" "${RA[@]}"; then gate g1 $1-g1 $RP; gate g3 $1-g3 2>&1 | tail -1; gate g2 $1-g2 $RP 256; srv_stop; fi
}
echo "== [2] REDCELL integ, --prefix-cache-n 0, twice ($(el))"
BIN=$NEWBIN; pc0 ri
if [ $HAVE4 = 1 ]; then echo "== [3] REDCELL branch a7ea678f9, --prefix-cache-n 0, twice ($(el))"; BIN=$B4; pc0 rb; fi
echo "== [4] baseline procedure (default cache, G1 -> G3 -> G2): branch, integ ($(el))"
[ $HAVE4 = 1 ] && { BIN=$B4; proc rbp; }
BIN=$NEWBIN; proc rip
if [ $HAVE4 = 1 ]; then
  echo "== [5] gemma4-12b G1 on the branch binary ($(el))"
  BIN=$B4
  if srv_start g12b $M/gemma4-12b-qat gemma-4-12b-it-qat-q4_0.gguf "" --max-model-len 8192 --max-seq-len 8192 -n 0:48 --dtype bf16; then gate g1 g12b-g1 $G4/prompts/gemma4-12b.json; srv_stop; fi
fi
BIN=$NEWBIN

echo "== comparisons ($(el))"
for f in ri-g1a ri-g1b rb-g1a rb-g1b rbp-g1 rip-g1 g12b-g1; do [ -e $O/$f.json ] && tmap gemma4 $O/$f.json; done
for p in "ri-g1a ri-g1b" "rb-g1a rb-g1b" "ri-g1a rb-g1a" "ri-g2a ri-g2b" "rb-g2a rb-g2b" "ri-g2a rb-g2a" "rip-g1 rbp-g1" "rip-g2 rbp-g2" "ri-g1a red-g1"; do
  set -- $p; [ -e $O/$1.json ] && [ -e $O/$2.json ] && same $O/$1.json $O/$2.json
done
for f in ri-g1a rb-g1a; do [ -e $O/$f.json ] && cmpg g1 $O/$f.json $R/ref-red-g1.json; done
for f in rip-g2 rbp-g2 ri-g2a rb-g2a; do [ -e $O/$f.json ] && { same $O/$f.json $R/baseline-redcell-g2.json; cmpg g2 $O/$f.json $R/baseline-redcell-g2.json | head -1; cmpg g2 $O/$f.json $R/ref-red-g2.json | head -1; }; done
[ -e $O/g12b-g1.json ] && { same $O/g12b-g1.json $O/g12-g1.json; cmpg g1 $O/g12b-g1.json $R/ref-g12-g1.json; }
free -m | head -2
echo "== integ-window3 end $(el)"
