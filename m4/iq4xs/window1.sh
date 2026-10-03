#!/bin/bash
# IQ4_XS dense window 1: oxide builds (iq4_xs mmvq/dequant + mutants, fmt_mmq IQ4_XS MMQ), nvcc reference cubin for
# mul_mat_q<IQ4_XS>, kernel gates, existing-PTX identity, staging, mistral.rs build, 9B e2e vs llama.cpp, regression core.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/top-iq4xs/m4/iq4xs/window1.sh
source $HOME/titan-engine/top-iq4xs/m4/iq4xs/lib.sh
exec > >(tee -a $I/window1-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin iq4xs-window1 58
echo "src: mistral.rs $(git -C $SRC log --oneline -1 | cut -c1-60) $(git -C $SRC diff --quiet && echo clean || echo DIRTY)"
echo "     candle $(git -C $E/top-iq4xs log --oneline -1 | cut -c1-60) $(git -C $E/top-iq4xs diff --quiet && echo clean || echo DIRTY); oxide $(git -C $OX log --oneline -1 | cut -c1-50) $(git -C $OX diff --quiet && echo clean || echo DIRTY)"
echo "     oxide-kernels symlink: $(readlink $E/top-iq4xs/oxide-kernels); TITAN_* in env: $(env | grep -c '^TITAN_')"
# fix loop: on a failed build the window waits for $I/fix.flag (written after a fix) while time allows
waitfix() { rm -f $I/fix.flag; date -Is > $O/$1-failed.flag; echo "!! $1 failed: waiting for $I/fix.flag ($(left)s left)";
  while [ ! -e $I/fix.flag ] && [ $(left) -gt $2 ]; do sleep 10; done; rm -f $O/$1-failed.flag; [ -e $I/fix.flag ] && rm -f $I/fix.flag; }

echo "== [N] nvcc reference cubin mul_mat_q<IQ4_XS> (background) ($(el))"
systemctl --user reset-failed iq4xs-nvcc 2>/dev/null; mkdir -p $OX/fmt_mmq/ref.new
systemd-run --user --unit=iq4xs-nvcc --collect -q -p MemoryMax=6G -p MemorySwapMax=0 -p RuntimeMaxSec=1500 \
  -p StandardOutput=truncate:$O/nvcc-mmq-iq4_xs.log -p StandardError=truncate:$O/nvcc-mmq-iq4_xs.log \
  nice -n 5 sh -c "sed '\$d' $OX/fmt_mmq/ref/nvcc_mmq_iq4_xs.sh > $O/nvcc_cubin_only.sh && sh $O/nvcc_cubin_only.sh $OX/fmt_mmq/ref.new && mv $OX/fmt_mmq/ref.new/mmq_iq4_xs.cubin $OX/fmt_mmq/ref/"

echo "== [K] oxide builds ($(el))"
git -C $OX show HEAD:iq4_xs/iq4_xs.ptx > $O/committed-iq4_xs.ptx; git -C $OX show HEAD:fmt_mmq/fmt_mmq.ptx > $O/committed-fmt_mmq.ptx
until oxbuild iq4_xs iq4xs-ox 900; do waitfix oxbuild-iq4_xs 1800 || exit 1; done
until oxbuild fmt_mmq iq4xs-ox 1200; do waitfix oxbuild-fmt_mmq 1800 || exit 1; done
python3 $I/ptxcmp.py $O/committed-iq4_xs.ptx $OX/iq4_xs/iq4_xs.ptx
python3 $I/ptxcmp.py $O/committed-fmt_mmq.ptx $OX/fmt_mmq/fmt_mmq.ptx
/usr/local/cuda/bin/ptxas -arch=sm_120a -O3 -o /dev/null $OX/fmt_mmq/fmt_mmq.ptx && echo "ptxas fmt_mmq.ptx sm_120a ok"
/usr/local/cuda/bin/ptxas -arch=sm_120 -O3 -o /dev/null $OX/iq4_xs/iq4_xs.ptx && echo "ptxas iq4_xs.ptx sm_120 ok"

echo "== [K1] iq4_xs gate ($(el))"
(cd $OX/iq4_xs && timeout 900 ./target/release/iq4_xs > $O/gate-iq4xs.log 2>&1); GX=$?; echo "iq4_xs gate rc=$GX"
grep -E "mutation|cpu oracle|FAIL|PASS|launches|failing" $O/gate-iq4xs.log | grep -v "^CASE" | tail -12
echo "== [K2] fmt_mmq IQ4_XS gate ($(el))"
for i in $(seq 1 90); do systemctl --user -q is-active iq4xs-nvcc || break; sleep 10; done
tail -3 $O/nvcc-mmq-iq4_xs.log; ls -la $OX/fmt_mmq/ref/mmq_iq4_xs.cubin
(cd $OX/fmt_mmq && timeout 1200 ./target/release/fmt_mmq --fmt=iq4_xs > $O/gate-fm-iq4xs.log 2>&1); GF=$?; echo "fmt_mmq iq4_xs gate rc=$GF"
grep -E "mutation|launches|FAIL|PASS|failing" $O/gate-fm-iq4xs.log | tail -8

echo "== [S] stage PTX ($(el))"
cp $OX/iq4_xs/iq4_xs.ptx $E/top-iq4xs/candle/candle-core/src/quantized/iq4_xs_oxide.ptx
(cd $OX/fmt_mmq && python3 export_ptx.py $SRC)
for f in iq4_nl mxfp4 nvfp4; do
  p=mistralrs-quant/src/gguf/${f}_mmq_oxide.ptx
  if git -C $SRC diff --quiet -- $p; then echo "  $p: byte-identical to committed"; else
    python3 $I/ptxcmp.py <(git -C $SRC show HEAD:$p) $SRC/$p; echo "  $p differs from committed: restoring the committed file"; git -C $SRC checkout -- $p; fi
done
python3 $I/ptxcmp.py $OX/fmt_mmq/fmt_mmq.ptx $SRC/mistralrs-quant/src/gguf/iq4_xs_mmq_oxide.ptx | head -1
[ $GX = 0 ] && [ $GF = 0 ] || echo "WARNING: a kernel gate did not pass; staged anyway (build + e2e still informative)"

echo "== [B] mistral.rs build ($(el))"
until mbuild 1500; do waitfix mbuild 1500 || exit 1; done
cp $MBIN $NEWBIN; BIN=$NEWBIN; echo "binary: $BIN sha256 $(sha256sum $BIN | cut -d' ' -f1)"
grep -E "^warning: unused|^error" $O/build-$WN.log | sort | uniq -c | head

e2e() { # TAG FILE [skip-llama]
  local T=$1 F=$2
  if [ "${3:-}" != skip-llama ]; then
    echo "-- e2e $T: llama.cpp ($(el))"
    systemctl --user reset-failed iq4xs-llama 2>/dev/null
    systemd-run --user --unit=iq4xs-llama --collect -q -p MemoryMax=16G -p MemorySwapMax=0 -p StandardOutput=truncate:$O/${T}_llama.log -p StandardError=truncate:$O/${T}_llama.log \
      $HOME/ai/llama.cpp/build/bin/llama-server -m $F -ngl 99 -c 4096 --port $LP --temp 0 --top-k 1 -np 1
    for i in $(seq 1 200); do curl -sf -m 2 -o /dev/null localhost:$LP/health && break; systemctl --user -q is-active iq4xs-llama || break; sleep 2; done
    (cd $I && timeout 900 python3 e2e.py llama $LP $T); systemctl --user stop iq4xs-llama; sleep 2
  fi
  echo "-- e2e $T: mistral.rs ($(el))"
  if srv_start e2e-$T $(dirname $F) $(basename $F) "${E2E_ENV:-}" --max-seq-len 4096; then
    (cd $I && timeout 1200 python3 e2e.py mistral $PORT $T); srv_stop
  fi
  (cd $I && python3 e2e.py cmp $T $F | tail -5)
}
echo "== [E] 9B IQ4_XS e2e vs llama.cpp ($(el))"
Q9=$M/iq4xs-e2e/Qwen3.5-9B-IQ4_XS.gguf
e2e iq4xs $Q9
grep -E "IQ4_XS|iq4_xs|unsupported|panicked" $O/e2e-iq4xs.server.log | grep -v " DEBUG " | head -5 | cut -c1-200

echo "== [R1] 35B gate pair ($(el))"
coll iq4xs-m2 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2 && cmp40 $E/m6/out/h-off.json $E/m3/out/iq4xs-m2.json
coll iq4xs-mf0 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=0 && cmp40 $E/m6/out/mtpfile-off.json $E/m3/out/iq4xs-mf0.json
echo "== [R2] Bonsai-2 B2-off, IQ2_M, Bonsai-27B Q1_0 vs the deployed binary's recorded runs (integ3 = bin/mistralrs-titan-swap) ($(el))"
if srv_start b-off $M/bonsai2-27b Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf "TITAN_MTP=0 TITAN_REASONING_EFFORT=medium" --max-seq-len 65536; then
  $C $CL greedy titan $PORT b-off 8 300; srv_stop
fi
$C $CL cmp o-off b-off
if srv_start i-new $M/q35-lowbit Qwen3.6-35B-A3B-UD-IQ2_M.gguf "TITAN_MTP=2 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64" --max-seq-len 65536; then
  $C $CL greedy titan $PORT i-new 8 300; srv_stop
fi
$C $CL cmp i-dep i-new
if srv_start q-new $M/bonsai/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf "TITAN_ATTN_FLASH_PREFILL=1" --max-seq-len 32768; then
  $C $CL greedy titan $PORT q-new 8 300; srv_stop
fi
$C $CL cmp q-dep q-new

echo "== [E2] Q8_0 control (llama.cpp side reused from m4/iq3s: same file, same llama.cpp build) ($(el))"
[ $(left) -gt 420 ] && e2e q8ctl $M/Qwen3.5-9B-Claude-Distill-v2-Q8_0.gguf skip-llama
free -m | head -2
echo "== iq4xs-window1 end $(el)"
