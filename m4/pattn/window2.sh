#!/bin/bash
# pattn window 2 (of 3): end to end with paged attention ON vs OFF on the same binary (bin/mistralrs-titan-pattn):
# Spark-X2.5 Q4_K_M (native spark2_5), qwen3-14b Q5_K_M (native qwen3), Qwen3.6-35B (titan qwen35 path; paged attention
# is disabled there, gguf_titan.rs). Per model: G1 (40 prompts, first-token top-20) on / off (+ the llama.cpp reference
# where one exists), a ~8k-token prompt (G3 at 28000 chars), prefill / decode tok/s (G5), VRAM; then the Spark
# concurrency probe (paged ON, --max-seqs 4 --max-batch-size 4: 4 concurrent requests vs 1, text vs sequential).
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/top-pattn/m4/pattn/window2.sh
source $HOME/titan-engine/top-pattn/m4/pattn/lib.sh
exec > >(tee -a $I/window2-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin pattn-window2 58
source $I/mut.sh
waitfix() { # TAG MIN_LEFT_S
  rm -f $I/fix.flag; date -Is > $O/$1-failed.flag; echo "!! $1 failed: waiting for $I/fix.flag ($(left)s left)"
  while [ ! -e $I/fix.flag ] && [ $(left) -gt $2 ]; do sleep 10; done
  rm -f $O/$1-failed.flag; [ -e $I/fix.flag ] && { rm -f $I/fix.flag; return 0; }; return 1
}
echo "src: mistral.rs $(git -C $SRC log --oneline -1 | cut -c1-60) $(git -C $SRC diff --quiet && echo clean || echo DIRTY); oxide $(git -C $OX log --oneline -1 | cut -c1-50) $(git -C $OX diff --quiet && echo clean || echo DIRTY)"
echo "== [K] crate a rebuild (v0.9.4 host error paths) + gate; mistral.rs rebuild in the background ($(el))"
cp $OX/mistralrs-paged-attn-a/mistralrs_paged_attn_a.ptx $O/w1-a.ptx
until oxbuild $OX mistralrs-paged-attn-a pattn-ox 900; do waitfix ox-a 1800 || exit 1; done
cmp $O/w1-a.ptx $OX/mistralrs-paged-attn-a/mistralrs_paged_attn_a.ptx && echo "crate a PTX unchanged by the host-side fix (byte-identical to window 1's)" || python3 $I/ptxkind.py $O/w1-a.ptx $OX/mistralrs-paged-attn-a/mistralrs_paged_attn_a.ptx
( mbuild 1800 > $O/mbuild2.status 2>&1 ) &
BP=$!
kgate $OX mistralrs-paged-attn-a $O/gate-a2.log 900
grep -E "exit case|family|FAIL" $O/gate-a2.log | head -20
echo "== [M] mutation checks redone (v_scale with the module loadable; noguard crash classification) ($(el))"
mutants_b_vscale
mut_prepare mistralrs-paged-attn-a; (cd $MX/mistralrs-paged-attn-a && GEN_HEADS=64 GEN_BLOCKS=16 python3 gen_kernels.py > /dev/null)
mut_apply mistralrs-paged-attn-a src/main.rs "        if batch_id >= num_seqs {
            return;
        }
" ""
mut_run a-gather-noguard mistralrs-paged-attn-a MPA_ONLY=gather_kv_cache
mut_prepare mistralrs-paged-attn-a; (cd $MX/mistralrs-paged-attn-a && GEN_HEADS=64 GEN_BLOCKS=16 python3 gen_kernels.py > /dev/null)
mut_apply mistralrs-paged-attn-a src/launch.rs '"src/cuda/reshape_and_cache_kernel.cu", 139' '"src/cuda/reshape_and_cache_kernel.cu", 140'
mut_run a-exitline-084 mistralrs-paged-attn-a MPA_ONLY=exit
wait $BP; cat $O/mbuild2.status
until grep -q "Finished" $O/build-$WN.log && [ $MBIN -nt $OX/titan-oxide-ffi/src/paged_attn_a.rs ]; do
  waitfix mbuild 1500 || { echo "no binary: ending the window"; exit 1; }; mbuild 1500
done
cp $MBIN $NEWBIN; BIN=$NEWBIN; sha256sum $BIN | tee $NEWBIN.sha256
echo "binary: $BIN sha256 $(sha256sum $BIN | cut -d' ' -f1) from $(git -C $SRC log --oneline -1 | cut -c1-50) $(git -C $SRC diff --quiet && echo clean || echo DIRTY); deployed: $(sha256sum $OLDBIN | cut -c1-16)"
tmaps() { timeout 300 systemd-run --user --unit=pattn-tmap --collect --wait --pipe -q -p MemoryMax=4G $HOME/ai/convert-env/bin/python3 $G/tokmap.py spark "$@"; }
e2e() { # TAG DIR FILE "ENV" PROMPTS ARGS...: G1, G3 ~8k, G5 with paged attention on and off
  local t=$1 d=$2 f=$3 env=$4 p=$5; shift 5
  for pa in on off; do
    local extra=""; [ $pa = on ] && extra="--pa-context-len ${PACTX:-16384}"
    if PA=$pa srv_start $t-$pa $d $f "$env" "$@" $extra; then
      echo "VRAM $t-$pa after load: $(smi)"; palines $SLOG 6
      gate g1 $t-$pa-g1 $p | tail -1
      gate g3 $t-$pa-g3l 28000 | tail -1; echo "VRAM $t-$pa after the ~8k prompts: $(smi)"
      gate g5 $t-$pa-g5 8000 128 | tail -1
      srv_stop
    fi
  done
}
SP=$G/prompts/spark.json; QP=$E/m4/devmap/prompts/qwen3-14b.json

echo "== [S] Spark-X2.5 Q4_K_M ($(el))"
e2e s $M/spark-x2.5 Spark-X2.5-4B-Q4_K_M.gguf "" $SP --max-model-len 16384 --max-seq-len 16384
same $O/s-on-g1.json $O/s-off-g1.json; same $O/s-off-g1.json $O/dep-is-g1.json; same $O/s-on-g3l.json $O/s-off-g3l.json
for f in s-on-g1 s-off-g1 s-on-g3l s-off-g3l; do cp $O/$f.json $O/$f.ids.json; done
tmaps $O/s-on-g1.json $O/s-off-g1.json $O/s-on-g3l.json $O/s-off-g3l.json
cmpg g1 $O/s-on-g1.json $O/s-off-g1.json; for pa in on off; do cmpg g1 $O/s-$pa-g1.json $G/out/refc-spark-g1.json; done
cmpg g3 $O/s-on-g3l.json $O/s-off-g3l.json
tps $O/s-on-g5.json $O/s-off-g5.json $O/s-on-g3l.json $O/s-off-g3l.json

echo "== [C] Spark concurrency: --max-seqs 4 --max-batch-size 4, 4 concurrent vs sequential ($(el))"
for pa in on off; do
  extra=""; [ $pa = on ] && extra="--pa-context-len 16384"
  if PA=$pa srv_start c-$pa $M/spark-x2.5 Spark-X2.5-4B-Q4_K_M.gguf "" --max-model-len 16384 --max-seq-len 16384 --max-seqs 4 --max-batch-size 4 $extra; then
    echo "VRAM c-$pa after load: $(smi)"
    conc $SP $O/c-$pa-conc.json 4 128 0
    conc $SP $O/c-$pa-conc2.json 4 256 8
    echo "VRAM c-$pa after: $(smi)"; srv_stop
  fi
done

echo "== [Q] qwen3-14b Q5_K_M ($(el))"
e2e q $M/qwen3-14b Qwen3-14B-vanilla-Q5_K_M.gguf "TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64" $QP --max-seq-len 16384 --prefix-cache-n 0
for pa in on off; do cp $O/q-$pa-g1.json $O/q-$pa-g1.ids.json 2>/dev/null; done
same $O/q-on-g1.json $O/q-off-g1.json; same $O/q-on-g3l.json $O/q-off-g3l.json
tmapq $O/q-on-g1.json $O/q-off-g1.json $O/q-on-g3l.json $O/q-off-g3l.json
cmpg g1 $O/q-on-g1.json $O/q-off-g1.json; for pa in on off; do cmpg g1 $O/q-$pa-g1.json $E/m4/devmap/out/lref-g1.json; done
cmpg g3 $O/q-on-g3l.json $O/q-off-g3l.json; for pa in on off; do cmpg g3 $O/q-$pa-g3l.json $E/m4/devmap/out/lref-g3l.json; done
tps $O/q-on-g5.json $O/q-off-g5.json $O/q-on-g3l.json $O/q-off-g3l.json

echo "== [T] Qwen3.6-35B (titan qwen35 path, deployed env) ($(el))"
QE="TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt TITAN_TIERED_RESERVE_MIB=1536 TITAN_MTP=2 TITAN_PFS=1 TITAN_PFS_HEADROOM_MIB=768 TITAN_DOORBELL=1 TITAN_CPU_ONEPASS=1 TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64"
e2e t $M/Qwen3.6-35B-A3B-MTP Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf "$QE" $QP --max-seq-len 65536
grep -h "without PagedAttention" $O/t-on.server.log | sed 's/\x1b\[[0-9;]*m//g' | sed 's/^.*WARN [a-z_:0-9]*: //' | head -1
same $O/t-on-g1.json $O/t-off-g1.json; same $O/t-on-g3l.json $O/t-off-g3l.json
tps $O/t-on-g5.json $O/t-off-g5.json
free -m | head -2
echo "== pattn-window2 end $(el)"
