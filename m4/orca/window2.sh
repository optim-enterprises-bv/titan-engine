#!/bin/bash
# orca window 2 (of 3), binary bin/mistralrs-titan-orca from window 1 (unchanged unless the fix loop rebuilds it):
# [L] llama.cpp OrcaSAQ references: G3 / G3l reference (-c 12288, f16 KV, default batching), its self-spread
#     (m4/g4acc/g3spread.sh: -ub 256 / -ub 128 / -b 512), greedy 8 x 300 decode speed, a context probe;
# [P] titan prompt-length root cause: 2048-row big chunks (default) vs 512-row chunks (TITAN_PREFILL_BIG_CHUNK=0);
# [O] titan all-GPU at 12288, MTP 0 and 1: context ladder (each length twice), G1 vs llama, G3 / G3l vs the spread,
#     greedy 8 x 300 (decode tok/s, MTP acceptance); [E] 9B IQ4_XS e2e vs the iq4xs branch; [Q] qwen3-14b 16k + G1.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/top-orca/m4/orca/window2.sh
source $HOME/titan-engine/top-orca/m4/orca/lib.sh
exec > >(tee -a $I/window2-$(date +%Y%m%d-%H%M).log) 2>&1
window_begin orca-window2 58
BIN=$NEWBIN; echo "binary: $BIN sha256 $(sha256sum $BIN | cut -d' ' -f1); src $(git -C $SRC log --oneline -1 | cut -c1-50) $(git -C $SRC diff --quiet && echo clean || echo DIRTY)"
mkdir -p $O/spread

echo "== [L] llama.cpp references ($(el))"
if llama_srv lref $OD/$OF -c 12288 -ctk f16 -ctv f16; then
  timeout 600 python3 $G/gates.py g3 $LP llama $O/lref-g3.json | tail -1
  timeout 600 python3 $G/gates.py g3 $LP llama $O/lref-g3l.json 28000 | tail -1
  (cd $I && timeout 600 $C $CL greedy llama $LP lref-g 8 300 | tail -2)
  timeout 300 python3 $I/probe.py $LP $O/lref-ctx.json 1000,8000 64 | tail -2
  echo "llama VRAM with 12288 ctx after requests: $(smi)"; grep -E "CUDA0 (model|KV|compute) buffer|token_embd" $O/lref.llama.log | head -6
  systemctl --user stop orca-llama
fi
timeout 1500 bash $E/m4/g4acc/g3spread.sh $OD/$OF $O/spread
ls $O/spread/*.json 2>/dev/null | wc -l

echo "== [P] big chunks (2048 rows, default) vs 512-row chunks, all-GPU, 12288, MTP 0, prefix cache off ($(el))"
for v in "pb2048 TITAN_MTP=0" "pb512 TITAN_MTP=0 TITAN_PREFILL_BIG_CHUNK=0"; do set -- $v; n=$1; shift
  if srv_start $n $OD $OF "$* TITAN_DEVMAP_LOG=1" -n 0:64 --max-seq-len 12288 --prefix-cache-n 0; then
    PT=500 probe $O/$n.json 4000,8000 32; memlines $SLOG | grep -E "after prompt step \([0-9]{4}" | tail -2; srv_stop
  fi
done

orca() { # TAG MTP: all-GPU at 12288, 512-row chunks, prefix cache default
  local t=$1 m=$2
  echo "== [O] $t: MTP $m ($(el))"
  srv_start $t $OD $OF "TITAN_MTP=$m TITAN_PREFILL_BIG_CHUNK=0 TITAN_DEVMAP_LOG=1" -n 0:64 --max-seq-len 12288 || return 1
  PT=900 probe $O/$t-ctx.json 2000,4000,6000,8000,8000,10000,10000,12000 64
  gate g1 $t-g1 $GP
  gate g3 $t-g3; gate g3 $t-g3l 28000
  (cd $I && timeout 600 $C $CL greedy titan $PORT $t-g 8 300 | tail -2)
  echo "VRAM at the end: $(smi)"; srv_stop
  sed 's/\x1b\[[0-9;]*m//g' $SLOG | grep -E "titan mtp stats" | tail -1 | sed 's/^.*INFO [a-z_:0-9]*: //'
  echo "OOM recoveries: $(grep -c 'titan: CUDA out of memory' $SLOG), panics: $(grep -c panicked $SLOG)"
  tmo $O/$t-g1.json $O/$t-g3.json $O/$t-g3l.json > /dev/null
  cmpg g1 $O/$t-g1.json $O/ref-orca-g1.json | head -1
  for g in g3 g3l; do python3 $E/m4/g4acc/g3spread.py $O/lref-$g.json $O/spread/lub256-$g.json,$O/spread/lub128-$g.json,$O/spread/lb512-$g.json -- $O/$t-$g.json | tail -2; done
}
orca om0 0
orca om1 1
[ -e $O/om0-g1.json ] && [ -e $O/om1-g1.json ] && same $O/om0-g1.json $O/om1-g1.json

echo "== [E] 9B IQ4_XS e2e (mistral side; llama side = the iq4xs branch's out/iq4xs_llama.json) ($(el))"
if [ $(left) -gt 400 ] && srv_start e2e-orca $M/iq4xs-e2e Qwen3.5-9B-IQ4_XS.gguf "" --max-seq-len 4096; then
  cp $O/iq4xs_llama.json $O/orca9b_llama.json; (cd $I && timeout 600 python3 e2e.py mistral $PORT orca9b > /dev/null); srv_stop
  (cd $I && python3 e2e.py cmp orca9b $M/iq4xs-e2e/Qwen3.5-9B-IQ4_XS.gguf | tail -4)
  python3 -c "import json,sys;a,b=json.load(open(sys.argv[1])),json.load(open(sys.argv[2]));print('9B mistral side vs iq4xs branch window 3:', 'IDENTICAL' if a==b else 'DIFFERENT', len(a), len(b))" $O/orca9b_mistral.json $O/iq4xs3_mistral.json
fi
echo "== [Q] qwen3-14b, deploy settings (no pin, 16384, prefix cache off): 16k prompt + G1 vs the deployed binary ($(el))"
QM="$M/qwen3-14b Qwen3-14B-vanilla-Q5_K_M.gguf"; QE="TITAN_CUDA_GRAPHS=1 TITAN_CUDA_GRAPHS_RESERVE_MIB=64"
if [ $(left) -gt 300 ] && MEM=14G srv_start fq $QM "$QE TITAN_DEVMAP_LOG=1" --max-seq-len 16384 --prefix-cache-n 0; then
  maplines $SLOG | grep -q ": cpu" && echo "Q mapping: FAIL offloads" || echo "Q mapping: 40/40 on the GPU"
  probe $O/fq-short.json 50 256; probe $O/fq-long.json 16200 64; probe $O/fq-short2.json 50 256
  gate g1 fq-g1 $I/prompts/qwen3-14b.json; srv_stop
  tmapq $O/fq-g1.json > /dev/null; same $O/fq-g1.json $O/dep-fq-g1.json; cmpg g1 $O/fq-g1.json $O/lref-q3-g1.json | head -1
fi
free -m | head -2
echo "== orca-window2 end $(el)"
