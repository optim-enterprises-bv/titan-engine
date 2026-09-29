# new.sh TAG "ENV": the new binary (target-fa-oxide) - logprobs at 13k / 28k with MTP=2 and MTP off, decode tok/s
# at ~3.3k / 13k / 28k with MTP=2 and MTP off (prefix cache off), then the comparisons.
[ -f $HOME/titan-engine/m4/fa/ctl/build.ok ] || { echo "no fresh build (ctl/build.ok): skipped"; exit 3; }
W=$HOME/titan-engine/m4/fa; S13=$HOME/titan-engine/m4/pc/sys13k.txt; S28=$HOME/titan-engine/m4/pc/sys28k.txt; D=$HOME/titan-engine/m4/kv/decode.py
export BIN=$HOME/titan-engine/target-fa-oxide/release/mistralrs
T=$1; X=$2; cd $W; rm -f out/lp-$T-m2.jsonl out/lp-$T-m0.jsonl
bash serve.sh $T-m2 "TITAN_PREFIX_CACHE=0 $X" \
  "python3 \$W/lp.py \$P $S13 0 \$W/out/lp-$T-m2.jsonl 13k 64" "python3 \$W/lp.py \$P $S28 0 \$W/out/lp-$T-m2.jsonl 28k 64" \
  "python3 $D \$P $S13 13600 \$O 4k 256" "python3 $D \$P $S13 0 \$O 13k 256" "python3 $D \$P $S28 0 \$O 28k 256"
bash serve.sh $T-m0 "TITAN_PREFIX_CACHE=0 TITAN_MTP=0 $X" \
  "python3 \$W/lp.py \$P $S13 0 \$W/out/lp-$T-m0.jsonl 13k 64" "python3 \$W/lp.py \$P $S28 0 \$W/out/lp-$T-m0.jsonl 28k 64" \
  "python3 $D \$P $S13 13600 \$O 4k 256" "python3 $D \$P $S13 0 \$O 13k 256" "python3 $D \$P $S28 0 \$O 28k 256"
python3 cmp_lp.py out/lp-old.jsonl out/lp-$T-m2.jsonl "new ($T) vs old, MTP=2"
python3 cmp_lp.py out/lp-$T-m0.jsonl out/lp-$T-m2.jsonl "new ($T) MTP off vs MTP=2"
