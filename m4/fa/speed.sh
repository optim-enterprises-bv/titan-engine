# speed.sh TAG "ENV": new binary, decode tok/s (256 tokens, prefix cache off) at ~2.1k / ~3.3k / 13k / 28k, MTP=2 then MTP off
[ -f $HOME/titan-engine/m4/fa/ctl/build.ok ] || { echo "no fresh build (ctl/build.ok): skipped"; exit 3; }
W=$HOME/titan-engine/m4/fa; S13=$HOME/titan-engine/m4/pc/sys13k.txt; S28=$HOME/titan-engine/m4/pc/sys28k.txt; D=$HOME/titan-engine/m4/kv/decode.py
export BIN=${BIN:-$HOME/titan-engine/target-fa-oxide/release/mistralrs}; T=$1; X=$2; cd $W
for m in 2 0; do
bash serve.sh $T-m$m "TITAN_PREFIX_CACHE=0 TITAN_MTP=$m $X" \
  "python3 $D \$P $S13 8400 \$O 2k 256" "python3 $D \$P $S13 13600 \$O 4k 256" "python3 $D \$P $S13 0 \$O 13k 256" "python3 $D \$P $S28 0 \$O 28k 256"
done
