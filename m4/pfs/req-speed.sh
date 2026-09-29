# speed.sh TAG "ENV": cold prefill (prefix cache off) at ~4k, ~13k (opencode system text and bigprompt's text) and ~28k
W=$HOME/titan-engine/m4/pfs; S13=$HOME/titan-engine/m4/pc/sys13k.txt; S28=$HOME/titan-engine/m4/pc/sys28k.txt
bash $W/serve.sh $1 "TITAN_PREFIX_CACHE=0 $2" \
  "python3 \$W/prompt.py \$P $S13 13600 \$O 4k" \
  "python3 \$W/prompt.py \$P $S13 0 \$O 13k" \
  "python3 \$W/prompt.py \$P x big:13000 \$O big13k" \
  "python3 \$W/prompt.py \$P $S28 0 \$O 28k"
