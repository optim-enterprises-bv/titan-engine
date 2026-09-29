# gates (d)+(e) on the service config + ENV: 3-turn live prefix-cache check (cold turn 1, warm turns 2-3), cached_tokens,
# then 3 x 13k back-to-back + a 27.7k prompt (no OOM). Args: TAG "ENV"
[ -f $HOME/titan-engine/m4/pfs/ctl/build.ok ] || { echo "no fresh build (ctl/build.ok): skipped"; exit 3; }
E=$HOME/titan-engine; W=$E/m4/pfs; T=$1
bash $W/serve.sh $T "$2" \
  "echo '== gate d: live 3-turn'; python3 \$E/sync/w094/live094.py \$P pfs-$T-live.json" \
  "python3 -c \"import json;r=json.load(open('\$E/m4/pc/out/pfs-$T-live.json'));print('turns (prompt tok, prompt s, wall s):',[(x['prompt_tokens'],round(x['prompt_time'],2),round(x['wall'],2)) for x in r])\"" \
  "echo '== gate e: 3 x 13k + 27.7k'" \
  "python3 \$E/m4/bigprompt.py \$P 13000" "python3 \$E/m4/bigprompt.py \$P 13000" "python3 \$E/m4/bigprompt.py \$P 13000" \
  "python3 \$W/prompt.py \$P \$E/m4/pc/sys28k.txt 0 \$O 28k" \
  "python3 \$W/prompt.py \$P \$E/m4/pc/sys13k.txt 0 \$O 13k-after"
