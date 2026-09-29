#!/bin/bash
# fm long-context decode: 13k (m4/pc/sys13k.txt, ~10.9k tok) and 27.7k (m4/pc/sys28k.txt) prompts, 256 decode
# tokens, MTP=2, service config, prefix cache off (each request is cold). Reuses m4/fa/serve.sh + m4/kv/decode.py.
# Args: TAG BIN
set -u
E=$HOME/titan-engine; FA=$E/m4/fa; S13=$E/m4/pc/sys13k.txt; S28=$E/m4/pc/sys28k.txt; D=$E/m4/kv/decode.py
T=$1; export BIN=$2
mkdir -p $E/m4/fm/out
cd $FA
rm -f $FA/out/fm-$T.jsonl
bash serve.sh fm-$T "TITAN_PREFIX_CACHE=0" \
  "python3 $D \$P $S13 0 \$O 13k-$T 256" "python3 $D \$P $S28 0 \$O 28k-$T 256"
cp $FA/out/fm-$T.jsonl $E/m4/fm/out/longctx-$T.jsonl
cat $E/m4/fm/out/longctx-$T.jsonl
