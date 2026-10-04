#!/bin/bash
# tmcmp.sh: map titan token ids to pieces (copies in out/tm) and score G1/G3 against llama.cpp references (CPU only)
cd $(dirname $0)/out/tm
P=$HOME/ai/convert-env/bin/python3; M4=$HOME/titan-engine/m4
run() { systemd-run --user --collect --wait --pipe -q -p MemoryMax=4G -p WorkingDirectory=$PWD "$@"; }
tm() { # KIND FILES...: spark|gemma4|qwen3|gguf:PATH
  local k=$1; shift; local fs=(); for f in "$@"; do [ -e $f ] && fs+=($PWD/$f); done; [ ${#fs[@]} = 0 ] && return
  case $k in
    spark|gemma4) run $P $M4/g4s/tokmap.py $k "${fs[@]}" ;;
    qwen3) run $P $M4/devmap/tokmap_qwen3.py "${fs[@]}" ;;
    gguf:*) run $P $M4/orca/tokmap_gguf.py ${k#gguf:} "${fs[@]}" ;;
  esac > /dev/null
}
cmpg() { [ -e $2 ] && [ -e $3 ] && python3 $M4/g4s/cmp.py $1 $2 $3 | head -1 | cut -c1-170; }
"$@"
