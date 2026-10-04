source $(dirname $0)/tmcmp.sh true
for f in "$@"; do cp ../$f.json . ; done
tm gemma4 $(for f in "$@"; do echo $f.json; done)
for f in "$@"; do g=${f##*-}; r=$M4/g4s/out/ref-red-g1.json; [ $g = g3 ] && r=$M4/redcell2/out/l-g3.json; echo "$f vs llama: $(cmpg $g $f.json $r)"; o=old-redcell-26b-$g.json; echo "$f vs deployed: $(python3 $M4/g4s/cmp.py $g $f.json $o | head -1 | cut -c1-120)"; done
