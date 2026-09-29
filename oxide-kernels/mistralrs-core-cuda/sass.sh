#!/bin/sh
# sass.sh <module> <mangled-or-substring>: concise SASS of one reference function
R=$HOME/titan-engine/oxide-kernels/reference/mistralrs-core
T=$HOME/titan-engine/oxide-kernels/tools/cuobjdump
f=$($T -sass $R/$1.cubin | grep 'Function :' | awk '{print $3}' | grep -- "$2" | head -1)
$T -sass -fun "$f" $R/$1.cubin | grep -E '^\s+/\*[0-9a-f]{4,}\*/' | sed -E 's@/\* 0x[0-9a-f]+ \*/@@; s/ +;/;/; s/^\s+//'
