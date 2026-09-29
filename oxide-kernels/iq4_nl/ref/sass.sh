#!/bin/bash
# sass.sh <cubin> <grep-pattern>: SASS of the first kernel whose name matches
T=$HOME/titan-engine/oxide-kernels/tools/cuobjdump
n=$($T -symbols "$1" | grep -oE "_Z[^ ]*" | grep -E -- "$2" | head -1)
echo "== $n"
$T -sass -fun "$n" "$1" | grep -E "^\s+/\*[0-9a-f]{4}\*/" | sed 's@/\* 0x[0-9a-f]* \*/@@'
