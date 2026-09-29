#!/bin/sh
# usage: sass.sh <cubin-name> <function-substring>   (prints compact SASS of matching functions)
R=$HOME/titan-engine/oxide-kernels/reference/mistralrs-quant
T=$HOME/titan-engine/oxide-kernels/tools/cuobjdump
for f in $($T -sass $R/$1.cubin | grep 'Function :' | sed 's/.*Function : //' | grep -- "$2"); do
  echo "=== $f"
  $T -sass -fun "$f" $R/$1.cubin | grep -E '^\s+/\*[0-9a-f]{4,}\*/' | sed -E 's@^\s+/\*([0-9a-f]+)\*/\s+@\1 @; s@\s*;\s*/\*.*@@; s@\s+$@@'
done
