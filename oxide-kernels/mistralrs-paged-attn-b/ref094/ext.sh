#!/bin/bash
# ext.sh file.ptx mangled-substring -> the entry body (param names shortened)
f=$1; pat=$2
awk -v p="$pat" 'on&&(index($0,".entry")||index($0,"// .globl")){exit} index($0,".entry")&&index($0,p){on=1} on{print}' $f | grep -v "^\s*$\|// end inline\|// begin inline\|^\s*//" | sed -E 's/_ZN10flashinfer[A-Za-z0-9_]+_param_0/P/g'
