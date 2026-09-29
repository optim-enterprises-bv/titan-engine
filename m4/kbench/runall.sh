#!/bin/bash
# All benchmark groups, one process each (a crash loses one group, not the run). Run under nsys.
K=$HOME/titan-engine/m4/kbench; O=$K/out
cd $K
g() { local name=$1; shift; echo "-- $name $(date +%T)"; timeout 600 ./kbench "$@" > $O/$name.tsv 2> $O/$name.log; echo "   rc=$? $(grep -c ^RES $O/$name.tsv) rows"; }
g q35gu   q35.exp_gate_up.q4_K
g q35down q35.exp_down
g q35ffn  q35.ffn_fused
g q35d1   q35.qkv q35.attn_gate
g q35d2   q35.ssm_out q35.shexp
g q35lm   q35.lm_head
g q35mx   q35mx.
g iq4nl   iq4_nl
g q80gu   q80.exp_gate_up
g q80down q80.exp_down
