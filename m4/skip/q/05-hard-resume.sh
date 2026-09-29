. ~/titan-engine/m4/skip/lib.sh
# window 3: resume the HumanEval + GSM8K runs (hardgen.py skips finished items)
ls -la $BIN
hard hbase 1700 $FIN $PROF
hard hs08 1250 $FIN $PROF TITAN_TIERED_SKIP_MISS_BELOW=0.08
