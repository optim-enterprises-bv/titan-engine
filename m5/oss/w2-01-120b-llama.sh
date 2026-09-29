. ~/titan-engine/m5/oss/lib.sh
# gate 2 reference: llama.cpp with the most expert layers on the GPU that fit, then 8 prompts + top-10 + practice
grep -q "^done" $M/oss-dl.log && [ "$(stat -c %s $G120 2>/dev/null || echo 0)" -gt 63000000000 ] || { echo "120b download not complete"; exit 1; }
ls -la $G120
timeout $(cap 900) ./tune_ncm.sh $G120 30
NCM=$(cat out/ncm.$(basename $G120))
PRACTICE=1 PRACTICE_MAX=384 timeout $(cap 2400) ./pair.sh $G120 $NCM llama-120b
