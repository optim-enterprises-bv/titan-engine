source $(dirname $0)/tmcmp.sh true
tm spark new-spark-x2.5-g1.json old-spark-x2.5-g1.json
tm gemma4 new-redcell-26b-g1.json old-redcell-26b-g1.json new-redcell-26b-g3.json old-redcell-26b-g3.json
tm qwen3 new-qwen3-14b-g1.json old-qwen3-14b-g1.json
tm gguf:$HOME/ai/models/orcasaq2-cyber-27b/OrcaSAQ-2-27B-Uncensored.gguf new-orcasaq2-cyber-27b-g1.json old-orcasaq2-cyber-27b-g1.json
for t in new old; do echo "== $t"
  cmpg g1 $t-spark-x2.5-g1.json $M4/g4s/out/refc-spark-g1.json
  cmpg g1 $t-qwen3-14b-g1.json $M4/devmap/out/lref-g1.json
  cmpg g1 $t-redcell-26b-g1.json $M4/g4s/out/ref-red-g1.json
  cmpg g1 $t-orcasaq2-cyber-27b-g1.json $M4/orca/out/ref-orca-g1.json
  cmpg g3 $t-redcell-26b-g3.json $M4/redcell2/out/l-g3.json
done
