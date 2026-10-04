source $(dirname $0)/tmcmp.sh true
for m in spark-x2.5 qwen3-14b gemma4-12b orcasaq2-cyber-27b bonsai-27b; do cp ../new2-$m-g3.json . ; cp ../new2-$m-g1.json . ; done
cp ../new2-redcell-26b-g1.json ../new2-redcell-26b-g3.json .
tm spark new2-spark-x2.5-g3.json new2-spark-x2.5-g1.json
tm qwen3 new2-qwen3-14b-g3.json new2-qwen3-14b-g1.json
tm gemma4 new2-gemma4-12b-g3.json new2-gemma4-12b-g1.json new2-redcell-26b-g1.json new2-redcell-26b-g3.json
tm gguf:$HOME/ai/models/orcasaq2-cyber-27b/OrcaSAQ-2-27B-Uncensored.gguf new2-orcasaq2-cyber-27b-g3.json new2-orcasaq2-cyber-27b-g1.json
python3 $(dirname $0)/../../sp.py new2-orcasaq2-cyber-27b-g3.json
S=$M4/g4acc/g3spread.py
SS=$HOME/titan-engine/top-pattn/m4/pattn/out/spread-spark
echo spark; python3 $S $SS/base-c13500.json $SS/lub256-c13500.json,$SS/lub128-c13500.json,$SS/lb512-c13500.json -- new2-spark-x2.5-g3.json | tail -1
echo qwen3-14b; D=$M4/devmap/out; python3 $S $D/lref-g3.json $D/spread/lub256-g3.json,$D/spread/lub128-g3.json,$D/spread/lb512-g3.json -- new2-qwen3-14b-g3.json | tail -1
echo gemma4; A=$M4/g4acc/out; python3 $S $M4/g4s/out/ref-g12-g3.json $A/lub256-g3.json,$A/lub128-g3.json,$A/lb512-g3.json -- new2-gemma4-12b-g3.json | tail -1
echo orca; R=$M4/orca/out; python3 $S $R/lref-g3.json $R/spread/lub256-g3.json,$R/spread/lub128-g3.json,$R/spread/lb512-g3.json -- new2-orcasaq2-cyber-27b-g3.json | tail -1
echo "gemma4 g1: $(cmpg g1 new2-gemma4-12b-g1.json $M4/g4s/out/ref-g12-g1.json)"
