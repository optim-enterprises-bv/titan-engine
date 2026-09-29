. ~/titan-engine/m5/oss/lib.sh
# gate 4: the 35B baseline, 40 prompts x 256 tokens, must stay identical to m3/out/q35-prof.json
timeout $(cap 900) systemd-run --user --unit=oss-q35 --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 \
  --setenv=BIN=$BIN --setenv=DIR=$M --setenv=FILE=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  bash $E/m3/collect.sh q35-oss $E/m4/prompts-eval.txt 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto \
  TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt 2>&1 | grep -v " INFO "
(cd $E/m3 && python3 -c "import json;a,b=json.load(open('out/q35-prof.json')),json.load(open('out/q35-oss.json'));print('q35-oss identical to q35-prof:',sum(x==y for x,y in zip(a,b)),'/',len(a))")
