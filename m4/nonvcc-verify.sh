#!/bin/bash
E=$HOME/titan-engine; export BIN=$E/target-oxide/release/mistralrs; cd $E/m3
t0=$(date +%s)
systemd-run --user --unit=titan-nv24 --collect --wait -q -p MemoryMax=10500M -p MemorySwapMax=0 -p WorkingDirectory=$E/m3 --setenv=BIN=$BIN -p StandardOutput=truncate:$E/m4/nv24.log -p StandardError=truncate:$E/m4/nv24.log \
  ./collect.sh nv-eval prompts-eval.txt 256 TITAN_TIERED=1 TITAN_TIERED_TRACE=/dev/null
echo "24L run (incl. cold PTX JIT) wall $(( $(date +%s) - t0 ))s"; grep -v INFO $E/m4/nv24.log | grep -E "tok/s|died"
python3 -c "import json;a,b=json.load(open('out/eval.json')),json.load(open('out/nv-eval.json'));print('24L: nvcc build vs nvcc-free build:',sum(x==y for x,y in zip(a,b)),'/',len(a))"
systemd-run --user --unit=titan-nv35 --collect --wait -q -p MemoryMax=16G -p MemorySwapMax=0 -p WorkingDirectory=$E/m3 --setenv=BIN=$BIN -p StandardOutput=truncate:$E/m4/nv35.log -p StandardError=truncate:$E/m4/nv35.log \
  env DIR=$HOME/ai/models FILE=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf ./collect.sh nv-q35 $E/m4/prompts-eval.txt 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt
grep -v INFO $E/m4/nv35.log | grep -E "tok/s|died"
python3 -c "import json;a,b=json.load(open('out/q35-prof.json')),json.load(open('out/nv-q35.json'));print('35B: nvcc build vs nvcc-free build:',sum(x==y for x,y in zip(a,b)),'/',len(a))"
