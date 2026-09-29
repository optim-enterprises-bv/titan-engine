#!/bin/bash
# Clean A/B on an idle titan: nvcc build vs nvcc-free build (GDN fix), same 40 prompts; then nsys on the nvcc-free build.
E=$HOME/titan-engine; cd $E/m3
for pair in "ab-nvcc:$E/mistral.rs/target/release/mistralrs" "ab-oxide:$E/target-oxide/release/mistralrs"; do
  n=${pair%%:*}; b=${pair#*:}
  echo "$n load before: $(cut -d' ' -f1 /proc/loadavg)"
  BIN=$b DIR=$HOME/ai/models FILE=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf ./collect.sh $n $E/m4/prompts-eval.txt 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt 2>&1 | grep -v INFO | grep tok/s
  python3 -c "import json;a,b=json.load(open('out/q35-prof.json')),json.load(open('out/$n.json'));print('  identical to nvcc baseline:',sum(x==y for x,y in zip(a,b)),'/',len(a))"
done
cd $E/m4/prof && ./prof.sh oxide-gdnfix $E/target-oxide/release/mistralrs | tail -1
python3 - <<'PY'
import csv
def load(f): return {r['Name']:float(r['Total Time (ns)']) for r in csv.DictReader(open(f))}
a=load('nvcc_kern_cuda_gpu_kern_sum.csv'); b=load('oxide-gdnfix_kern_cuda_gpu_kern_sum.csv')
print(f"GPU kernel total: nvcc {sum(a.values())/1e9:.2f}s  oxide(GDN fix) {sum(b.values())/1e9:.2f}s")
for k in ['gdn_decode_recurrence','moe_router_topk','gated_delta_rule_recurrence_kernel_tiled']:
    ta=sum(v for n,v in a.items() if k in n); tb=sum(v for n,v in b.items() if k in n)
    print(f"  {k}: nvcc {ta/1e6:.0f} ms  oxide {tb/1e6:.0f} ms")
PY
