# gate 4: 80B 8-prompt vs m4/next/out/w2-8-final.json; gpt-oss-20b stock + tiered 50% vs m5/oss (fafce74). Args: BIN tag
E=$HOME/titan-engine; M=$HOME/ai/models; BIN=${1:-$E/target-094-oxide/release/mistralrs}; T=${2:-g4}
F80=$M/qwen3-next-80b/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf; G20=$M/gpt-oss-20b-F16.gguf
cd $E/m5/oss
BIN=$BIN timeout 700 ./pair.sh $G20 0 $T-oss20-stock "$T-oss20-t50 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=0.5"
python3 -c "
import json
s=json.load(open('out/$T-oss20-stock.json')); t=json.load(open('out/$T-oss20-t50.json'))
rs=json.load(open('out/oss20-stock.json')); rt=json.load(open('out/oss20-t50.json')); n=len(rs)
print(f'GATE oss20: stock vs t50 {sum(a==b for a,b in zip(s,t))}/{n}; stock vs fafce74 {sum(a==b for a,b in zip(s,rs))}/{n}; t50 vs fafce74 {sum(a==b for a,b in zip(t,rt))}/{n} (len {len(s)},{len(t)})')"
python3 cmp.py $T-oss20-stock oss20-stock 2>&1 | head -3
