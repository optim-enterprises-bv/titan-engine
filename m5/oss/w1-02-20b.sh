. ~/titan-engine/m5/oss/lib.sh
# gate 1: 20b all on GPU in both engines; mistral.rs stock (every expert in a GPU slot) and tiered with half on the CPU
PRACTICE=1 timeout $(cap 2400) ./pair.sh $G20 0 llama-20b oss20-stock "oss20-t50 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=0.5"
python3 cmp.py llama-20b oss20-stock
python3 cmp.py oss20-stock oss20-t50 | grep -v '^  [0-9] '
python3 pcmp.py llama-20b oss20-stock | tail -1
python3 pcmp.py oss20-stock oss20-t50 | tail -1
