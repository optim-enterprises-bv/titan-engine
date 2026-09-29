. ~/titan-engine/m5/oss/lib.sh
# gate 2: mistral.rs tiered, host experts in the GGUF mmap, GPU share from free VRAM
PRACTICE=1 PRACTICE_MAX=384 timeout $(cap 2400) ./pair.sh $G120 0 \
  "oss120-mmap TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_MMAP=1"
python3 cmp.py llama-120b oss120-mmap
python3 pcmp.py llama-120b oss120-mmap
