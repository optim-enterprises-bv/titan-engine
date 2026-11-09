#!/bin/bash
# Per-kernel GPU time for one decode step: nsys profile, then a cuda_gpu_kern_sum report.
#   ./prof-kernels.sh [max_tokens, default 128]
# Prod (titan-swap) holds :1234 and the GPU, so it is stopped for the profile and restored on exit.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO/bench-lib.sh"
OUT="${OUT:-/tmp/audit}"
BIN="${TITAN_BIN:-/usr/local/bin/mistralrs}"
CTX="${CTX:-65536}"
TOK="${1:-128}"

export TITAN_MTP=0 TITAN_GDN_DIRECT=1 TITAN_Q3K_ROWS=1

systemctl --user stop titan-swap
sleep 2
rm -rf "${OUT}.nsys-rep" "${OUT}.sqlite"

nsys profile -f true --trace=cuda -o "$OUT" --sample=none \
  "$BIN" serve -p 1234 --no-ui --paged-attn off --max-seq-len "$CTX" --format gguf \
  -m "$REPO/models-optim" -f Qwen3.8-27B-DevSecOps-QLoRA-abliterated-Q3_K_M.gguf -n 0:65 \
  >/tmp/audit.log 2>&1 &
NPID=$!

for i in $(seq 1 40); do
  sleep 3
  ss -tln | grep -q ':1234' && break
done
"$REPO/decode-once.py" 8 >/dev/null 2>&1
"$REPO/decode-once.py" "$TOK"
sleep 1

nsys stats --report cuda_gpu_kern_sum --format csv "${OUT}.nsys-rep" 2>/dev/null >/tmp/audit-kern.csv
echo "kernel rows: $(wc -l </tmp/audit-kern.csv)"
