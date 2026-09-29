trap 'systemctl --user start titan-mistral' EXIT
systemctl --user stop titan-mistral
# CUDA graphs step 1 (measure only): nsys traces of the 35B service config, MTP off and MTP=2, plus
# no-nsys baselines to calibrate CUPTI overhead. Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/graph/window1.sh
trap 'exit 130' INT TERM HUP
set -u
E=$HOME/titan-engine; G=$E/m4/graph
exec > >(tee -a $G/window1-$(date +%Y%m%d-%H%M).log) 2>&1
T0=$(date +%s); el() { echo "$(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s"; }
echo "== window1 start $(date -Is), titan-mistral: $(systemctl --user is-active titan-mistral)"
for p in $(ps -eo pid=,ppid=,comm= | awk '$2 == 878468 && $3 ~ /rust-analyzer/ {print $1}'); do echo "killing rust-analyzer $p"; kill $p; done
BIN=${BIN:-$E/target-oss-oxide/release/mistralrs}
ls -la --time-style=full-iso $BIN; free -m | head -2
P=$E/m4/prompts-eval.txt
run() { # run NAME NSYS NPROMPTS ENV...
  local n=$1 ns=$2 np=$3; shift 3
  timeout 600 systemd-run --user --unit=graph-$n --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=570 \
    --setenv=NSYS=$ns bash $G/bench.sh $n $BIN $P $np 64 "$@"
  echo "-- $n rc=$? ($(el))"
}
run base-off 0 8
run base-mtp2 0 8 TITAN_MTP=2
run nsys-off 1 4
run nsys-mtp2 1 4 TITAN_MTP=2
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "== window1 end $(el)"
