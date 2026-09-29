# flash-decode window 3: interleaved A/B of the deployed binary and the candidate on the bench's e2e tier
# (service + mtpoff plans: short-context decode, 4k/13k prefill, identity), old/new/old/new.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m4/fa/window3.sh
set -u
E=$HOME/titan-engine; W=$E/m4/fa
exec > >(tee -a $W/window3-$(date +%Y%m%d-%H%M).log) 2>&1
echo "== lock acquired $(date -Is)"
trap 'systemctl --user start titan-mistral; echo "== titan-mistral restarted $(date -Is)"' EXIT
trap 'exit 130' INT TERM HUP
systemctl --user stop titan-mistral
for p in $(ps -eo pid=,ppid=,comm= | awk '$2 == 878468 && $3 ~ /rust-analyzer/ {print $1}'); do kill $p; done
OLD=$E/bin/mistralrs-titan-kv; NEW=$W/mistralrs-fa-ad65365
for i in 1 2; do
  BENCH_SKIP=kernel,ident,nsys2,nsys0 timeout 900 bash $E/bench/run.sh $OLD fa-ab-old$i | grep -E "8x256|cold|CONTAM|identity" 
  BENCH_SRC=$E/mr-fa BENCH_SKIP=kernel,ident,nsys2,nsys0 timeout 900 bash $E/bench/run.sh $NEW fa-ab-new$i | grep -E "8x256|cold|CONTAM|identity"
done
echo "== window end $(date -Is)"
