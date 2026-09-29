trap 'systemctl --user start titan-mistral' EXIT
systemctl --user stop titan-mistral
# upstream mistral.rs v0.9.4 bench, window 2 (no build). Window 1 found: auto map at 16k "does not fit";
# at 4k it maps layers 0-15 GPU / 16-39 CPU, then every prompt fails: "moe experts forward: dtype mismatch in
# matmul, lhs: BF16, rhs: F32" (CPU GGUF experts). Workaround ladder: --dtype f32, paged off, explicit -n.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/sync/window-up2.sh
trap 'exit 130' INT TERM HUP
set -u
E=$HOME/titan-engine; S=$E/sync; M=$HOME/ai/models; F=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
exec > >(tee -a $S/window-up2-$(date +%Y%m%d-%H%M).log) 2>&1
T0=$(date +%s); DEADLINE=$((T0 + 57 * 60))
el() { echo "$(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
echo "== window-up2 start $(date -Is), titan-mistral: $(systemctl --user is-active titan-mistral)"
for p in $(ps -eo pid=,ppid=,comm= | awk '$2 == 878468 && $3 ~ /rust-analyzer/ {print $1}'); do echo "killing rust-analyzer $p"; kill $p; done
free -m | head -2
BIN=$E/target-upstream/release/mistralrs; $BIN --version || exit 1
P=$E/m4/prompts-eval.txt
cmp() { python3 -c "import json,sys;a,b=json.load(open(sys.argv[1])),json.load(open(sys.argv[2]));print('IDENT',sys.argv[2].split('/')[-1],'vs',sys.argv[1].split('/')[-1],':',sum(x==y for x,y in zip(a,b)),'/',len(b),'completed')" $1 $2; }
ok() { python3 -c "import json,sys;r=json.load(open(sys.argv[1]));sys.exit(0 if any(x is not None for x in r) else 1)" $S/out/$1.json 2>/dev/null; }
run() { # run NAME BUDGET_SECS [ENV=..]  (env SFLAGS/NPROMPTS/BIG from caller)
  local n=$1 budget=$2; shift 2
  [ $(left) -gt 240 ] || { echo "skip $n: $(left)s left"; return 1; }
  [ $budget -gt $(( $(left) - 60 )) ] && budget=$(( $(left) - 60 ))
  local to=$(( budget + 120 ))
  echo "== $n ($(el)), budget ${budget}s"
  rm -f $S/out/$n.json
  timeout $to systemd-run --user --unit=up-$n --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=$((to-10)) \
    --setenv=BIN=$BIN --setenv=DIR=$M/Qwen3.6-35B-A3B-MTP --setenv=FILE=$F "--setenv=SFLAGS=$SFLAGS" \
    --setenv=NPROMPTS=${NPROMPTS:-40} --setenv=BIG=${BIG:-0} --setenv=UB_DEADLINE=$(( $(date +%s) + budget - (${BIG:-0} * 480) )) \
    bash $S/collect-up.sh $n $P 256 "$@" 2>&1 | grep -v " INFO "
  systemctl --user stop up-$n 2>/dev/null
  [ -f $S/out/$n.json ] && cmp $E/m3/out/q35-prof.json $S/out/$n.json
  echo "-- $n done ($(el))"
  ok $n
}
B="--dtype f32"
# 1: auto device map (4k sizing), f32 activations, paged attn auto, graphs default
if SFLAGS="$B --max-seq-len 4096 --pa-context-len 16384" NPROMPTS=8 run upE-auto 420; then BASE="$B"; PA="--pa-context-len 16384"
elif SFLAGS="$B --max-seq-len 4096 --paged-attn off" NPROMPTS=8 run upE-auto-nopa 420; then BASE="$B"; PA="--paged-attn off"
elif SFLAGS="--dtype f16 --max-seq-len 4096 --paged-attn off" NPROMPTS=8 run upE-auto-f16 420; then BASE="--dtype f16"; PA="--paged-attn off"
else echo "UPSTREAM CANNOT SERVE THIS MODEL WITH CPU OFFLOAD (all dtype/paged variants failed)"; exit 0; fi
echo "working base: $BASE $PA"
# 2: explicit layer split, more layers on GPU (auto map over-estimates), full 40 + 4k/13k prefill and repeats
if SFLAGS="$BASE -n 0:24 $PA" BIG=1 run upF-n24 1500; then N="-n 0:24"
elif SFLAGS="$BASE -n 0:20 $PA" BIG=1 run upF-n20 1200; then N="-n 0:20"
else N="--max-seq-len 4096"; fi
echo "best split: $N"
# 3: CUDA graphs off (only meaningful with paged attn)
case "$PA" in *off*) echo "graphs need paged attn: skipping graphs-off";; *) SFLAGS="$BASE $N $PA" NPROMPTS=10 run upG-nograph 420 MISTRALRS_CUDA_GRAPHS=0;; esac
# 4: upstream MTP (built-in head) from the GGUF
SFLAGS="$BASE $N $PA --mtp --mtp-n-predict 2" NPROMPTS=10 run upH-mtp2 420
# 5: paged attn off (sequence-level prefix cache) prefill + repeats, if the base used paged attn
case "$PA" in *off*) ;; *) SFLAGS="$BASE $N --paged-attn off" NPROMPTS=3 BIG=1 run upI-nopa 660;; esac
free -m | head -2
echo "== window-up2 end $(el)"
