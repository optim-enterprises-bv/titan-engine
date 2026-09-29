# 30-minute service gap (RULES-agents.md): after the flock, before stopping the service
while :; do t=$(systemctl --user show titan-mistral -p ActiveEnterTimestampMonotonic --value)
  now=$(awk '{print int($1*1000000)}' /proc/uptime); [ $(( (now - t) / 1000000 )) -ge $( [ -e $HOME/titan-engine/.nogap ] && echo 0 || echo 1800 ) ] && break; sleep 30; done
trap 'systemctl --user start titan-mistral' EXIT
systemctl --user stop titan-mistral
echo "== service stopped $(date -Is)"
# n-gram drafting on titan-094, window 2 (no build: target-ngram94-oxide from window 1, ngram-spec-094 1fcb4047b):
# (1) code bench ngram off (per-request timeout + diagnostics), (2) code bench agree,
# (3) 40-prompt MTP=2: off, agree, first+MIN=3 (gate 2 + gate 1 vs h-off), (4) code bench first+MIN=3 if time.
# Run as: flock ~/titan-engine/.gpu.lock bash ~/titan-engine/m6/ngram/window2.sh
trap 'exit 130' INT TERM HUP
set -u
E=$HOME/titan-engine; N=$E/m6/ngram; M=$HOME/ai/models; F=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
mkdir -p $N/out
exec > >(tee -a $N/window2-$(date +%Y%m%d-%H%M).log) 2>&1
T0=$(date +%s); DEADLINE=$((T0 + 57 * 60))
el() { echo "$(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
killra() { for p in $(ps -eo pid=,ppid=,comm= | awk '$2 == 878468 && $3 ~ /rust-analyzer/ {print $1}'); do echo "killing rust-analyzer $p"; kill $p; done; }
hoststate() { echo "-- host $(date -Is): $(uptime)"; free -m | head -2; ps -eo pid,rss,pcpu,etime,comm --sort=-rss | head -6; }
# load average > 8 or a non-titan process > 4 GB RSS: wait up to 10 minutes for it to settle
busy() {
  local l=$(awk '{print int($1)}' /proc/loadavg)
  local big=$(ps -eo rss=,comm= --sort=-rss | awk '$1 > 4194304 && $2 !~ /^mistralrs/ {print $2 "(" int($1/1024) "MB)"}' | tr '\n' ' ')
  [ $l -gt 8 ] || [ -n "$big" ] && { echo "busy: load1=$l big=[$big]"; return 0; }; return 1
}
settle() {
  killra; local w=0
  while busy && [ $w -lt 600 ]; do sleep 30; w=$((w + 30)); killra; done
  [ $w -gt 0 ] && echo "waited ${w}s for the host to settle"; busy && echo "host still busy after ${w}s, running anyway"; true
}
echo "== ngram window2 start $(date -Is), titan-mistral: $(systemctl --user is-active titan-mistral)"
hoststate; settle; hoststate
BIN=$E/target-ngram94-oxide/release/mistralrs
ls -la $BIN || exit 1
vmstat -t 30 > $N/out/w2-vmstat.log 2>&1 & VM=$!
trap 'kill $VM 2>/dev/null; exit 130' INT TERM HUP

P=$E/m4/prompts-eval.txt
coll() {
  local n=$1 dir=$2; shift 2
  [ $(left) -gt 330 ] || { echo "skip $n: $(left)s left"; return 1; }
  killra; echo "-- start $n: $(uptime)"
  timeout 600 systemd-run --user --unit=ngram-$n --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=580 \
    --setenv=BIN=$BIN --setenv=DIR=$dir --setenv=FILE=$F \
    bash $E/sync/w094/collect094.sh $n $P 256 TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt "$@" 2>&1 | grep -v " INFO "
  echo "-- $n ($(el))"; grep -o "titan ngram stats.*\|titan mtp stats.*\|panicked.*" $E/m3/out/$n.server.log | tail -2
  cp $E/m3/out/$n.json $E/m3/out/$n.server.log $N/out/ 2>/dev/null
}
cmp() { python3 -c "import json,sys;a,b=json.load(open(sys.argv[1])),json.load(open(sys.argv[2]));print('GATE',sys.argv[2].split('/')[-1],'vs',sys.argv[1].split('/')[-1],':',sum(x==y for x,y in zip(a,b)),'/',len(a),'len',len(b))" $1 $2; }
code() { # code NAME ENV...  -- service config (MTP=2, tiered, 64k, paged off), the 20-prompt code-editing bench
  local n=$1; shift
  [ $(left) -gt 960 ] || { echo "skip $n: $(left)s left"; return 1; }
  killra; echo "-- start $n: $(uptime)"
  timeout 960 systemd-run --user --unit=ngram-$n --collect --wait --pipe -q -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=940 bash -c "
    cd $N; env LD_LIBRARY_PATH=$E/lib TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=$E/m4/profile-q35.txt TITAN_MTP=2 $* \
      $BIN --seed 0 serve -p 18493 --no-ui --paged-attn off --max-seq-len 65536 --format gguf -m $M/Qwen3.6-35B-A3B-MTP -f $F > out/$n.server.log 2>&1 &
    pid=\$!; for i in \$(seq 1 200); do curl -sf -m 2 -o /dev/null localhost:18493/v1/models && break; kill -0 \$pid 2>/dev/null || break; sleep 2; done
    SERVER_PID=\$pid REQ_TIMEOUT=240 python3 $N/run_code_bench.py 18493 $N/code-bench.json out/$n || echo FAIL
    kill -TERM \$pid; sleep 5; kill -KILL \$pid 2>/dev/null; true"
  echo "-- $n ($(el))"; grep -o "titan ngram stats.*\|titan mtp stats.*\|panicked.*\|out of memory.*" $N/out/$n.server.log | tail -3
}
ccmp() { python3 -c "
import json,sys;a,b=json.load(open(sys.argv[1])),json.load(open(sys.argv[2]))
ok=[i for i,(x,y) in enumerate(zip(a,b)) if x is not None and y is not None]
print('CODE identity',sys.argv[2].split('/')[-1],'vs',sys.argv[1].split('/')[-1],':',sum(a[i]==b[i] for i in ok),'/',len(ok),'comparable of',len(a))" $1 $2; }

echo "== (1) code bench, ngram off ($(el))"
code cb2-off
echo "== (2) code bench, agree ($(el))"
code cb2-agree TITAN_NGRAM=1 TITAN_NGRAM_POLICY=agree; ccmp $N/out/cb2-off.json $N/out/cb2-agree.json
echo "== (3) 40 x 256 MTP=2: off, agree, first + MIN=3 ($(el))"
coll ng2-m2off $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2; cmp $E/m6/out/h-off.json $E/m3/out/ng2-m2off.json
coll ng2-agree $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2 TITAN_NGRAM=1 TITAN_NGRAM_POLICY=agree; cmp $E/m6/out/h-off.json $E/m3/out/ng2-agree.json
coll ng2-min3 $M/Qwen3.6-35B-A3B-MTP TITAN_MTP=2 TITAN_NGRAM=1 TITAN_NGRAM_MIN=3; cmp $E/m6/out/h-off.json $E/m3/out/ng2-min3.json
echo "== (4) code bench, first + MIN=3 ($(el))"
code cb2-min3 TITAN_NGRAM=1 TITAN_NGRAM_MIN=3; ccmp $N/out/cb2-off.json $N/out/cb2-min3.json
kill $VM 2>/dev/null
hoststate
echo "== ngram window2 end $(el)"
