#!/bin/bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO/bench-lib.sh"
set -u
bench_stop_prod; sleep 2
pkill -f "mistralrs ser[v]e" 2>/dev/null; sleep 4
export TITAN_MTP=0 TITAN_Q3K_ROWS=1
rm -rf /tmp/wk.nsys-rep /tmp/wk.sqlite
nsys profile -f true --trace=cuda -o /tmp/wk --sample=none \
  /home/dingo/src/titan-engine/mistral.rs/target/release/mistralrs serve \
  -p 1234 --no-ui --paged-attn off --max-seq-len 8192 --format gguf \
  -m models-optim -f optim-Q3_K_M.gguf -n 0:64 >/tmp/wk.log 2>&1 &
NPID=$!
for i in $(seq 1 40); do sleep 3; ss -tln | grep -q ':1234' && break; done
cd /tmp && python3 "$REPO/decode-once.py" 8 >/dev/null 2>&1; python3 "$REPO/decode-once.py" 16 >/dev/null 2>&1
sleep 1; kill -INT $NPID 2>/dev/null; sleep 15; kill -9 $NPID 2>/dev/null
nsys export --type sqlite -o /tmp/wk.sqlite --force-overwrite=true /tmp/wk.nsys-rep >/dev/null 2>&1
python3 - <<'PY'
import sqlite3
db=sqlite3.connect('/tmp/wk.sqlite')
q="""select s.value, k.blockY, k.gridX, count(*), avg((k.end-k.start)/1000.0)
from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName
where s.value like '%mmvq%' group by s.value, k.blockY, k.gridX order by count(*) desc limit 10"""
for name,by,grid,n,us in db.execute(q):
    print(f"{name[:46]:46s} blockY={by} grid={grid:6d} n={n:6d} {us:7.1f}us")
PY
