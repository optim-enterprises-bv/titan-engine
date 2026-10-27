#!/bin/bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO/bench-lib.sh"
set -u
bench_stop_prod; sleep 2
sudo -n pkill -f "mistralrs ser[v]e" 2>/dev/null; sleep 5
export TITAN_MTP=1
rm -rf /tmp/vk.nsys-rep /tmp/vk.sqlite
nsys profile -f true --trace=cuda -o /tmp/vk --sample=none \
  /home/dingo/src/titan-engine/mistral.rs/target/release/mistralrs serve \
  -p 1234 --no-ui --paged-attn off --max-seq-len 8192 --format gguf \
  -m /home/dingo/src/titan-engine/models-optim \
  -f Qwen3.8-27B-DevSecOps-QLoRA-abliterated-Q3_K_M.gguf -n 0:65 >/tmp/vk.log 2>&1 &
NPID=$!
for i in $(seq 1 40); do sleep 3; ss -tln | grep -q ':1234' && break; done
cd /tmp && python3 "$REPO/decode-once.py" 8 >/dev/null 2>&1; python3 "$REPO/decode-once.py" 40 >/dev/null 2>&1
sleep 1; kill -INT $NPID 2>/dev/null; sleep 15; kill -9 $NPID 2>/dev/null
sudo -n /usr/local/bin/nsys export --type sqlite -o /tmp/vk.sqlite --force-overwrite=true /tmp/vk.nsys-rep >/dev/null 2>&1
python3 - <<'PY'
import sqlite3, collections
db=sqlite3.connect('/tmp/vk.sqlite')
rows=list(db.execute("""select s.value, k.gridX, count(*), avg((k.end-k.start)/1000.0)
from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName
where s.value like '%mmvq_gguf_q%' group by s.value, k.gridX order by count(*) desc limit 12"""))
tot=collections.Counter()
for name,grid,n,us in rows:
    kind = 'rows' if '_rows' in name else ('batch' if any(name.endswith(f'cuda{i}') for i in (2,3,4,5,6,7,8)) else 'cuda1')
    tot[kind]+=n
    print(f"{name[:44]:44s} grid={grid:6d} n={n:6d} {us:7.1f}us")
print("\nkernel families sampled:", dict(tot))
PY
