#!/bin/bash
# AUDIT.md B5: build the release tree nvcc-free from a copy of release/staging and smoke-test it on the 35B.
# Runs INSIDE a GPU window (titan-mistral stopped, lock held): campaign.sh calls it when CAMPAIGN_POST=staging-build.
#   staging-build.sh DEADLINE_EPOCH
# Build: a reflink copy of staging in release/build-test/src (staging itself stays pristine), CUDA_HOME pointing at a
# tree with only include/ and lib64/ (no bin/, so nvcc cannot run), CUDA_COMPUTE_CAP=120, --features oxide, under
# systemd-run MemoryMax=12G, CARGO_BUILD_JOBS=2 (RULES-agents.md). Cargo is incremental: a build cut off by the
# deadline continues in the next window.
# Smoke: the built binary with the deployed unit's env and args on a private port; the 8 bench prompts x 256 greedy
# with MTP=2 and with MTP off, identity vs bench/ref/h-off.first8.json (re-baselined 2026-09-29 to titan-idle2).
# Output: release/bench/staging-build/result.json + build.log + server logs.
set -u
E=$HOME/titan-engine; S=$E/release/staging; B=$E/release/build-test; O=$E/release/bench/staging-build
DEADLINE=${1:?deadline epoch}; PORT=18561
mkdir -p $O; T0=$(date +%s)
left() { echo $(( DEADLINE - $(date +%s) )); }
say() { echo "$(date +%H:%M:%S) staging-build: $*"; }
[ "$(systemctl --user is-active titan-mistral)" = active ] && { say "titan-mistral is running: refusing"; exit 3; }

rev=$(git -C $E/mr-094 rev-parse --short=12 titan-094)
if [ ! -f $B/src/.staging-rev ] || [ "$(cat $B/src/.staging-rev)" != "$rev" ]; then
  rm -rf $B/src; mkdir -p $B; cp -a --reflink=auto $S $B/src; echo $rev > $B/src/.staging-rev
fi
TGT=$E/target-staging-oxide
[ -d $TGT ] || cp -a --reflink=auto $E/target-graph-oxide $TGT
budget=$(( $(left) - 300 ))
[ $budget -lt 300 ] && { say "only $(left)s left: build deferred to the next window"; exit 4; }
say "building from $B/src (titan-094 $rev), budget ${budget}s"
t=$(date +%s)
timeout $budget systemd-run --user --unit=staging-build-$$ --collect --wait -q -p MemoryMax=12G -p MemorySwapMax=0 \
  -p RuntimeMaxSec=$budget -p WorkingDirectory=$B/src/mistral.rs \
  --setenv=CARGO_TARGET_DIR=$TGT --setenv=TITAN_OXIDE_DIR=$B/src/oxide-kernels --setenv=CUDA_HOME=$E/nocuda-bin \
  --setenv=CUDA_PATH=$E/nocuda-bin --setenv=CUDA_COMPUTE_CAP=120 --setenv=CUDARC_CUDA_VERSION=13030 \
  --setenv=CARGO_BUILD_JOBS=2 --setenv=CUDAFORGE_THREADS=1 --setenv=NVCC_THREADS=1 "--setenv=RUSTFLAGS=-L $E/lib" \
  --setenv=PATH=$HOME/.cargo/bin:/usr/bin:/bin \
  -p StandardOutput=truncate:$O/build.log -p StandardError=truncate:$O/build.log \
  nice -n 10 cargo build --release -p mistralrs-cli --features oxide
rc=$?; systemctl --user stop staging-build-$$ 2>/dev/null
grep -q "^error" $O/build.log && rc=1
build_s=$(( $(date +%s) - t ))
nvcc_lines=$(grep -c -i -E "(^|[ /])nvcc( |$)|Running.*nvcc" $O/build.log)
say "build rc=$rc in ${build_s}s; nvcc mentions in the log: $nvcc_lines"
BIN=$TGT/release/mistralrs
smoke='{}'
if [ $rc = 0 ] && [ -x $BIN ] && [ $(left) -gt 150 ]; then
  mapfile -t SENV < <(python3 $E/bench/lib/svcconf.py env)
  mapfile -t SARGS < <(python3 $E/bench/lib/svcconf.py args $PORT)
  for mode in mtp2 off; do
    extra=""; [ $mode = off ] && extra="TITAN_MTP=0"
    timeout $(( $(left) - 30 )) systemd-run --user --unit=staging-smoke-$mode-$$ --collect --wait --pipe -q -p MemoryMax=20G \
      -p MemorySwapMax=0 bash -c "
      env ${SENV[*]} $extra $BIN --seed 0 ${SARGS[*]} > $O/smoke-$mode.server.log 2>&1 &
      pid=\$!
      for i in \$(seq 1 300); do curl -sf -m 2 -o /dev/null localhost:$PORT/v1/models && break; kill -0 \$pid 2>/dev/null || break; sleep 1; done
      python3 - <<'PY' > $O/smoke-$mode.json
import json, sys
sys.path.insert(0, '$E/bench/lib')
import client as C
C.warm($PORT)
e = C.eight($PORT, '$E/bench/ref/h-off.first8.json')
e.pop('texts', None)
print(json.dumps(e))
PY
      kill -TERM \$pid; for i in \$(seq 1 20); do kill -0 \$pid 2>/dev/null || break; sleep 0.5; done; kill -KILL \$pid 2>/dev/null; true"
    say "smoke $mode: $(python3 -c "import json;d=json.load(open('$O/smoke-$mode.json'));print('identity', d.get('identity'), 'decode', round(d.get('decode_tok_s') or 0,1), 'tok/s')" 2>&1)"
  done
  smoke=$(python3 -c "
import json
out = {}
for m in ('mtp2', 'off'):
    try: out[m] = json.load(open('$O/smoke-' + m + '.json'))
    except Exception as e: out[m] = {'error': str(e)}
print(json.dumps(out))")
fi
python3 - <<PY
import json, hashlib, os
b = "$BIN"
sha = hashlib.sha256(open(b, "rb").read()).hexdigest() if $rc == 0 and os.path.exists(b) else None
json.dump({"time": "$(date -Is)", "staging_rev_mistral_rs": "$rev", "build_rc": $rc, "build_s": $build_s,
           "nvcc_mentions_in_build_log": $nvcc_lines, "cuda_home": "$E/nocuda-bin (include/ and lib64/ only, no bin/)",
           "binary": b, "binary_sha256": sha, "smoke": json.loads('''$smoke''')},
          open("$O/result.json", "w"), indent=1)
PY
say "done in $(( $(date +%s) - T0 ))s: $O/result.json"
