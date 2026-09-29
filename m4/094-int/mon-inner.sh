set -u
E=$HOME/titan-engine; MON=$E/m4/mon; O=$MON/out; P=18492
mapfile -t SENV < <(python3 "$E/bench/lib/svcconf.py" env)
mapfile -t SARGS < <(python3 "$E/bench/lib/svcconf.py" args "$P")
cd "$MON"
env "${SENV[@]}" "$BIN" --seed 0 "${SARGS[@]}" > "$O/094-live.server.log" 2>&1 &
pid=$!
for i in $(seq 1 200); do curl -sf -m 2 -o /dev/null localhost:$P/v1/models && break; kill -0 $pid 2>/dev/null || break; sleep 2; done
echo "server up after $((i*2))s"
curl -s localhost:$P/monitor | head -c 200; echo
curl -s localhost:$P/v1/titan/stats > "$O/094-idle-first.json"
python3 "$MON/poll.py" $P "$O/094-live-poll.jsonl" & ppid=$!
echo "-- 13k prompt (cold) $(date +%T)"
python3 "$E/m4/bigprompt.py" $P 13000 > "$O/094-big1.txt" 2>&1
cat "$O/094-big1.txt"
echo "-- long decode MTP=2 $(date +%T)"
python3 "$MON/longgen.py" $P 1500 > "$O/094-decode.txt" 2>&1
cat "$O/094-decode.txt"
curl -s localhost:$P/v1/titan/stats > "$O/094-final.json"
kill $ppid 2>/dev/null
kill -TERM $pid 2>/dev/null; sleep 5; kill -KILL $pid 2>/dev/null; true
