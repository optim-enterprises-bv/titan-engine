#!/usr/bin/env python3
"""Deploy dry run client: swap through every roster entry once (toml order), one short chat request each.
usage: dryrun.py PORT TOML SERVER_LOG OUT.json
Per model: request wall time (includes the swap), the server's `titan swap: loaded NAME in X s` line, nvidia-smi used
MiB after the reply, finish reason, token counts, and the reply text (content and reasoning) for a coherence read."""
import json, re, subprocess, sys, time, tomllib, urllib.error, urllib.request

port, toml, slog, out = sys.argv[1:5]
import os
models = [m["name"] for m in tomllib.load(open(toml, "rb"))["models"]]
if os.environ.get("MODELS"):  # A/B re-runs: only these models, each REPEAT times (the first request swaps it in)
    models = [m for m in os.environ["MODELS"].split(",") for _ in range(int(os.environ.get("REPEAT", "1")))]
Q = "In two or three sentences, explain why the sky is blue."


def smi():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(r.stdout.strip().splitlines()[0]) if r.returncode == 0 and r.stdout.strip() else -1


def loaded_line(name):
    txt = re.sub(r"\x1b\[[0-9;]*m", "", open(slog, errors="replace").read())
    hits = re.findall(rf"titan swap: loaded {re.escape(name)} in ([0-9.]+) s:?([^\n]*)", txt)
    return hits[-1] if hits else None


rows = []
for name in models:
    body = {"model": name, "messages": [{"role": "user", "content": Q}], "max_tokens": 400, "temperature": 0}
    t = time.time()
    row = {"model": name}
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=900))
        ch = r["choices"][0]
        msg = ch.get("message") or {}
        u = r.get("usage") or {}
        row.update(ok=True, finish=ch.get("finish_reason"), content=msg.get("content") or "",
                   reasoning=msg.get("reasoning_content") or "", prompt_tokens=u.get("prompt_tokens"),
                   completion_tokens=u.get("completion_tokens"), decode_tps=u.get("avg_compl_tok_per_sec"))
    except urllib.error.HTTPError as e:
        row.update(ok=False, error=f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
    except Exception as e:  # noqa: BLE001 - recorded, the run goes on
        row.update(ok=False, error=repr(e)[:300])
    row["wall_s"] = round(time.time() - t, 1)
    row["vram_used_mib"] = smi()
    ll = loaded_line(name)
    row["load_s"], row["load_line"] = (float(ll[0]), ll[1].strip()) if ll else (None, None)
    rows.append(row)
    text = (row.get("content") or "").strip().replace("\n", " ")
    print(f"DRY {name}: {'ok' if row['ok'] else 'ERR'} wall {row['wall_s']} s, load {row['load_s']} s ({row['load_line']}), "
          f"VRAM used {row['vram_used_mib']} MiB, finish {row.get('finish')}, tokens {row.get('completion_tokens')}, "
          f"{row.get('decode_tps')} tok/s", flush=True)
    print(f"    content: {text[:400]!r}", flush=True)
    if row.get("reasoning"):
        print(f"    reasoning: {row['reasoning'].strip()[:200]!r}", flush=True)
    if not row["ok"]:
        print(f"    error: {row['error']}", flush=True)
    json.dump(rows, open(out, "w"), ensure_ascii=False, indent=1)
print(f"DRYRUN: {sum(r['ok'] for r in rows)}/{len(rows)} models replied")
