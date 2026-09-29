#!/usr/bin/env python3
"""prefill2k.py PORT NAME [N]: prefill rate on a ~N-token raw prompt (bigprompt.py's text), max_tokens=1.
One warm-up, then the median of 5; a different first word per request defeats prefix caches.
Prints wall-clock tok/s (prompt tokens / time to the single generated token) and the server's own
prompt rate when it reports one (llama.cpp timings.prompt_per_second, mistral.rs avg_prompt_tok_per_sec)."""
import json, sys, time, urllib.error, urllib.request

port, name = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
chunk = "The titan engine tiers mixture-of-experts weights between the GPU and the CPU. "
text = chunk * (n // 16)  # ~16 tokens per chunk


def req(word):
    body = {"model": "default", "prompt": word + ". " + text, "max_tokens": 1, "temperature": 0.0, "top_k": 1, "cache_prompt": False}
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    dt = time.time() - t0
    u = r.get("usage") or {}
    srv = (r.get("timings") or {}).get("prompt_per_second") or u.get("avg_prompt_tok_per_sec")
    return u.get("prompt_tokens"), dt, srv


def try_req(word):
    # mistral.rs dense 14B at --max-seq-len 4096 sometimes OOMs a 2k prompt right after another; count and move on
    global fails
    try:
        return req(word)
    except urllib.error.HTTPError:
        fails += 1
        return None


fails = 0
try_req("Zulu")
res = [r for r in (try_req(w) for w in ("Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel")) if r][:5]
if not res:
    sys.exit(f"[{name}] prefill: every request failed")
ptoks = res[0][0]
ts = sorted(x[1] for x in res)
med = ts[len(ts) // 2]
srv = sorted(x[2] for x in res if x[2])
srv_med = srv[len(srv) // 2] if srv else None
print(f"[{name}] prefill {ptoks} tokens: {med * 1000:.0f} ms median -> {ptoks / med:.0f} tok/s wall"
      + (f", server-reported {srv_med:.0f} tok/s" if srv_med else "") + f" ({len(res)} ok, {fails} failed)")
json.dump({"prompt_tokens": ptoks, "ttft_s": ts, "wall_tok_s": ptoks / med, "server_tok_s": srv_med, "ok": len(res), "failed": fails},
          open(f"out/{name}_prefill.json", "w"))
