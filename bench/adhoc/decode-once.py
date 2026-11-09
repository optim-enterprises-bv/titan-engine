#!/usr/bin/env python3
"""One greedy decode against the live server and report tok/s -- the helper the ab-*/prof-* harnesses share.

    ./decode-once.py [max_tokens] [--prompt TEXT] [--port N]

Prints the decode rate (client-side, excluding prefill), the server's own usage numbers when it sends
them, a hash of the text so two runs can be compared for bit-identity, and the head of the output.
Exits non-zero if the server is not answering, so a harness can tell "slow" from "not serving".
"""
import argparse, hashlib, json, sys, time, urllib.error, urllib.request

DEFAULT_PROMPT = ("Write a very long, detailed chapter about GPU memory hierarchies and bandwidth "
                  "limits in inference engines. Keep going; do not summarise.")

ap = argparse.ArgumentParser()
ap.add_argument("max_tokens", nargs="?", type=int, default=256)
ap.add_argument("--prompt", default=DEFAULT_PROMPT)
ap.add_argument("--port", type=int, default=1234)
ap.add_argument("--model", default="default")
ap.add_argument("--timeout", type=float, default=1800.0)
a = ap.parse_args()

body = {"model": a.model, "temperature": 0.0, "max_tokens": a.max_tokens,
        "messages": [{"role": "user", "content": a.prompt}]}
req = urllib.request.Request(
    f"http://127.0.0.1:{a.port}/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})

t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=a.timeout) as f:
        raw = f.read()
except (urllib.error.URLError, ConnectionError) as e:
    print(f"decode-once: server on :{a.port} not answering ({e})", file=sys.stderr)
    sys.exit(2)
elapsed = time.time() - t0

d = json.loads(raw.decode())
msg = d["choices"][0]["message"]
# Thinking models answer into reasoning_content; with a small max_tokens the visible content can be
# empty for the whole run, which used to hash the empty string and make every A/B comparison vacuous.
content = msg.get("content") or ""
reason = msg.get("reasoning_content") or ""
txt = content or reason
field = "content" if content else ("reasoning_content" if reason else "none")
u = d.get("usage") or {}
compl = u.get("completion_tokens") or 0
srv = u.get("avg_compl_tok_per_sec")
rate = (compl / elapsed) if compl and elapsed > 0 else float("nan")
print(f"  {a.max_tokens:>5} tok  {elapsed:6.2f}s  {rate:6.1f} tok/s end-to-end"
      + (f"  server={srv:.1f} tok/s" if isinstance(srv, (int, float)) else "")
      + f"  sha={hashlib.sha256(txt.encode()).hexdigest()[:12]}"
      + f"  {field} {len(content)}c text/{len(reason)}c reason"
      + f"  head={txt[:60]!r}")
