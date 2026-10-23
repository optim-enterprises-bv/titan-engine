#!/usr/bin/env python3
"""Aggregate vs single-stream throughput against a running OpenAI-compatible server.

The 367.9 tok/s figure is aggregate across 8 concurrent requests, so that is the
number to reproduce: completion tokens summed over wall clock, not per-stream speed.
"""
import json, sys, threading, time, urllib.request

URL = "http://127.0.0.1:1234/v1/chat/completions"
PROMPT = ("Write a long detailed technical explanation of how a Linux kernel scheduler "
          "decides which process runs next. Be thorough and concrete.")


def one(idx, ntok, out):
    body = {"model": "default", "messages": [{"role": "user", "content": f"req {idx}: {PROMPT}"}],
            "temperature": 0.0, "max_tokens": ntok}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.loads(r.read().decode())
        u = d.get("usage", {})
        out[idx] = (u.get("completion_tokens", 0), time.time() - t0)
    except Exception as e:
        out[idx] = (0, time.time() - t0)
        print(f"  stream {idx} failed: {e}")


def run(n_streams, ntok):
    out = {}
    th = [threading.Thread(target=one, args=(i, ntok, out)) for i in range(n_streams)]
    t0 = time.time()
    for t in th: t.start()
    for t in th: t.join()
    wall = time.time() - t0
    toks = sum(v[0] for v in out.values())
    per = [v[0] / v[1] for v in out.values() if v[1] > 0]
    print(f"  {n_streams:2d} concurrent: {toks:5d} tok in {wall:6.1f}s -> aggregate {toks/wall:6.1f} tok/s"
          f"  (per-stream {min(per) if per else 0:.1f}-{max(per) if per else 0:.1f})")
    return toks / wall


if __name__ == "__main__":
    ntok = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    run(1, 16)                      # warm
    agg1 = run(1, ntok)
    agg8 = run(8, ntok)
    print(f"  batch-8 speedup vs single stream: {agg8/agg1:.2f}x")
