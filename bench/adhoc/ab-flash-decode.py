#!/usr/bin/env python3
"""Depth sweep for one titan-engine endpoint: decode tok/s at several KV depths.

One greedy request per depth, nonce at the front so no prefix cache can help, raw
/v1/completions (avoids chat-template render failures), timings read from `usage`.

    python3 ab-flash-decode.py --port 1234 --label prod --depths 1024,8192,49152
"""
import argparse
import json
import sys
import time
import urllib.request

# Varying per-line numbers keep the filler from tokenizing well below the requested depth.
def body_of(chars: int) -> str:
    out = []
    i = 0
    n = 0
    while n < chars:
        line = f"step {i}: the quick brown fox jumps over {i * 7919 % 10007} lazy dogs, pass {i * 31 % 9973}.\n"
        out.append(line)
        n += len(line)
        i += 1
    return "".join(out)[:chars]


def build_prompt(tok_target: int, nonce: str, tok_per_char: float) -> str:
    chars = int(tok_target / tok_per_char)
    return (
        f"[run {nonce}] Read the text below, then continue the document in the same style.\n"
        f"{body_of(chars)}\nContinue the document at length, in full sentences:\n\nStep"
    )


def post(base: str, path: str, body: dict, timeout: int = 1800) -> dict:
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1235)
    ap.add_argument("--depths", default="256,1024,4096,16384,49152")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default="/home/dingo/src/titan-engine/release/bench/flash-decode.jsonl")
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"

    try:
        post(base, "/v1/completions", {"model": "default", "prompt": "hi", "max_tokens": 1, "temperature": 0})
    except Exception as e:  # noqa: BLE001
        print(f"warmup failed: {e}", file=sys.stderr)

    # Calibrate tokens/char on this filler instead of trusting a published ratio.
    probe_chars = 20000
    tok_per_char = 0.46
    try:
        pr = post(base, "/v1/completions", {
            "model": "default",
            "prompt": body_of(probe_chars) + "\ncount",
            "max_tokens": 1,
            "temperature": 0,
        })
        got = pr.get("usage", {}).get("prompt_tokens", 0)
        if got > 0:
            tok_per_char = got / probe_chars
        print(json.dumps({"calibration": {"chars": probe_chars, "prompt_tokens": got,
                                          "tok_per_char": round(tok_per_char, 4)}}), flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"calibration failed: {e}", file=sys.stderr)

    rows = []
    for depth in [int(x) for x in a.depths.split(",")]:
        nonce = f"{a.label}-{depth}-{int(time.time() * 1000)}"
        body = {
            "model": "default",
            "prompt": build_prompt(depth, nonce, tok_per_char),
            "max_tokens": a.max_tokens,
            "temperature": 0,
            "top_k": 1,
            "stream": False,
        }
        t0 = time.time()
        try:
            r = post(base, "/v1/completions", body)
        except Exception as e:  # noqa: BLE001
            row = {"label": a.label, "target": depth, "error": str(e)}
            rows.append(row)
            print(json.dumps(row), flush=True)
            continue
        u = r.get("usage", {})
        row = {
            "label": a.label,
            "target": depth,
            "prompt_tokens": u.get("prompt_tokens"),
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "prefill_tok_s": round(u.get("avg_prompt_tok_per_sec", 0.0), 1),
            "decode_tok_s": round(u.get("avg_compl_tok_per_sec", 0.0), 2),
            "completion_tokens": u.get("completion_tokens"),
            "wall_s": round(time.time() - t0, 2),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    if a.out:
        with open(a.out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


main()
