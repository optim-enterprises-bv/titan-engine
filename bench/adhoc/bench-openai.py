#!/usr/bin/env python3
"""Benchmark one OpenAI-compatible endpoint the way systema uses it.

Cases (all greedy, one request at a time):
  short     ~20 tokens            - cold latency floor
  planner   the real systema planner prompt (~1.3k tokens, built from
            GET /commands)         - cold, then repeated (prefix-cache reuse)
  3k, 13k   filler prompts with a nonce at the front so no cache can help
  decode    256-token generation  - decode tok/s

Prints one line per case: wall seconds, prompt tokens, prefill tok/s (when the
server reports timings), decode tok/s.
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request

def post(base, body, timeout=900):
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
        wall = time.time() - t0
        return json.loads(raw), wall
    except Exception as e:
        return {"error": str(e)}, time.time() - t0

def planner_prompt(daemon="http://127.0.0.1:11999"):
    """The exact action list the planner sends, rendered the way planner.rs does."""
    try:
        with urllib.request.urlopen(daemon + "/commands", timeout=10) as r:
            cmds = json.loads(r.read().decode())["commands"]
    except Exception:
        return "ACTIONS:\n- read_file [auto]: Read a file args={}\n"
    lines = []
    for c in cmds:
        if c["id"] == "list_windows":
            continue
        lines.append(f'- {c["id"]} [{c["tier"]}]: {c["desc"]} args={json.dumps(c["args_schema"])}')
    head = (
        "You are Systema Rector, an agent controlling a Wayland desktop (Umbriel + Noctalia).\n"
        "You are given a GOAL, this node's memory, recent events and the steps so far.\n"
        "Each turn output EXACTLY one JSON object and NOTHING else.\n\nACTIONS:\n"
    )
    return head + "\n".join(lines)

def case(model, prompt, max_tokens, tag, base, timeout=900):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    d, wall = post(base, body, timeout)
    if "error" in d:
        print(f"{tag:22s} FAILED after {wall:7.1f}s: {d['error'][:90]}")
        return
    u = d.get("usage", {})
    t = d.get("timings", {}) or {}
    ptok = u.get("prompt_tokens", 0)
    ctok = u.get("completion_tokens", 0)
    pp = t.get("prompt_per_second")
    gp = t.get("predicted_per_second")
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    pp_s = f"{pp:8.1f}" if pp else f"{ptok/wall if wall else 0:8.1f}*"
    gp_s = f"{gp:8.1f}" if gp else "       -"
    print(
        f"{tag:22s} wall={wall:7.2f}s  prompt={ptok:6d}  prefill={pp_s} tok/s  "
        f"gen={gp_s} tok/s  out={ctok}" + (f"  cached={cached}" if cached else "")
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="optim")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    print(f"# {args.tag or args.base}  model={args.model}")
    case(args.model, "say ok", 8, "short", args.base)

    pp = planner_prompt()
    print(f"# planner prompt: {len(pp)} chars (~{len(pp)//3.6:.0f} tok est)")
    case(args.model, pp + "\n\nGOAL: report the failed systemd units\n\nSTEPS SO FAR:\n(none)\n\nOutput EXACTLY one JSON object, nothing else.", 64, "planner cold", args.base)
    case(args.model, pp + "\n\nGOAL: report the failed systemd units\n\nSTEPS SO FAR:\n(none)\n\nOutput EXACTLY one JSON object, nothing else.", 64, "planner repeat", args.base)

    for name, words in (("3k", 2200), ("13k", 9500)):
        nonce = f"request {time.time_ns()}\n"
        filler = ("The machine reports a window titled 'notes' on workspace dev. " * 1)
        case(args.model, nonce + filler * words + "\nReply with the JSON {\"action\":\"done\"}", 16, name + " cold", args.base)

    case(args.model, "Count from 1 to 200, comma separated, no other text.", 256, "decode 256", args.base)

if __name__ == "__main__":
    main()
