#!/usr/bin/env python3
"""Run the code-editing benchmark against a running server.
usage: run_code_bench.py PORT PROMPTS_JSON OUT_PREFIX
Writes OUT_PREFIX.json (completions) and OUT_PREFIX.metrics.json; prints a summary line.
Greedy (temperature 0, seed 0), thinking off (an editing turn that emits the edited text)."""
import json, os, subprocess, sys, time, urllib.request

TIMEOUT = int(os.environ.get("REQ_TIMEOUT", "240"))  # one hung request must not stall the run


def diag(tag):
    """Host + server state when a request times out (the window-1 off run stalled 11 min in one request)."""
    pid = os.environ.get("SERVER_PID", "")
    cmds = ["uptime", "free -m", "ps -eo pid,ppid,rss,pcpu,etime,comm --sort=-rss | head -6",
            "ps -eo pid,pcpu,comm --sort=-pcpu | head -6", "cat /proc/pressure/memory /proc/pressure/io /proc/pressure/cpu"]
    if pid:
        cmds += [f"ps -L -o tid,stat,pcpu,wchan:32 -p {pid} | sort -k2 | uniq -c -f1 | sort -rn | head -12",
                 f"grep -E 'VmRSS|Threads' /proc/{pid}/status"]
    with open(out + f".diag-{tag}.txt", "w") as f:
        for c in cmds:
            f.write(f"$ {c}\n" + subprocess.run(c, shell=True, capture_output=True, text=True).stdout + "\n")

port, pf, out = sys.argv[1], sys.argv[2], sys.argv[3]
prompts = json.load(open(pf))
res, rows, t0 = [], [], time.time()
for p in prompts:
    body = json.dumps({"model": "default", "messages": p["messages"], "temperature": 0.0, "seed": 0,
                       "max_tokens": p["max_tokens"], "enable_thinking": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=TIMEOUT))
    except Exception as e:  # timeout (or a dropped connection): record, keep diagnostics, go on
        print(f"  {p['id']:2d} {p['kind']:4s} FAILED after {time.time() - t:.0f}s: {e!r}", flush=True)
        diag(p["id"])
        res.append(None)
        rows.append({"id": p["id"], "kind": p["kind"], "wall": time.time() - t, "error": repr(e),
                     "prompt_tokens": None, "completion_tokens": None, "total_prompt_time_sec": None,
                     "total_completion_time_sec": None, "avg_compl_tok_per_sec": None})
        continue
    u = r.get("usage", {})
    m = r["choices"][0]["message"]
    res.append((m.get("reasoning_content") or "") + (m.get("content") or ""))
    rows.append({"id": p["id"], "kind": p["kind"], "wall": time.time() - t,
                 **{k: u.get(k) for k in ("prompt_tokens", "completion_tokens", "total_prompt_time_sec",
                                          "total_completion_time_sec", "avg_compl_tok_per_sec")}})
    print(f"  {p['id']:2d} {p['kind']:4s} prompt {u.get('prompt_tokens')} compl {u.get('completion_tokens')} "
          f"decode {u.get('completion_tokens', 0) / max(u.get('total_completion_time_sec') or 1e-9, 1e-9):.1f} tok/s", flush=True)
# a tiny last request makes the server log its cumulative n-gram / MTP counters
body = json.dumps({"model": "default", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1,
                   "temperature": 0.0}).encode()
try:
    urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                                  headers={"Content-Type": "application/json"}), timeout=TIMEOUT).read()
except Exception as e:
    print("  flush request failed:", repr(e))
json.dump(res, open(out + ".json", "w"), ensure_ascii=False, indent=1)
ct = sum(r["completion_tokens"] or 0 for r in rows)
dt = sum(r["total_completion_time_sec"] or 0 for r in rows)
pt = sum(r["prompt_tokens"] or 0 for r in rows)
summary = {"requests": len(rows), "failed": [r["id"] for r in rows if r.get("error")], "prompt_tokens": pt, "completion_tokens": ct, "decode_s": dt,
           "decode_tok_s": ct / dt if dt else 0, "wall_s": time.time() - t0, "rows": rows}
for k in ("file", "diff", "fix"):
    sub = [r for r in rows if r["kind"] == k]
    c, d = sum(r["completion_tokens"] or 0 for r in sub), sum(r["total_completion_time_sec"] or 0 for r in sub)
    summary[f"decode_tok_s_{k}"] = c / d if d else 0
json.dump(summary, open(out + ".metrics.json", "w"), indent=1)
print(f"{out.split('/')[-1]}: {len(rows)} requests ({len(summary['failed'])} failed), {pt} prompt / {ct} completion tokens, decode {summary['decode_tok_s']:.1f} tok/s "
      f"(file {summary['decode_tok_s_file']:.1f}, diff {summary['decode_tok_s_diff']:.1f}, fix {summary['decode_tok_s_fix']:.1f}), wall {summary['wall_s']:.0f}s")
