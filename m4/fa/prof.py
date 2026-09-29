#!/usr/bin/env python3
"""Decode profile after a long prompt: one cold request (max_tokens=1) fills the prefix cache, then
`nsys start`, the same request again (prefix-warm) with MAXTOK greedy tokens, `nsys stop`.
usage: prof.py PORT SYSFILE OUT TAG MAXTOK SESSION NSYSOUT"""
import json, subprocess, sys, time, urllib.request
port, sysf, out, tag, maxtok, sess, nout = sys.argv[1:8]
NSYS = "/opt/nvidia/nsight-compute/2026.2.1/host/target-linux-x64/nsys"
text = open(sysf).read()
def req(mt):
    body = {"model": "default", "messages": [{"role": "system", "content": text},
            {"role": "user", "content": "Explain the system text above section by section, in as much detail as you can. Do not stop early."}],
            "max_tokens": mt, "temperature": 0, "seed": 0}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1500))
    return r, time.time() - t
r, w = req(1)
print(f"{tag}: warm-up prompt {r['usage']['prompt_tokens']} tok in {w:.1f} s", flush=True)
subprocess.run([NSYS, "start", f"--session={sess}", "--sample=none", "--cpuctxsw=none", "-o", nout, "-f", "true"], check=True, timeout=120)
time.sleep(0.3)
r, w = req(int(maxtok))
time.sleep(0.3)
subprocess.run([NSYS, "stop", f"--session={sess}"], timeout=900, stdout=subprocess.DEVNULL)
u = r["usage"]
rec = {"tag": tag, "prompt_tokens": u.get("prompt_tokens"), "prompt_time": u.get("total_prompt_time_sec"),
       "completion_tokens": u.get("completion_tokens"), "decode_tok_s": u.get("avg_compl_tok_per_sec"), "wall": w}
open(out, "a").write(json.dumps(rec) + "\n")
print(f"{tag}: under nsys: prompt {rec['prompt_tokens']} (time {rec['prompt_time']:.2f} s), decode {rec['completion_tokens']} tok at {rec['decode_tok_s']:.1f} tok/s", flush=True)
