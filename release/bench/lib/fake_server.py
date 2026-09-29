#!/usr/bin/env python3
"""Fake inference server for the campaign's CPU-only dry run (camp.py dry). NOT a benchmark: it streams
OpenAI-style SSE in the dialect of either engine so the client, the sweep logic, the gates, the result files and the
report can be exercised without a GPU.

  --flavor ours   mistral.rs dialect: health on /v1/models; usage with total_prompt_time_sec in the last chunk;
                  several tokens per chunk sometimes (as MTP bursts do)
  --flavor llama  llama-server dialect: /health; usage + "timings" in the last chunk; exits at start with an
                  out-of-memory message when --n-cpu-moe is below the fake model's limit (to exercise the sweep)
Every other argument is accepted and ignored. Rates are scaled down so the dry run takes minutes."""
import json, os, random, re, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

a = sys.argv[1:]


def arg(names, default=None):
    for n in names:
        if n in a:
            return a[a.index(n) + 1]
    return default


FLAVOR = arg(["--flavor"], "llama")
MODEL = arg(["--model"], "x")
PORT = int(arg(["--port", "-p"], "18580"))
NCMOE = arg(["--n-cpu-moe"])
FIT = arg(["--fit"]) == "on"
MTP = arg(["--spec-draft-n-max"])
LIMIT = {"q35xl": 16, "q35mx": 18, "next80": 35, "oss120": 29, "oss20": 1}.get(MODEL, 0)
if FLAVOR == "llama" and NCMOE is not None and not FIT and int(NCMOE) < LIMIT + (1 if MTP else 0):
    print(f"ggml_backend_cuda_buffer_type_alloc_buffer: allocating 16384 MiB on device 0: cudaMalloc failed: out of memory",
          flush=True)
    sys.exit(1)
time.sleep(1.0)  # "load"
rnd = random.Random(os.getpid())
BASE = {"ours": (4000.0, 1000.0), "llama": (3000.0, 700.0)}[FLAVOR]  # (prefill, decode) tok/s, scaled for speed
if FLAVOR == "llama":
    BASE = (BASE[0] * (1 + 0.02 * (int(arg(["-ub"], "512")) > 512)), BASE[1] * (1 + 0.1 * bool(MTP)) * (1 - 0.01 * int(NCMOE or 0) / 10))


def jitter():
    return 1 + rnd.uniform(-0.004, 0.004) * float(os.environ.get("FAKE_JITTER", "1"))


class H(BaseHTTPRequestHandler):
    def log_message(self, *x):
        pass

    def do_GET(self):
        if self.path in ("/health", "/v1/models"):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/completions":  # quality: non-streamed raw completion
            p = body["prompt"]
            m = re.search(r"What is (\d+) \* (\d+)", p)
            ans = str(int(m.group(1)) * int(m.group(2))) if m else "Paris"
            txt = ("<|channel|>analysis<|message|>Simple.<|end|><|start|>assistant<|channel|>final<|message|>" + ans) \
                if "<|start|>" in p else ans
            out = {"choices": [{"text": txt}], "usage": {"prompt_tokens": len(p) // 4, "completion_tokens": 5}}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(out).encode())
            return
        chars = sum(len(m["content"]) for m in body["messages"])
        ptok = max(8, chars // 4)
        n = int(body.get("max_tokens", 16))
        pre, dec = BASE[0] * jitter(), BASE[1] * jitter()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        t0 = time.time()
        time.sleep(ptok / pre / 20)  # scaled: prompt 20x faster than the nominal rate
        tp = time.time() - t0
        sent = 0
        while sent < n:
            k = min(n - sent, 1 if FLAVOR == "llama" else rnd.choice([1, 1, 2, 3]))
            d = {"choices": [{"delta": {"content": "tok " * k}}]}
            self.wfile.write(f"data: {json.dumps(d)}\n\n".encode())
            self.wfile.flush()
            sent += k
            time.sleep(k / dec)
        u = {"prompt_tokens": ptok, "completion_tokens": n, "total_tokens": ptok + n}
        last = {"choices": [], "usage": u}
        if FLAVOR == "ours":
            u["total_prompt_time_sec"] = tp
            u["avg_compl_tok_per_sec"] = dec
        else:
            last["timings"] = {"prompt_n": ptok, "prompt_ms": tp * 1e3, "predicted_n": n, "predicted_per_second": dec}
        self.wfile.write(f"data: {json.dumps(last)}\n\ndata: [DONE]\n\n".encode())


print(f"fake {FLAVOR} server for {MODEL} on {PORT}", flush=True)
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
