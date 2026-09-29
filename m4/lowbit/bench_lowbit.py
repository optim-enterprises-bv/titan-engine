#!/usr/bin/env python3
"""Low-bit Qwen3.6-35B-A3B vs tiered Q4_K_XL benchmark, llama.cpp only.
Starts llama-server per model config, measures decode tok/s (8x128 greedy),
prefill tok/s at ~4k/~13k tokens, VRAM fit, quality (practice20 + q100),
and first-token top-1 drift vs the Q4_K_XL baseline on the 40-prompt eval set.

usage: bench_lowbit.py DEADLINE_EPOCH
writes out/lowbit-results.json incrementally (one model appended at a time).
"""
import gzip, json, os, re, subprocess, sys, tempfile, time, urllib.request, urllib.error

HOME = os.path.expanduser("~")
E = f"{HOME}/titan-engine"
L = f"{E}/m4/lowbit"
M = f"{HOME}/ai/models"
BIN = f"{HOME}/ai/llama.cpp/build/bin/llama-server"
CTX = 16384
THREADS = 23
PORT = 18700
GPU_TOTAL_MIB = 16303  # RTX 5080 16GB laptop, hardware spec (not queried - nvidia-smi/NVML is unreliable here)
SAFE_CEILING_MIB = 15600  # headroom for CUDA context overhead not captured in the buffer-size log lines

deadline = float(sys.argv[1]) if len(sys.argv) > 1 else time.time() + 40 * 60

RUNS = [
    {"name": "q4kxl", "file": f"{M}/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf", "ncmoe": 18, "search": "down", "baseline": True, "skip_shrink": True},
    {"name": "iq3xxs", "file": f"{M}/q35-lowbit/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf", "ncmoe": 0, "search": "up", "baseline": False},
    {"name": "iq2m", "file": f"{M}/q35-lowbit/Qwen3.6-35B-A3B-UD-IQ2_M.gguf", "ncmoe": 0, "search": "up", "baseline": False},
]

# window1's q4kxl decode (22 tok/s at ncmoe=14) looked wrong: historical llama.cpp notes on this
# exact model measured ~61 tok/s at ncmoe=18, and more offload (18>14) should be slower not
# faster, not this much faster. Prime suspects: no warmup (--no-warmup was set, so the first
# timed request pays cold-start cost), and/or CPU contention from other agents sharing the box.
# window2 re-measures with a warmup request first and sweeps ncmoe to find the real number.
Q4_NCMOE_CANDIDATES = [14, 16, 18]

SHORT_PROMPTS = [
    "Write a short story about a robot learning to paint.",
    "Explain how photosynthesis works in two sentences.",
    "List three benefits of regular exercise.",
    "Describe the water cycle briefly.",
    "What makes a good leader? Give a short answer.",
    "Summarize the plot of a fairy tale in one paragraph.",
    "Give a brief explanation of how a computer CPU works.",
    "Write a short poem about the ocean.",
]

FILLER = "The titan engine tiers mixture-of-experts weights between the GPU and the CPU. "


def left():
    return deadline - time.time()


def http_json(url, body=None, timeout=120):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def tokenize(n_predict_text):
    return http_json(f"http://127.0.0.1:{PORT}/tokenize", {"content": n_predict_text})["tokens"]


def build_prompt(target_tokens):
    """Build a filler prompt of ~target_tokens tokens. Calibrates tokens-per-chunk first (the
    naive target//12 guess used earlier overshot badly - a 13000-token target built to 17329
    tokens, past CTX=16384, and the server 400'd), then grows one chunk at a time so the
    overshoot is bounded to a single chunk, and hard-caps at CTX regardless."""
    sample_chunks = 8
    per_chunk = len(tokenize(FILLER * sample_chunks)) / sample_chunks
    guess_chunks = max(1, int(target_tokens / per_chunk * 0.92))
    text = FILLER * guess_chunks
    toks = tokenize(text)
    while len(toks) < target_tokens:
        text += FILLER
        toks = tokenize(text)
    max_allowed = CTX - 512  # leave room for n_predict + special tokens
    while len(toks) > max_allowed:
        text = text[: -len(FILLER)]
        toks = tokenize(text)
    return text, len(toks)


def parse_buffer_sizes(log_path):
    """Parse llama.cpp's load-time buffer-size lines from the server log (e.g.
    'load_tensors:        CUDA0 model buffer size = 12345.67 MiB', '... KV buffer size = ...',
    '... compute buffer size = ...'). These come straight from the CUDA runtime via libcuda,
    not NVML/nvidia-smi, so they stay reliable even though nvidia-smi is broken on this box
    (NVML is version-strict here; see titan-nvidia-smi-broken-cuda-fine). This is the
    authoritative source for VRAM fit, not nvidia-smi."""
    try:
        text = open(log_path, errors="replace").read()
    except Exception:
        return {}
    out = {}
    for label, key in (("model buffer size", "model"), ("KV buffer size", "kv"), ("compute buffer size", "compute")):
        total, found = 0.0, False
        for m in re.finditer(rf"CUDA\d+[^\n]*?{re.escape(label)}\s*=\s*([\d.]+)\s*MiB", text):
            total += float(m.group(1))
            found = True
        if found:
            out[f"cuda_{key}_mib"] = round(total, 1)
    if out:
        out["cuda_total_mib"] = round(sum(out.values()), 1)
    cpu_total, cpu_found = 0.0, False
    for m in re.finditer(r"CPU[^\n]*?model buffer size\s*=\s*([\d.]+)\s*MiB", text):
        cpu_total += float(m.group(1))
        cpu_found = True
    if cpu_found:
        out["cpu_model_mib"] = round(cpu_total, 1)
    return out


def vram_used_mib_best_effort():
    """nvidia-smi/NVML is known broken on this box (version-strict NVML fails while CUDA
    itself works fine). Treat this purely as optional extra telemetry: any failure here must
    never abort the run or be interpreted as 'doesn't fit' - only parse_buffer_sizes() and a
    failed load/OOM in the server log decide fit."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5, stderr=subprocess.DEVNULL,
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def start_server(path, ncmoe, log_path):
    args = [
        BIN, "-m", path, "-ngl", "99", "-c", str(CTX), "-t", str(THREADS),
        "-fa", "on", "--reasoning", "off", "-np", "1", "--port", str(PORT),
        "--no-warmup", "-lv", "4",
    ]
    if ncmoe:
        args += ["--n-cpu-moe", str(ncmoe)]
    logf = open(log_path, "w")
    p = subprocess.Popen(args, stdout=logf, stderr=subprocess.STDOUT)
    return p, logf


def wait_health(p, timeout_s):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if p.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            return True
        except Exception:
            time.sleep(2)
    return False


def stop_server(p, logf):
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=15)
    try:
        logf.close()
    except Exception:
        pass


def log_has_oom(log_path):
    try:
        text = open(log_path, errors="replace").read()
    except Exception:
        return False
    return bool(re.search(r"out of memory|cudaMalloc failed|failed to allocate|CUDA error", text, re.I))


def load_with_search(run):
    """Try to start the server, adjusting --n-cpu-moe if it doesn't fit. Fit is decided by
    whether the server comes up healthy with no OOM in its log (parse_buffer_sizes() then
    gives the authoritative CUDA buffer breakdown); nvidia-smi is never consulted for this.
    Returns (proc, logf, ncmoe, cuda_buffers, vram_smi_best_effort, tried) or None."""
    ncmoe = run["ncmoe"]
    tried = []
    max_attempts = 3
    for attempt in range(max_attempts):
        if left() < 300:
            print(f"[{run['name']}] not enough time left ({left():.0f}s) to keep searching n-cpu-moe")
            return None
        log_path = f"{L}/out/{run['name']}.server.log"
        print(f"[{run['name']}] attempt {attempt+1}: --n-cpu-moe {ncmoe}")
        p, logf = start_server(run["file"], ncmoe, log_path)
        timeout_s = 240 if run["baseline"] else 180
        ok = wait_health(p, timeout_s) and not log_has_oom(log_path)
        if not ok:
            oom = log_has_oom(log_path)
            stop_server(p, logf)
            tried.append((ncmoe, "oom" if oom else "fail"))
            ncmoe = (ncmoe or 0) + (8 if attempt == 0 else 4)
            continue
        buf = parse_buffer_sizes(log_path)
        vram_smi = vram_used_mib_best_effort()  # best-effort telemetry only, may be None
        print(f"[{run['name']}] healthy with --n-cpu-moe {ncmoe}, cuda_buffers={buf} (nvidia-smi best-effort={vram_smi})")
        cuda_total = buf.get("cuda_total_mib")
        # one optional shrink step for the baseline if there's clear headroom (by the log's own
        # buffer accounting, not nvidia-smi) and time to spare
        if run["baseline"] and not run.get("skip_shrink") and ncmoe and cuda_total is not None and (SAFE_CEILING_MIB - cuda_total) > 1500 and left() > 900:
            smaller = max(0, ncmoe - 6)
            print(f"[{run['name']}] cuda buffers {cuda_total} MiB vs {SAFE_CEILING_MIB} MiB ceiling; trying smaller --n-cpu-moe {smaller}")
            stop_server(p, logf)
            log_path2 = f"{L}/out/{run['name']}.server2.log"
            p2, logf2 = start_server(run["file"], smaller, log_path2)
            if wait_health(p2, 240) and not log_has_oom(log_path2):
                buf2 = parse_buffer_sizes(log_path2)
                cuda_total2 = buf2.get("cuda_total_mib")
                if cuda_total2 is not None and cuda_total2 < SAFE_CEILING_MIB:
                    print(f"[{run['name']}] smaller --n-cpu-moe {smaller} also fits, cuda_buffers={buf2}")
                    return p2, logf2, smaller, buf2, vram_used_mib_best_effort(), tried + [(ncmoe, "ok-but-shrunk")]
                stop_server(p2, logf2)
            else:
                stop_server(p2, logf2)
            # shrink didn't pan out; reload at the known-good ncmoe
            p, logf = start_server(run["file"], ncmoe, log_path)
            wait_health(p, 240)
            buf = parse_buffer_sizes(log_path)
            vram_smi = vram_used_mib_best_effort()
        return p, logf, ncmoe, buf, vram_smi, tried
    print(f"[{run['name']}] FAILED to fit after {max_attempts} attempts: {tried}")
    return None


def cpu_load():
    """1/5/15min load averages, for spotting CPU contention from other agents sharing the box."""
    try:
        return list(os.getloadavg())
    except Exception:
        return None


def warmup(n_predict=32):
    """One throwaway request so the timed requests don't pay for cold CUDA-graph capture / first
    mmap page faults. --no-warmup is set at server startup to save load time, so this is required
    before trusting any predicted_per_second number."""
    try:
        http_json(f"http://127.0.0.1:{PORT}/completion",
                  {"prompt": "Warm up before timing.", "n_predict": n_predict, "temperature": 0, "top_k": 1,
                   "cache_prompt": False}, timeout=120)
    except Exception as e:
        print(f"  warmup request failed (continuing anyway): {e}")


def decode_bench(prompts=None, n_predict=128):
    prompts = SHORT_PROMPTS if prompts is None else prompts
    warmup()
    speeds = []
    load_before = cpu_load()
    for prompt in prompts:
        body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0, "top_k": 1, "cache_prompt": False}
        r = http_json(f"http://127.0.0.1:{PORT}/completion", body, timeout=180)
        t = r.get("timings", {})
        pps = t.get("predicted_per_second")
        if pps:
            speeds.append(pps)
    load_after = cpu_load()
    avg = sum(speeds) / len(speeds) if speeds else None
    return {"per_prompt": speeds, "avg_tok_s": avg, "cpu_load_before": load_before, "cpu_load_after": load_after}


def prefill_bench(target):
    text, n = build_prompt(target)
    body = {"prompt": text, "n_predict": 4, "temperature": 0, "top_k": 1, "cache_prompt": False}
    r = http_json(f"http://127.0.0.1:{PORT}/completion", body, timeout=300)
    t = r.get("timings", {})
    return {"target_tokens": target, "actual_tokens": t.get("prompt_n", n), "tok_s": t.get("prompt_per_second")}


def quality_eval():
    practice = json.load(open(f"{E}/m5/oss/questions.json"))
    q100 = json.load(open(f"{E}/m4/skip/q100.json"))
    out = {}
    for sname, qs in (("practice", practice), ("q100", q100)):
        res = []
        for q, pat in qs:
            body = {"model": "default", "messages": [{"role": "user", "content": f"{q} Answer with just the answer."}],
                    "max_tokens": 48, "temperature": 0.0, "seed": 0}
            try:
                r = http_json(f"http://127.0.0.1:{PORT}/v1/chat/completions", body, timeout=180)
                text = r["choices"][0]["message"]["content"] or ""
            except Exception as e:
                text = f"__ERROR__ {e}"
            res.append({"q": q, "reply": text, "ok": re.search(pat, text, re.I) is not None})
        out[sname] = res
        n_ok = sum(r["ok"] for r in res)
        print(f"  quality {sname}: {n_ok}/{len(res)}")
    return out


def drift_eval():
    prompts = [ln for ln in open(f"{E}/m4/prompts-eval.txt").read().splitlines() if ln.strip()]
    tops = []
    for p in prompts:
        body = {"prompt": p, "n_predict": 1, "temperature": 0, "n_probs": 1, "post_sampling_probs": False}
        try:
            r = http_json(f"http://127.0.0.1:{PORT}/completion", body, timeout=60)
            cp = r.get("completion_probabilities") or []
            tok = cp[0]["top_logprobs"][0]["token"] if cp and cp[0].get("top_logprobs") else r.get("content", "")[:8]
        except Exception as e:
            tok = f"__ERROR__{e}"
        tops.append(tok)
    return tops


# ---- HumanEval-30 (optional, time-permitting): mirrors m4/skip/hardscore.py's he_program()/he_ok()
# verbatim (same sandbox command: systemd-run --user --scope, MemoryMax=1G, TasksMax=32, network-
# isolated via unshare -rn, 10s timeout, python3 -I). Generated code is NEVER executed outside that
# sandboxed subprocess. Reuses the skip agent's data file directly instead of copying it.
HUMANEVAL_PATH = f"{E}/m4/skip/data/HumanEval.jsonl.gz"


def _he_load(n=30):
    items = [json.loads(ln) for ln in gzip.open(HUMANEVAL_PATH, "rt")]
    return items[:n]


def _he_prompt(item):
    return ("Complete the following Python function. Reply with the complete function (including the signature and any "
            "imports it needs) in a single ```python code block, and nothing else.\n\n```python\n" + item["prompt"] + "```")


def _he_program(reply, item):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", reply, re.S)
    code = blocks[0] if blocks else reply
    if re.search(rf"def\s+{re.escape(item['entry_point'])}\s*\(", code):
        imports = "\n".join(ln for ln in item["prompt"].splitlines() if re.match(r"\s*(import|from)\s", ln))
        prog = imports + "\n" + code
    else:
        prog = item["prompt"] + code
    return prog + "\n\n" + item["test"] + f"\n\ncheck({item['entry_point']})\n"


def _he_ok(reply, item):
    with tempfile.TemporaryDirectory() as d:
        open(f"{d}/prog.py", "w").write(_he_program(reply, item))
        cmd = ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=1G", "-p", "MemorySwapMax=0", "-p", "TasksMax=32",
               "unshare", "-rn", "timeout", "-k", "2", "10", "python3", "-I", "prog.py"]
        try:
            r = subprocess.run(cmd, cwd=d, capture_output=True, timeout=30, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return False
        return r.returncode == 0


def humaneval_eval(n=30):
    items = _he_load(n)
    res = []
    for i, item in enumerate(items):
        body = {"model": "default", "messages": [{"role": "user", "content": _he_prompt(item)}],
                "max_tokens": 512, "temperature": 0.0, "seed": 0}
        try:
            r = http_json(f"http://127.0.0.1:{PORT}/v1/chat/completions", body, timeout=300)
            reply = r["choices"][0]["message"]["content"] or ""
            ok = _he_ok(reply, item)
        except Exception as e:
            reply, ok = f"__ERROR__ {e}", False
        res.append({"idx": i, "task_id": item.get("task_id"), "ok": ok})
    n_ok = sum(r["ok"] for r in res)
    print(f"  humaneval{n}: pass@1 {n_ok}/{len(res)}")
    return res


def quick_decode_probe(path, ncmoe, log_path):
    """Fast decode-only load for the q4 n-cpu-moe sweep: warmup + 3x64-token samples, then stop."""
    p, logf = start_server(path, ncmoe, log_path)
    ok = wait_health(p, 220) and not log_has_oom(log_path)
    if not ok:
        stop_server(p, logf)
        return {"ncmoe": ncmoe, "fit": False}
    d = decode_bench(prompts=SHORT_PROMPTS[:3], n_predict=64)
    buf = parse_buffer_sizes(log_path)
    vram_smi = vram_used_mib_best_effort()
    stop_server(p, logf)
    return {"ncmoe": ncmoe, "fit": True, "decode_avg_tok_s": d["avg_tok_s"], "samples": d["per_prompt"],
            "cpu_load": d["cpu_load_before"], "cuda_buffers_mib": buf, "vram_nvidia_smi_mib": vram_smi}


def main():
    results_path = f"{L}/out/lowbit-results.json"
    results = json.load(open(results_path)) if os.path.exists(results_path) else {"runs": {}}
    baseline_drift = None
    if "q4kxl" in results["runs"] and results["runs"]["q4kxl"].get("drift_tokens"):
        baseline_drift = results["runs"]["q4kxl"]["drift_tokens"]

    # q4 n-cpu-moe sweep: window1's 22 tok/s at ncmoe=14 (--no-warmup, no sweep) looked wrong
    # against a historical ~61 tok/s at ncmoe=18. Re-measure with a warmup request across a few
    # ncmoe values so the IQ3/IQ2 comparison is against a real baseline, and record cpu_load so
    # contention from other agents sharing the box is visible if that's the actual cause.
    q4_run = RUNS[0]
    sweep_key = "q4_ncmoe_sweep"
    if sweep_key in results:
        sweep = results[sweep_key]
    elif left() > 500:
        print(f"== q4 n-cpu-moe sweep {Q4_NCMOE_CANDIDATES} ({left():.0f}s left) ==")
        sweep = []
        for nc in Q4_NCMOE_CANDIDATES:
            if left() < 400:
                print(f"  sweep: stopping early ({left():.0f}s left, reserving time for the full runs)")
                break
            r = quick_decode_probe(q4_run["file"], nc, f"{L}/out/q4kxl-sweep-ncmoe{nc}.server.log")
            print(f"  ncmoe={nc}: {r}")
            sweep.append(r)
        results[sweep_key] = sweep
        json.dump(results, open(results_path, "w"), indent=1)
    else:
        print(f"  skipping ncmoe sweep, only {left():.0f}s left")
        sweep = []
    fits = [s for s in sweep if s.get("fit") and s.get("decode_avg_tok_s")]
    if fits:
        best = max(fits, key=lambda s: s["decode_avg_tok_s"])
        q4_run["ncmoe"] = best["ncmoe"]
        print(f"  sweep winner: ncmoe={best['ncmoe']} ({best['decode_avg_tok_s']:.1f} tok/s) -> using for the full q4kxl run")
    else:
        print(f"  sweep produced nothing usable, keeping ncmoe={q4_run['ncmoe']} (historical reference)")

    CORE_RESERVE_PER_MODEL = 360  # seconds reserved per still-to-come model's core pipeline
    for idx, run in enumerate(RUNS):
        name = run["name"]
        remaining_after = len(RUNS) - (idx + 1)
        if name in results["runs"] and results["runs"][name].get("complete"):
            print(f"[{name}] already complete, skipping")
            if run["baseline"]:
                baseline_drift = results["runs"][name].get("drift_tokens")
            continue
        if left() < 240:
            print(f"[{name}] SKIPPED: only {left():.0f}s left in window")
            results["runs"][name] = {"skipped": True, "reason": "out of time"}
            json.dump(results, open(results_path, "w"), indent=1)
            continue

        print(f"== {name} ({left():.0f}s left) ==")
        loaded = load_with_search(run)
        rec = {"complete": False}
        if not loaded:
            rec["load_failed"] = True
            rec["vram_fit"] = False  # exhausted --n-cpu-moe search attempts without a healthy, OOM-free load
            results["runs"][name] = rec
            json.dump(results, open(results_path, "w"), indent=1)
            continue
        p, logf, ncmoe, buf, vram_smi, tried = loaded
        rec["ncmoe_used"] = ncmoe
        rec["cuda_buffers_mib"] = buf  # authoritative: parsed from llama-server's own load log
        rec["gpu_total_mib"] = GPU_TOTAL_MIB
        rec["vram_nvidia_smi_mib"] = vram_smi  # best-effort only; nvidia-smi/NVML is unreliable on this box
        rec["vram_fit"] = True  # reaching here means the server loaded healthy with no OOM in its log
        rec["ncmoe_attempts"] = tried

        # each phase gets its own try/except: one bad request must not erase the phases after it
        # (a 13k-token prompt once overshot CTX and 400'd, wiping quality+drift for every model)
        errors = []
        if left() > 180:
            print(f"  decode bench...")
            try:
                rec["decode"] = decode_bench()
            except Exception as e:
                errors.append(f"decode: {e}")
        if left() > 180:
            print(f"  prefill ~4k...")
            try:
                rec["prefill_4k"] = prefill_bench(4096)
            except Exception as e:
                errors.append(f"prefill_4k: {e}")
        if left() > 180:
            print(f"  prefill ~13k...")
            try:
                rec["prefill_13k"] = prefill_bench(13000)
            except Exception as e:
                errors.append(f"prefill_13k: {e}")
        if left() > 240:
            print(f"  quality...")
            try:
                rec["quality"] = quality_eval()
            except Exception as e:
                errors.append(f"quality: {e}")
        if left() > 90:
            print(f"  drift...")
            try:
                rec["drift_tokens"] = drift_eval()
                if run["baseline"]:
                    baseline_drift = rec["drift_tokens"]
                elif baseline_drift:
                    agree = sum(1 for a, b in zip(baseline_drift, rec["drift_tokens"]) if a == b)
                    rec["drift_agreement"] = f"{agree}/{len(baseline_drift)}"
            except Exception as e:
                errors.append(f"drift: {e}")
        # quality+drift for all three models is required; humaneval is "if time allows" - reserve
        # enough for the still-to-come models' own core pipeline before spending it here
        humaneval_gate = 400 + remaining_after * CORE_RESERVE_PER_MODEL
        if left() > humaneval_gate:
            print(f"  humaneval30 (sandboxed)...")
            try:
                rec["humaneval30"] = humaneval_eval(30)
            except Exception as e:
                errors.append(f"humaneval30: {e}")
        else:
            print(f"  skipping humaneval30, only {left():.0f}s left (need >{humaneval_gate}s, reserving for {remaining_after} more model(s))")
        rec["complete"] = not errors
        if errors:
            rec["errors"] = errors
        stop_server(p, logf)
        results["runs"][name] = rec
        json.dump(results, open(results_path, "w"), indent=1)
        print(f"[{name}] done: {json.dumps({k: v for k, v in rec.items() if k != 'quality'}, indent=1)}")

    print("== all runs finished or skipped ==")


if __name__ == "__main__":
    main()
