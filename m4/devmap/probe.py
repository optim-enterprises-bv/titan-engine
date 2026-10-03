#!/usr/bin/env python3
"""Send chat prompts of ~N tokens and record VRAM. usage: probe.py PORT OUT.json N1[,N2,...] [MAX_TOKENS] [MODEL]
Each prompt is the same sentence repeated (calibrated on a short request to land near N prompt tokens); nvidia-smi is
polled every 50 ms during the request. Per prompt: prompt tokens, ok / error, nvidia-smi used before / peak / after,
prompt and decode tok/s, the first words of the reply."""
import json, subprocess, sys, threading, time, urllib.error, urllib.request

port, out, ns = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 32
model = sys.argv[5] if len(sys.argv) > 5 else "default"
CH = "The titan engine tiers mixture-of-experts weights between the GPU and the CPU. "


def smi():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    try:
        return int(r.stdout.split()[0])
    except Exception:  # noqa: BLE001
        return -1


def chat(reps, n):
    body = {"model": model, "messages": [{"role": "system", "content": CH * reps},
                                         {"role": "user", "content": "Summarise the system text in one line."}],
            "max_tokens": n, "temperature": 0}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


cal = chat(64, 1)["usage"]["prompt_tokens"]
base = chat(0, 1)["usage"]["prompt_tokens"]
per = (cal - base) / 64
rows = []
for n in ns:
    reps = max(1, int((n - base) / per))
    peak, stop = [smi()], threading.Event()
    before = peak[0]

    def poll():
        while not stop.is_set():
            peak.append(smi()); time.sleep(0.05)
    th = threading.Thread(target=poll, daemon=True); th.start()
    t = time.time(); row = {"target": n}
    try:
        r = chat(reps, max_tokens); u = r["usage"]
        row.update(ok=True, prompt_tokens=u["prompt_tokens"], completion_tokens=u["completion_tokens"],
                   prompt_tps=u.get("avg_prompt_tok_per_sec"), decode_tps=u.get("avg_compl_tok_per_sec"),
                   text=(r["choices"][0]["message"].get("content") or "")[:120])
    except urllib.error.HTTPError as e:
        row.update(ok=False, error=f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
    except Exception as e:  # noqa: BLE001
        row.update(ok=False, error=repr(e)[:300])
    stop.set(); th.join()
    row.update(wall_s=round(time.time() - t, 1), smi_before=before, smi_peak=max(peak), smi_after=smi())
    rows.append(row)
    print(f"PROBE {out.split('/')[-1]} n~{n}: {'ok' if row['ok'] else 'FAILED'} prompt {row.get('prompt_tokens')} tok, "
          f"VRAM before {before} peak {row['smi_peak']} after {row['smi_after']} MiB, prefill {row.get('prompt_tps')} "
          f"decode {row.get('decode_tps')} tok/s, {row['wall_s']} s {row.get('error', '')[:200]}", flush=True)
    json.dump(rows, open(out, "w"), indent=1)
