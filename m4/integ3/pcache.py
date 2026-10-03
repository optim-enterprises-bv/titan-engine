#!/usr/bin/env python3
"""m4/pcache client for a from-config (swap) server; every request names its model.
  pcache.py long PORT MODEL NAME COUNT [CHARS=13500] [REPEAT=0]
      COUNT ~4k-token prompts (m4/g4s/gates.py's G3 construction from idle/tf/docs.json, so the first 4 are the G3 set),
      then REPEAT re-sends of the last one (a prefix-cache hit if the cache kept it); /v1/completions, 1 token, top-20,
      the gates.py body. Per request: ok / error (OOM), prompt tokens, wall time, nvidia-smi used MiB. out/NAME.json
  pcache.py pc PORT MODEL NAME     m4/integ/pc.py (8 chat requests sharing a ~6.4k-token system prefix, 64 greedy tokens,
                                   TTFT) with the model named. out/NAME.pc.json
  pcache.py touch PORT MODEL         one 1-token request (swaps MODEL in), then nvidia-smi used MiB
  pcache.py same A.json B.json     top-20 identity of the first-token lists both files have (gates.py format)"""
import json, subprocess, sys, time, urllib.error, urllib.request

E = "~/titan-engine/m4"
OUT = f"{E}/../top-integ3/m4/integ3/out"


def smi():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(r.stdout.strip().splitlines()[0]) if r.returncode == 0 and r.stdout.strip() else -1


def post(port, path, body, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def long_prompts(chars, count):  # gates.py long_prompts
    docs = [d["text"] for d in json.load(open(f"{E}/idle/tf/docs.json"))]
    blob = "\n\n".join(docs * 4)
    step = len(docs[0]) * 5
    return [blob[i * step:i * step + chars] + "\n\nIn one sentence, the text above is about" for i in range(count)]


def long(port, model, name, count, chars=13500, repeat=0):
    ps = long_prompts(chars, count)
    ps += [ps[-1]] * repeat
    res = []
    for i, p in enumerate(ps):
        body = {"model": model, "prompt": p, "max_tokens": 1, "temperature": 0, "seed": 0, "logprobs": 20}
        t = time.time()
        row = {"i": i, "chars": len(p)}
        try:
            r = json.load(post(port, "/v1/completions", body))
            c = (r["choices"][0].get("logprobs") or {}).get("content") or []
            u = r.get("usage", {})
            row.update(ok=True, prompt_tokens=u.get("prompt_tokens"), prefill_tps=u.get("avg_prompt_tok_per_sec"),
                       tokens=[x["token"] for x in c],
                       top=[[(y["token"], y["logprob"]) for y in (x.get("top_logprobs") or [])] for x in c])
        except urllib.error.HTTPError as e:
            row.update(ok=False, error=f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
        except Exception as e:  # noqa: BLE001 - recorded, the run goes on
            row.update(ok=False, error=repr(e)[:300])
        row.update(wall=round(time.time() - t, 2), vram_mib=smi())
        res.append(row)
        print(f"  {name}[{i}] {'ok ' if row['ok'] else 'ERR'} prompt {row.get('prompt_tokens')} wall {row['wall']:.2f}s "
              f"VRAM {row['vram_mib']} MiB {row.get('error', '')[:160]}", flush=True)
    json.dump(res, open(f"{OUT}/{name}.json", "w"), ensure_ascii=False)
    ok = sum(r["ok"] for r in res)
    oom = sum("memory" in r.get("error", "").lower() for r in res)
    print(f"LONG {name} ({model}): {ok}/{len(res)} ok, {oom} OOM errors, max VRAM {max(r['vram_mib'] for r in res)} MiB")


def pc(port, model, name):  # m4/integ/pc.py with the model named
    CHUNK = "The titan engine tiers mixture-of-experts weights between the GPU and the CPU. "
    SYS = CHUNK * 400
    QS = ["Summarise the system text in one line.", "What does the text say about the GPU?", "Count the sentences, roughly.",
          "Name one word that repeats.", "Is the text about cooking?", "Give the text a title.",
          "What is tiered here?", "Answer in French: what is the text about?"]
    res = []
    for q in QS:
        body = {"model": model, "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": q}],
                "max_tokens": 64, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
        t0 = time.time(); first = None; last = None; n = 0; usage = {}
        with post(port, "/v1/chat/completions", body) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                d = json.loads(line[5:])
                if d.get("usage"):
                    usage = d["usage"]
                for c in d.get("choices") or []:
                    if (c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning_content"):
                        now = time.time(); first = first or now; last = now; n += 1
        ttft = (first or time.time()) - t0
        dec = (n - 1) / (last - first) if n > 1 and last > first else 0.0
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        res.append({"q": q, "prompt_tokens": usage.get("prompt_tokens"), "cached": cached, "ttft": ttft, "decode_tps": dec, "chunks": n})
        print(f"  {name}: prompt {usage.get('prompt_tokens')} cached {cached} ttft {ttft:.2f}s decode {dec:.1f} tok/s", flush=True)
    json.dump(res, open(f"{OUT}/{name}.pc.json", "w"), indent=1)
    t = sorted(r["ttft"] for r in res[1:]); d = sorted(r["decode_tps"] for r in res)
    print(f"PC {name}: first ttft {res[0]['ttft']:.2f}s, repeats (7) ttft median {t[len(t)//2]:.2f}s sum {sum(t):.1f}s, "
          f"decode median {d[len(d)//2]:.1f} tok/s")


def touch(port, model):
    t = time.time()
    body = {"model": model, "prompt": "Hello", "max_tokens": 1, "temperature": 0, "seed": 0}
    try:
        json.load(post(port, "/v1/completions", body)); ok = "ok"
    except Exception as e:  # noqa: BLE001
        ok = f"ERR {e!r}"[:200]
    print(f"TOUCH {model}: {ok} in {time.time() - t:.1f}s; VRAM used {smi()} MiB", flush=True)


def same(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    n = min(len(A), len(B))
    # tokens are ids here and strings in tokmap'd gates.py files: compare the top-20 logprob values, in order
    lps = lambda r: [[lp for _, lp in step] for step in (r.get("top") or [])]
    eq = sum(x.get("ok", True) and y.get("ok", True) and lps(x) == lps(y) and x.get("prompt_tokens") == y.get("prompt_tokens")
             for x, y in zip(A[:n], B[:n]))
    print(f"SAME {a.split('/')[-1]} vs {b.split('/')[-1]}: top-20 logprobs + prompt_tokens identical {eq}/{n}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "long":
        long(a[1], a[2], a[3], int(a[4]), int(a[5]) if len(a) > 5 else 13500, int(a[6]) if len(a) > 6 else 0)
    elif a[0] == "pc":
        pc(a[1], a[2], a[3])
    elif a[0] == "touch":
        touch(a[1], a[2])
    elif a[0] == "same":
        same(a[1], a[2])
    else:
        sys.exit(__doc__)
