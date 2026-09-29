import json, sys, time, urllib.request
name, port, max_tokens, pf, nprom, big = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), sys.argv[6] == "1"
def post(body, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))
prompts = [l.strip() for l in open(pf) if l.strip()][:nprom]
res, t0, toks, dec, srv = [], time.time(), 0, [], []
import os
deadline = float(os.environ.get("UB_DEADLINE", "0") or 0)
fails = 0
for p in prompts:
    if deadline and time.time() > deadline:
        print(f"deadline reached after {len(res)} prompts"); break
    if fails >= 2:
        print("2 failed requests: stopping"); break
    t = time.time()
    try:
        r = post({"model": "default", "messages": [{"role": "user", "content": p}],
                  "temperature": 0.0, "max_tokens": max_tokens, "seed": 0})
    except Exception as e:
        print("request FAILED:", e); res.append(None); fails += 1; continue
    res.append(r["choices"][0]["message"]["content"])
    json.dump(res, open(f"out/{name}.json", "w"), ensure_ascii=False, indent=1)
    u = r.get("usage", {}); n = u.get("completion_tokens", 0)
    toks += n; dec.append(n / (time.time() - t)); srv.append(u.get("avg_compl_tok_per_sec", 0))
dt = time.time() - t0
json.dump(res, open(f"out/{name}.json", "w"), ensure_ascii=False, indent=1)
med = lambda v: sorted(v)[len(v)//2] if v else 0
print(f"{name}: {len(res)} completions, {toks} tokens, {dt:.1f}s, {toks/dt:.1f} tok/s (median per-request {med(dec):.1f}, server avg_compl median {med(srv):.1f})")
if big and fails < 2:
    chunk = "The titan engine tiers mixture-of-experts weights between the GPU and the CPU. "
    for n in (4000, 4000, 13000, 13000):
        text = chunk * (n // 16)
        body = {"model": "default", "messages": [{"role": "system", "content": text}, {"role": "user", "content": "Summarise the system text in one line."}], "max_tokens": 32, "temperature": 0}
        t = time.time()
        try:
            u = post(body, 900)["usage"]
            c = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            print(f"big {n}: prompt_tokens={u['prompt_tokens']} cached={c} prompt_tok_s={u.get('avg_prompt_tok_per_sec'):.1f} prompt_time={u.get('total_prompt_time_sec'):.2f}s wall={time.time()-t:.1f}s")
        except Exception as e:
            print(f"big {n}: FAILED {e}")
