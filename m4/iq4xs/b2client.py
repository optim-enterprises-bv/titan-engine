#!/usr/bin/env python3
"""Client for the Bonsai 2 27B gates, the same method against titan-mistral and the reference llama-server.

    b2client.py greedy KIND PORT NAME [N=8] [MAX=300]  raw completions, greedy: out/NAME.json (texts, tokens, tok/s)
    b2client.py top KIND PORT NAME                      first-token top-20 logprobs of the 8 prompts: out/NAME.top.json
    b2client.py cmp A B                                 text identity of two greedy runs
    b2client.py kl REF OURS                             first-token top-5 agreement + KL(ref || ours) on ref's top-20
    b2client.py probe KIND PORT NAME [greedy|sampled]   the author's probe.py (3 prompts x 3 runs, 400 tokens,
                                                        thinking off), client-measured decode tok/s, medians
    b2client.py long KIND PORT NAME TOKENS [greedy|sampled]  a TOKENS-long cold prompt (prefill tok/s), then 3 x
                                                        200-token decodes at that depth (tok/s)
KIND: llama (llama-server) or titan. Prompts use one fixed raw chat rendering (thinking off), so both engines see the
same text; every timed request is streamed and decode tok/s = (completion tokens - 1) / (last - first token time)."""
import json, math, os, statistics as st, sys, time, urllib.request, uuid

B = os.path.dirname(os.path.abspath(__file__))
PROMPTS = [l.strip() for l in open(os.path.expanduser("~/titan-engine/m4/prompts-eval.txt")) if l.strip()]
PROBE = [
    "write a python function that merges two sorted lists into one sorted list, with docstring.",
    "explain the difference between mmap and read for loading large files, one paragraph.",
    "write a bash script that watches a directory and prints new files as they appear.",
]
SAMPLED = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
GREEDY = {"temperature": 0.0, "top_k": 1}


def render(user, system=None):
    s = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    return s + f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def post(port, path, body, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def complete(kind, port, prompt, n, samp, stream=True):
    """Raw completion; returns (text, completion_tokens, ttft, decode tok/s, prompt tokens, server prompt tok/s)."""
    body = dict(samp, prompt=prompt, max_tokens=n, seed=0, stream=stream)
    if kind == "titan":
        body["model"] = "default"
    if stream:
        body["stream_options"] = {"include_usage": True}
    t0 = time.time()
    r = post(port, "/v1/completions", body)
    if not stream:
        d = json.load(r)
        u, tm = d.get("usage") or {}, d.get("timings") or {}
        pps = tm.get("prompt_per_second") or ((u["prompt_tokens"] / u["total_prompt_time_sec"]) if u.get("total_prompt_time_sec") else None)
        return d["choices"][0]["text"], u.get("completion_tokens"), time.time() - t0, None, u.get("prompt_tokens") or tm.get("prompt_n"), pps
    text, t_first, t_last, usage, chunks, timings = [], None, None, None, 0, None
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        d = json.loads(data)
        if d.get("usage"):
            usage = d["usage"]
        if d.get("timings"):
            timings = d["timings"]
        for ch in d.get("choices") or []:
            piece = ch.get("text") or ""
            if piece:
                now = time.time()
                t_first = t_first or now
                t_last = now
                chunks += 1
                text.append(piece)
    # one streamed chunk per sampled token on both servers (MTP included); usage is not in titan's completion chunks
    ntok = chunks
    dec = (ntok - 1) / (t_last - t_first) if t_first and t_last and t_last > t_first and ntok > 1 else None
    ptoks = (usage or {}).get("prompt_tokens")
    pps = None
    if timings and timings.get("prompt_per_second"):
        pps = timings["prompt_per_second"]
    elif usage and usage.get("prompt_tokens") and usage.get("total_prompt_time_sec"):
        pps = usage["prompt_tokens"] / usage["total_prompt_time_sec"]
    return "".join(text), ntok, (t_first - t0) if t_first else None, dec, ptoks, pps


def greedy(kind, port, name, n=8, mx=300):
    out, rates = [], []
    for p in PROMPTS[:n]:
        text, ntok, ttft, dec, _, _ = complete(kind, port, render(p), mx, GREEDY)
        out.append({"prompt": p, "text": text, "tokens": ntok, "decode_tok_s": dec})
        if dec:
            rates.append(dec)
    json.dump(out, open(f"{B}/out/{name}.json", "w"), ensure_ascii=False, indent=1)
    print(f"{name}: {len(out)} x {mx} greedy, decode median {st.median(rates):.1f} tok/s" if rates else f"{name}: no rates")


def top(kind, port, name, n=8):
    res = []
    for p in PROMPTS[:n]:
        pr = render(p)
        if kind == "llama":
            d = json.load(post(port, "/completion", {"prompt": pr, "n_predict": 1, "temperature": 0, "n_probs": 20, "post_sampling_probs": False}))
            top = [(t["token"], t["logprob"]) for t in d["completion_probabilities"][0]["top_logprobs"]]
        else:
            d = json.load(post(port, "/v1/completions", {"model": "default", "prompt": pr, "max_tokens": 1, "temperature": 0, "logprobs": 20}))
            lp = d["choices"][0]["logprobs"] or {}
            c = lp.get("content") or []
            if c:
                top = [(t["token"], t["logprob"]) for t in c[0]["top_logprobs"]]
            else:  # legacy completions logprobs layout
                top = sorted(((k, v) for k, v in (lp.get("top_logprobs") or [{}])[0].items()), key=lambda x: -x[1])
        res.append(top)
    vp = f"{B}/out/vocab.json"
    if os.path.exists(vp):  # titan reports token ids: store the strings, as llama-server does
        vocab = json.load(open(vp))
        res = [[(vocab[t] if isinstance(t, int) else t, l) for t, l in row] for row in res]
    json.dump(res, open(f"{B}/out/{name}.top.json", "w"), ensure_ascii=False)
    print(name, "top done")


def cmp(a, b):
    A, Bb = json.load(open(f"{B}/out/{a}.json")), json.load(open(f"{B}/out/{b}.json"))
    same = 0
    for i, (x, y) in enumerate(zip(A, Bb)):
        if x["text"] == y["text"]:
            same += 1
        else:
            k = next((j for j in range(min(len(x["text"]), len(y["text"]))) if x["text"][j] != y["text"][j]), min(len(x["text"]), len(y["text"])))
            print(f"  prompt {i}: diverge at char {k}: {x['text'][max(0,k-40):k+40]!r} vs {y['text'][max(0,k-40):k+40]!r}")
    print(f"GATE text identity {b} vs {a}: {same}/{len(A)}")
    return same, len(A)


def kl(ref, ours):
    R, O = json.load(open(f"{B}/out/{ref}.top.json")), json.load(open(f"{B}/out/{ours}.top.json"))
    agree1 = agree5 = 0
    kls = []
    for r, o in zip(R, O):
        r = sorted(r, key=lambda t: -t[1]); o = sorted(o, key=lambda t: -t[1])
        agree1 += r[0][0] == o[0][0]
        agree5 += set(t for t, _ in r[:5]) == set(t for t, _ in o[:5])
        om = dict(o)
        floor = min(v for _, v in o) - math.log(10)  # tokens missing from ours: below its 20th
        pr = [math.exp(v) for _, v in r]
        zr = sum(pr)
        qo = [math.exp(om.get(t, floor)) for t, _ in r]
        zo = sum(qo)
        kls.append(sum((p / zr) * math.log((p / zr) / (q / zo)) for p, q in zip(pr, qo)))
    n = len(R)
    print(f"GATE first token vs {ref}: top-1 {agree1}/{n}, top-5 set {agree5}/{n}, KL(ref||ours) on ref top-20 mean {st.mean(kls):.4f} max {max(kls):.4f}")


def probe(kind, port, name, mode="sampled"):
    samp = SAMPLED if mode == "sampled" else GREEDY
    complete(kind, port, render("warmup"), 40, samp)
    allr, rows = [], []
    for p in PROBE:
        rs = []
        for _ in range(3):
            _, ntok, _, dec, _, _ = complete(kind, port, render(p), 400, samp)
            rs.append(dec or 0.0)
        allr += rs
        rows.append({"prompt": p, "runs": rs, "median": st.median(rs)})
        print(f"{st.median(rs):6.1f} tok/s median | runs: {[round(x, 1) for x in rs]} | {p[:50]}", flush=True)
    res = {"name": name, "mode": mode, "rows": rows, "mean": st.mean(allr), "median": st.median(allr)}
    json.dump(res, open(f"{B}/out/{name}.probe.json", "w"), indent=1)
    print(f"PROBE {name} ({mode}): mean {res['mean']:.1f} median {res['median']:.1f} tok/s")


def long(kind, port, name, tokens, mode="greedy"):
    samp = SAMPLED if mode == "sampled" else GREEDY
    corpus = open(os.path.expanduser("~/titan-engine/bench/prompts/corpus-sys28k.txt")).read()
    # ~3.9 chars per token on this corpus; repeat it with a nonce per copy to reach the depth
    reps, body = 0, []
    while sum(len(b) for b in body) < tokens * 3.9:
        body.append(f"[part {reps} {uuid.uuid4().hex[:8]}] " + corpus)
        reps += 1
    sysmsg = "".join(body)[: int(tokens * 3.9)]
    prompt = render("Summarise the system text in three sentences.", system=f"[{uuid.uuid4().hex}] " + sysmsg)
    t0 = time.time()
    text, ntok, ttft, dec, ptoks, pps = complete(kind, port, prompt, 1, samp, stream=False)
    wall = time.time() - t0
    res = {"name": name, "tokens_target": tokens, "prompt_tokens": ptoks, "prefill_wall_s": wall, "server_prompt_tok_s": pps,
           "client_prompt_tok_s": (ptoks / wall) if ptoks else None, "decode": []}
    print(f"LONG {name}: {ptoks} prompt tokens, prefill {wall:.1f}s client ({res['client_prompt_tok_s'] or 0:.0f} tok/s), server {pps or 0:.0f} tok/s", flush=True)
    for i in range(3):
        _, ntok, _, dec, _, _ = complete(kind, port, prompt, 200, samp)
        res["decode"].append(dec)
        print(f"  decode at depth: {dec or 0:.1f} tok/s ({ntok} tokens)", flush=True)
    res["decode_median"] = st.median([d for d in res["decode"] if d] or [0])
    json.dump(res, open(f"{B}/out/{name}.long.json", "w"), indent=1)
    print(f"LONG {name}: decode median {res['decode_median']:.1f} tok/s at {ptoks} tokens")


if __name__ == "__main__":
    a = sys.argv[1:]
    {"greedy": lambda: greedy(a[1], a[2], a[3], *(int(x) for x in a[4:6])),
     "top": lambda: top(a[1], a[2], a[3]),
     "cmp": lambda: cmp(a[1], a[2]),
     "kl": lambda: kl(a[1], a[2]),
     "probe": lambda: probe(a[1], a[2], a[3], *(a[4:5])),
     "long": lambda: long(a[1], a[2], a[3], int(a[4]), *(a[5:6])),
     }[a[0]]()
