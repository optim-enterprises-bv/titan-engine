#!/usr/bin/env python3
"""IQ4_XS end-to-end gate (copied from m4/iq3s): llama.cpp vs mistral.rs (titan) on the same GGUF.

  e2e.py llama   PORT TAG   greedy 64-token raw completions + teacher-forced top-5 at every prefix
                            of llama.cpp's own greedy continuation (token boundaries from /tokenize)
  e2e.py mistral PORT TAG   the same prefixes and greedy runs against mistral.rs (/v1/completions)
  e2e.py cmp     TAG        greedy text identity, teacher-forced top-1 agreement, logprob deltas

Writes out/TAG_{llama,mistral}.json. Prefix list comes from out/TAG_llama.json, so run llama first.
"""
import json, sys, time, urllib.request, math

PROMPTS = [
    "The capital of France is",
    'def fibonacci(n):\n    """Return the nth Fibonacci number."""\n',
    "In Rust, the borrow checker ensures that",
    "A mixture-of-experts layer routes each token to",
    "The three laws of thermodynamics are",
    "SELECT name, COUNT(*) FROM orders GROUP BY",
    "WireGuard is a VPN protocol that",
    "Once upon a time, in a small village by the sea,",
]
N = 64


def post(port, path, body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def llama_top(port, prompt, k=5):
    r = post(port, "/completion", {"prompt": prompt, "n_predict": 1, "temperature": 0, "n_probs": k, "post_sampling_probs": False, "cache_prompt": False})
    return [(t["token"], t["logprob"]) for t in r["completion_probabilities"][0]["top_logprobs"]]


def mistral_top(port, prompt, k=5):
    r = post(port, "/v1/completions", {"model": "default", "prompt": prompt, "max_tokens": 1, "temperature": 0, "logprobs": k})
    lp = r["choices"][0].get("logprobs") or {}
    c = lp.get("content") or []
    if not c:
        return [(r["choices"][0]["text"], 0.0)]
    return [(t["token"], t["logprob"]) for t in c[0]["top_logprobs"]]


def greedy(port, prompt, kind):
    t = time.time()
    if kind == "llama":
        r = post(port, "/completion", {"prompt": prompt, "n_predict": N, "temperature": 0, "top_k": 1, "cache_prompt": False})
        txt, n = r["content"], r.get("tokens_predicted", 0)
    else:
        r = post(port, "/v1/completions", {"model": "default", "prompt": prompt, "max_tokens": N, "temperature": 0, "top_k": 1, "seed": 0})
        txt, n = r["choices"][0]["text"], r.get("usage", {}).get("completion_tokens", 0)
    return txt, n, time.time() - t


def main():
    mode = sys.argv[1]
    if mode == "cmp":
        tag = sys.argv[2]
        a = json.load(open(f"out/{tag}_llama.json")); b = json.load(open(f"out/{tag}_mistral.json"))
        # mistral.rs reports top_logprobs tokens as vocabulary ids: decode them with the GGUF's
        # byte-level BPE vocabulary (argv[3]) so they compare with llama.cpp's token texts.
        if len(sys.argv) > 3:
            import types
            sys.modules.setdefault("yaml", types.ModuleType("yaml")); sys.path.insert(0, "~/ai/llama.cpp/gguf-py")
            from gguf import GGUFReader
            f = GGUFReader(sys.argv[3]).fields["tokenizer.ggml.tokens"]
            vocab = [bytes(f.parts[i]).decode("utf-8") for i in f.data]
            bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)); cs = bs[:]; n = 0
            for x in range(256):
                if x not in bs: bs.append(x); cs.append(256 + n); n += 1
            inv = {chr(c): x for x, c in zip(bs, cs)}
            def dec(t):
                try: t = int(t)
                except (TypeError, ValueError): return t
                return bytes(inv.get(ch, 63) for ch in vocab[t]).decode("utf-8", "replace")
            b["tf"] = [[[[dec(t), l] for t, l in pos] for pos in pr] for pr in b["tf"]]
        same = 0
        for i, (x, y) in enumerate(zip(a["greedy"], b["greedy"])):
            x, y = x["text"].lstrip(), y["text"].lstrip()
            k = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
            ok = x == y; same += ok
            print(f"  greedy {i}: {'identical' if ok else f'diverges at char {k}/{len(x)}: {x[k:k+40]!r} vs {y[k:k+40]!r}'}")
        agree = tot = in5 = 0; dl = []
        for pa, pb in zip(a["tf"], b["tf"]):
            for ta, tb in zip(pa, pb):
                tot += 1
                la, lb = ta[0][0], tb[0][0]
                agree += la == lb
                in5 += la in [t for t, _ in tb]
                d = dict(tb)
                if la in d:
                    dl.append(abs(ta[0][1] - d[la]))
        tok = lambda r: sum(g["n"] for g in r["greedy"]) / max(1e-9, sum(g["secs"] for g in r["greedy"]))
        dl.sort()
        print(f"{tag}: greedy 64-token texts identical {same}/{len(a['greedy'])} (leading whitespace ignored)")
        print(f"{tag}: teacher-forced top-1 agreement {agree}/{tot} = {100*agree/max(1,tot):.1f}%, llama top-1 in mistral top-5 {in5}/{tot}")
        if dl:
            print(f"{tag}: |logprob(llama top-1) llama - mistral| mean {sum(dl)/len(dl):.4f} median {dl[len(dl)//2]:.4f} p95 {dl[int(0.95*(len(dl)-1))]:.4f} max {dl[-1]:.4f} (n={len(dl)})")
        print(f"{tag}: greedy decode incl. prefill: llama {tok(a):.1f} tok/s, mistral {tok(b):.1f} tok/s")
        json.dump({"greedy_identical": same, "greedy_total": len(a["greedy"]), "tf_top1": agree, "tf_total": tot, "tf_in_top5": in5,
                   "dlogp_mean": sum(dl)/len(dl) if dl else None, "dlogp_max": dl[-1] if dl else None,
                   "llama_tok_s": tok(a), "mistral_tok_s": tok(b)}, open(f"out/{tag}_cmp.json", "w"), indent=1)
        return
    port, tag = sys.argv[2], sys.argv[3]
    res = {"greedy": [], "tf": [], "prefixes": []}
    if mode == "llama":
        for p in PROMPTS:
            txt, n, s = greedy(port, p, "llama")
            res["greedy"].append({"prompt": p, "text": txt, "n": n, "secs": s})
            ids = post(port, "/tokenize", {"content": txt, "add_special": False})["tokens"]
            pref = [p + post(port, "/detokenize", {"tokens": ids[:i]})["content"] for i in range(min(N, len(ids)))]
            res["prefixes"].append(pref)
            res["tf"].append([llama_top(port, q) for q in pref])
            print(f"llama: {p[:30]!r}: {n} tokens in {s:.2f}s, {len(pref)} teacher-forced positions", flush=True)
        # the vocabulary for decoding mistral.rs's token ids in cmp
        json.dump(res, open(f"out/{tag}_llama.json", "w"), ensure_ascii=False, indent=1)
    else:
        ref = json.load(open(f"out/{tag}_llama.json"))
        for p, pref in zip(PROMPTS, ref["prefixes"]):
            txt, n, s = greedy(port, p, "mistral")
            res["greedy"].append({"prompt": p, "text": txt, "n": n, "secs": s})
            res["tf"].append([mistral_top(port, q) for q in pref])
            print(f"mistral: {p[:30]!r}: {n} tokens in {s:.2f}s", flush=True)
        json.dump(res, open(f"out/{tag}_mistral.json", "w"), ensure_ascii=False, indent=1)


main()
