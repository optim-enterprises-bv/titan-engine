#!/usr/bin/env python3
"""Concurrency probe (titan /v1/completions, greedy).
usage: conc.py PORT PROMPTS.json OUT.json N_CONC MAX_TOKENS [FIRST]
Takes N_CONC prompts (starting at FIRST), runs them one at a time (sequential), then all at once (one thread each),
and reports per-request decode tok/s, the aggregate tok/s of the concurrent burst (sum of completion tokens / burst wall
time) vs the single-request rate, and whether each concurrent response equals its sequential text / tokens."""
import json, os, sys, threading, time, urllib.request

port, pf, out, n, mx = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
first = int(sys.argv[6]) if len(sys.argv) > 6 else 0
prompts = json.load(open(pf))[first:first + n]


def complete(p):
    body = {"model": os.environ.get("MODEL", "default"), "prompt": p, "max_tokens": mx, "temperature": 0, "seed": 0, "logprobs": 2}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    c = (r["choices"][0].get("logprobs") or {}).get("content") or []
    u = r.get("usage", {})
    return {"text": r["choices"][0].get("text", ""), "tokens": [x["token"] for x in c],
            "top2": [[(y["token"], y["logprob"]) for y in (x.get("top_logprobs") or [])][:2] for x in c],
            "completion_tokens": u.get("completion_tokens"), "prompt_tokens": u.get("prompt_tokens"),
            "decode_tps": u.get("avg_compl_tok_per_sec"), "t0": t, "t1": time.time()}


complete(prompts[0])  # warm-up (JIT / graphs)
seq = []
for p in prompts:
    seq.append(complete(p))
res = [None] * n


def worker(i):
    res[i] = complete(prompts[i])


th = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
t0 = time.time()
for t in th:
    t.start()
for t in th:
    t.join()
wall = time.time() - t0
seq_tok = sum(r["completion_tokens"] or 0 for r in seq)
seq_wall = sum(r["t1"] - r["t0"] for r in seq)
con_tok = sum(r["completion_tokens"] or 0 for r in res)
single = seq[0]["completion_tokens"] / (seq[0]["t1"] - seq[0]["t0"])
same_text = [a["text"] == b["text"] for a, b in zip(seq, res)]
same_tok = [a["tokens"] == b["tokens"] for a, b in zip(seq, res)]
first_diff = []
for a, b in zip(seq, res):
    k = 0
    for x, y in zip(a["tokens"], b["tokens"]):
        if x != y:
            break
        k += 1
    first_diff.append(k)
# at the first divergence: the sequential run's top-1 / top-2 logprob gap and the concurrent run's, at that position
gaps = []
for a, b, k in zip(seq, res, first_diff):
    if k < len(a["tokens"]) and k < len(b["tokens"]) and len(a["top2"][k]) == 2 and len(b["top2"][k]) == 2:
        gaps.append({"pos": k, "seq": a["top2"][k], "conc": b["top2"][k],
                     "seq_gap": round(a["top2"][k][0][1] - a["top2"][k][1][1], 4),
                     "conc_gap": round(b["top2"][k][0][1] - b["top2"][k][1][1], 4)})
    else:
        gaps.append(None)
summary = {
    "n": n, "max_tokens": mx,
    "sequential": {"tokens": seq_tok, "wall_s": round(seq_wall, 3), "agg_tps": round(seq_tok / seq_wall, 2),
                   "per_request_decode_tps": [r["decode_tps"] for r in seq]},
    "single_request_tps_wall": round(single, 2),
    "concurrent": {"tokens": con_tok, "wall_s": round(wall, 3), "agg_tps": round(con_tok / wall, 2),
                   "per_request_decode_tps": [r["decode_tps"] for r in res],
                   "overlap_s": round(min(r["t1"] for r in res) - max(r["t0"] for r in res), 3)},
    "speedup_vs_single": round((con_tok / wall) / single, 3),
    "divergence_gaps": gaps,
    "text_identical": same_text, "tokens_identical": same_tok, "common_prefix_tokens": first_diff,
}
json.dump({"summary": summary, "sequential": seq, "concurrent": res}, open(out, "w"), ensure_ascii=False, indent=1)
print(f"CONC n={n} max_tokens={mx}: single {single:.1f} tok/s (wall), sequential agg {seq_tok / seq_wall:.1f} tok/s, "
      f"concurrent agg {con_tok / wall:.1f} tok/s ({con_tok} tok in {wall:.2f}s, x{(con_tok / wall) / single:.2f} vs single), "
      f"per-request decode {summary['concurrent']['per_request_decode_tps']}")
print(f"CONC identity vs sequential: text {sum(same_text)}/{n}, tokens {sum(same_tok)}/{n}, common prefix {first_diff} "
      f"(of {[len(r['tokens']) for r in seq]})")
print("CONC first-divergence top1-top2 logprob gaps (sequential / concurrent): " + ", ".join(
    "-" if g is None else f"pos {g['pos']}: {g['seq_gap']} / {g['conc_gap']}" for g in gaps))
