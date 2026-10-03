#!/usr/bin/env python3
"""E2E client for bench/run.sh. Talks to one server (OpenAI chat API, streaming) and writes one JSON.

    client.py PLAN PORT OUT.json [nsys-session nsys-out]

PLAN
  service  warm-up; 8 eval prompts x 256 greedy, twice (decode tok/s; identity vs ref/h-off.first8.json);
           TTFT of the 8 prompts, twice, each cold (nonce prefix, max_tokens 1); ~4k and ~13k cold prompts
           twice each (prompt tok/s, TTFT); the last 13k prompt re-sent twice (prefix-cache warm TTFT).
  mtpoff   warm-up; the 8 prompts twice (identity vs h-off); TTFT twice.
  ident    the 8 prompts once, no warm-up (as m3/collect.sh), identity vs ref/q35-prof.first8.json.
  nsys     warm-up; `nsys start`; twice: one ~2.1k-token cold prompt + 32 decode tokens; `nsys stop`.

Every request is streamed: TTFT = send -> first content chunk; decode tok/s (client) = (completion_tokens - 1) /
(last chunk - first chunk). The server's usage (last chunk) is kept too. Identity compares the concatenated stream
text; a mismatch is re-checked with a non-streamed request (reported separately) so a stream artefact is not
mistaken for an engine change."""
import json, os, statistics, subprocess, sys, time, urllib.request, uuid

B = os.path.expanduser("~/titan-engine/bench")
MODEL = os.environ.get("BENCH_MODEL", "default")
NSYS = "/opt/nvidia/nsight-compute/2026.2.1/host/target-linux-x64/nsys"
PROMPTS = [l.strip() for l in open(f"{B}/prompts/eval8.txt") if l.strip()][:8]
CORPUS = open(f"{B}/prompts/corpus-sys28k.txt").read()
CHARS = {"4k": 15600, "13k": 50800, "2k": 8200}


def post(port, body, stream=True, timeout=1800):
    body = dict(body, model=MODEL, temperature=0.0, seed=0, stream=stream)
    if stream:
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    r = urllib.request.urlopen(req, timeout=timeout)
    if not stream:
        d = json.load(r)
        m = d["choices"][0]["message"]
        return {"text": (m.get("reasoning_content") or "") + (m.get("content") or ""), "usage": d.get("usage"),
                "wall": time.time() - t0}
    text, t_first, t_last, usage, nchunks = [], None, None, None, 0
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
        for ch in d.get("choices") or []:
            delta = ch.get("delta") or {}
            piece = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
            if piece:
                now = time.time()
                t_first = t_first or now
                t_last = now
                nchunks += 1
                text.append(piece)
    t_end = time.time()
    ntok = (usage or {}).get("completion_tokens") or nchunks
    dec = (ntok - 1) / (t_last - t_first) if t_first and t_last and t_last > t_first and ntok > 1 else None
    return {"text": "".join(text), "usage": usage, "ttft": (t_first - t0) if t_first else None, "wall": t_end - t0,
            "completion_tokens": ntok, "decode_tok_s": dec, "chunks": nchunks}


def chat(p):
    return {"messages": [{"role": "user", "content": p}]}


def longp(nonce, chars):
    return {"messages": [{"role": "system", "content": f"[bench {nonce}] " + CORPUS[:chars]},
                         {"role": "user", "content": "Summarise the system text in one line."}]}


def warm(port):
    post(port, dict(chat("Say hi."), max_tokens=16))


def eight(port, ref=None):
    rs = [post(port, dict(chat(p), max_tokens=256)) for p in PROMPTS]
    toks = sum(r["completion_tokens"] - 1 for r in rs if r["decode_tok_s"])
    span = sum((r["completion_tokens"] - 1) / r["decode_tok_s"] for r in rs if r["decode_tok_s"])
    srv = [r["usage"]["avg_compl_tok_per_sec"] for r in rs if r.get("usage")]
    out = {"decode_tok_s": toks / span if span else None,
           "decode_tok_s_server_median": statistics.median(srv) if srv else None,
           "ttft_ms_median": 1e3 * statistics.median(r["ttft"] for r in rs),
           "completion_tokens": sum(r["completion_tokens"] for r in rs),
           "wall_s": sum(r["wall"] for r in rs), "texts": [r["text"] for r in rs]}
    if ref:
        refs = json.load(open(ref))
        eq = [a == b for a, b in zip(out["texts"], refs)]
        out["identity"] = f"{sum(eq)}/{len(eq)}"
        bad = [i for i, e in enumerate(eq) if not e]
        if bad:  # re-check the mismatches without streaming
            ns = [post(port, dict(chat(PROMPTS[i]), max_tokens=256), stream=False)["text"] for i in bad]
            out["identity_nonstream_recheck"] = f"{sum(a == refs[i] for a, i in zip(ns, bad))}/{len(bad)}"
            out["mismatch_idx"] = bad
    return out


def ttft8(port):
    nonce = uuid.uuid4().hex[:8]
    ts = [post(port, dict(chat(f"[{nonce}] {p}"), max_tokens=1))["ttft"] for p in PROMPTS]
    return 1e3 * statistics.median(ts)


def long_req(port, nonce, key, maxtok=16):
    r = post(port, dict(longp(nonce, CHARS[key]), max_tokens=maxtok))
    u = r["usage"] or {}
    pt, ptime = u.get("prompt_tokens"), u.get("total_prompt_time_sec")
    return {"prompt_tokens": pt, "prompt_time_s": ptime, "prompt_tok_s": pt / ptime if pt and ptime else None,
            "ttft_s": r["ttft"], "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "wall_s": r["wall"], "decode_tok_s": r["decode_tok_s"]}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    plan, port, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    res = {"plan": plan}
    t0 = time.time()
    if plan in ("service", "mtpoff"):
        warm(port)
        for rep in (1, 2):
            e = eight(port, f"{B}/ref/h-off.first8.json")
            res[f"eight_{rep}"] = e
            log(f"8x256 rep{rep}: {e['decode_tok_s']:.1f} tok/s (server median {e['decode_tok_s_server_median']:.1f}), "
                f"ttft {e['ttft_ms_median']:.0f} ms, identity vs h-off {e['identity']}"
                + (f" (non-stream recheck {e['identity_nonstream_recheck']})" if "identity_nonstream_recheck" in e else ""))
        res["repeat_identical"] = res["eight_1"]["texts"] == res["eight_2"]["texts"]
        res["ttft_cold_ms"] = [ttft8(port), ttft8(port)]
        log(f"cold TTFT (8 prompts, median): {res['ttft_cold_ms']}")
    if plan == "service":
        for key in ("4k", "13k"):
            res[f"cold_{key}"] = []
            for rep in (1, 2):
                nonce = uuid.uuid4().hex[:12]
                r = long_req(port, nonce, key)
                r["nonce"] = nonce
                res[f"cold_{key}"].append(r)
                log(f"cold {key} rep{rep}: {r['prompt_tokens']} tok in {r['prompt_time_s']:.2f}s = {r['prompt_tok_s']:.0f} tok/s, "
                    f"ttft {r['ttft_s']:.2f}s")
        nonce = res["cold_13k"][-1]["nonce"]
        res["warm_13k"] = []
        for rep in (1, 2):
            r = long_req(port, nonce, "13k")
            res["warm_13k"].append(r)
            log(f"warm 13k rep{rep}: ttft {r['ttft_s']:.3f}s, prompt time {r['prompt_time_s']:.3f}s, cached {r['cached_tokens']}")
    if plan == "ident":
        e = eight(port, f"{B}/ref/q35-prof.first8.json")
        res["eight_1"] = e
        log(f"8x256 old file, MTP off: identity vs q35-prof {e['identity']}"
            + (f" (non-stream recheck {e['identity_nonstream_recheck']})" if "identity_nonstream_recheck" in e else ""))
    if plan == "nsys":
        session, nout = sys.argv[4], sys.argv[5]
        warm(port)
        subprocess.run([NSYS, "start", f"--session={session}", "--sample=none", "--cpuctxsw=none", "-o", nout, "-f", "true"],
                       check=True, timeout=120)
        res["reqs"] = []
        for rep in (1, 2):
            time.sleep(0.5)
            r = long_req(port, uuid.uuid4().hex[:12], "2k", maxtok=32)
            res["reqs"].append(r)
            log(f"nsys rep{rep}: prompt {r['prompt_tokens']} tok, decode {r['decode_tok_s'] or 0:.1f} tok/s (under nsys)")
        time.sleep(0.5)
        subprocess.run([NSYS, "stop", f"--session={session}"], timeout=600, stdout=subprocess.DEVNULL)
    res["client_s"] = time.time() - t0
    json.dump(res, open(outp, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
