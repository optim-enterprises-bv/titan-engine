#!/usr/bin/env python3
"""Measurement client for the release campaign. Engine-neutral: it speaks the OpenAI chat API that both
mistral.rs and llama-server expose, streams every timed request, and uses one stopwatch for both engines.

Definitions (identical for both engines):
  decode tok/s   sum(completion_tokens - 1) / sum(t_last_chunk - t_first_chunk); prefill excluded.
  TTFT (short)   send -> end of a max_tokens=1 response, with a fresh nonce so nothing is prefix-cached.
  TTFT (long)    send -> first streamed chunk that carries text (content or reasoning).
  prompt tok/s   prompt_tokens / TTFT (long). The server's own prompt timing is recorded next to it.
Token counts come from the server's usage block (both engines send it with stream_options.include_usage);
if a server omits it the chunk count is used and the source is recorded."""
import json, statistics, time, urllib.request, uuid

BENCH = __import__("os").path.expanduser("~/titan-engine/bench")
PROMPTS8 = [l.strip() for l in open(f"{BENCH}/prompts/eval8.txt") if l.strip()][:8]
CORPUS = open(f"{BENCH}/prompts/corpus-sys28k.txt").read()
# characters of corpus per nominal length; ~4 chars/token on the Qwen tokenizer (bench: 50800 chars = 12.7k tokens).
CHARS = {"4k": 15600, "13k": 50800, "28k": len(CORPUS)}


def post(port, body, timeout=3600):
    body = dict(body, model="default", temperature=0.0, top_k=1, seed=0, stream=True,
                stream_options={"include_usage": True})
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    r = urllib.request.urlopen(req, timeout=timeout)
    t_first = t_last = None
    usage, timings, nchunks, text = None, None, 0, []
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
        if d.get("timings"):  # llama-server extension
            timings = d["timings"]
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
    u = usage or {}
    ntok, src = u.get("completion_tokens"), "usage"
    if not ntok and timings and timings.get("predicted_n"):
        ntok, src = timings["predicted_n"], "timings"
    if not ntok:
        ntok, src = nchunks, "chunks"
    ptok = u.get("prompt_tokens") or (timings or {}).get("prompt_n")
    srv_prompt_s = u.get("total_prompt_time_sec")
    if srv_prompt_s is None and timings and timings.get("prompt_ms") is not None:
        srv_prompt_s = timings["prompt_ms"] / 1e3
    dec = (ntok - 1) / (t_last - t_first) if t_first and t_last and t_last > t_first and ntok > 1 else None
    return {"text": "".join(text), "ttft": (t_first - t0) if t_first else None, "wall": t_end - t0,
            "decode_span": (t_last - t_first) if t_first and t_last else None, "completion_tokens": ntok,
            "ntok_src": src, "prompt_tokens": ptok, "server_prompt_s": srv_prompt_s, "decode_tok_s": dec,
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens")
                             if isinstance(u.get("prompt_tokens_details"), dict) else None}


def chat(p, max_tokens):
    return {"messages": [{"role": "user", "content": p}], "max_tokens": max_tokens}


def long_body(nonce, key, max_tokens):
    return {"messages": [{"role": "system", "content": f"[bench {nonce}] " + CORPUS[:CHARS[key]]},
                         {"role": "user", "content": "Summarise the system text in one line."}],
            "max_tokens": max_tokens}


def warm(port):
    post(port, chat("Say hi.", 16), timeout=900)


def short_decode(port, nprompts, ntok):
    """One rep: the first nprompts eval prompts x ntok tokens, greedy. Returns the rep's aggregate + per-prompt rows."""
    rows = [post(port, chat(p, ntok)) for p in PROMPTS8[:nprompts]]
    ok = [r for r in rows if r["decode_tok_s"]]
    toks = sum(r["completion_tokens"] - 1 for r in ok)
    span = sum(r["decode_span"] for r in ok)
    return {"decode_tok_s": toks / span if span else None, "completion_tokens": sum(r["completion_tokens"] for r in rows),
            "ntok_src": sorted({r["ntok_src"] for r in rows}), "wall_s": sum(r["wall"] for r in rows),
            "prompt_tokens": [r["prompt_tokens"] for r in rows],
            "texts_sha": __import__("hashlib").sha256("\x00".join(r["text"] for r in rows).encode()).hexdigest()[:16]}


def ttft_short(port, nprompts):
    nonce = uuid.uuid4().hex[:8]
    ts = [post(port, chat(f"[{nonce}] {p}", 1))["wall"] for p in PROMPTS8[:nprompts]]
    return {"ttft_ms": 1e3 * statistics.median(ts), "all_ms": [round(1e3 * t, 1) for t in ts]}


def long_prompt(port, key, dec_tokens, nonce=None):
    nonce = nonce or uuid.uuid4().hex[:12]
    r = post(port, long_body(nonce, key, dec_tokens))
    pt = r["prompt_tokens"]
    return {"nonce": nonce, "prompt_tokens": pt, "ttft_s": r["ttft"],
            "prompt_tok_s": pt / r["ttft"] if pt and r["ttft"] else None,
            "server_prompt_s": r["server_prompt_s"],
            "server_prompt_tok_s": pt / r["server_prompt_s"] if pt and r["server_prompt_s"] else None,
            "decode_tok_s": r["decode_tok_s"], "completion_tokens": r["completion_tokens"], "ntok_src": r["ntok_src"],
            "cached_tokens": r["cached_tokens"], "wall_s": r["wall"]}


if __name__ == "__main__":  # manual smoke: client.py PORT
    import sys
    p = sys.argv[1]
    warm(p)
    print(json.dumps({"short": short_decode(p, 2, 32), "ttft": ttft_short(p, 2), "4k": long_prompt(p, "4k", 16)}, indent=1))
