#!/usr/bin/env python3
"""Prefix-cache gates for titan-engine (qwen35moe hybrid, unpaged): opencode-shaped multi-turn chats.

usage:
  pc_test.py warm  PORT LOG OUT            turn 1, turn 2 (turn 3) of each case on one server; records
                                            the resume point the server log reports for every request
  pc_test.py cold  PORT LOG IN OUT [CASE..] the later turns of IN (exactly the same messages) on a
                                            fresh server (run it with TITAN_PREFIX_CACHE=0)
  pc_test.py short PORT LOG OUT [IN]        20 short tool-loop chats (prompts-eval.txt); with IN, only
                                            their turn 2 (cold reference)
  pc_test.py big   PORT LOG OUT SYSFILE N   N back-to-back chats with the system text of SYSFILE
  pc_test.py cmp   A B                      turn-by-turn greedy identity, first divergence, logprob diffs
"""
import json, os, re, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MAXTOK = 128


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    return r, time.time() - t


def log_size(log):
    try:
        return os.path.getsize(log)
    except OSError:
        return 0


def resumes_since(log, pos):
    try:
        with open(log, "rb") as f:
            f.seek(pos)
            text = f.read().decode(errors="replace")
    except OSError:
        return []
    return [(int(a), int(b)) for a, b in re.findall(r"Hybrid prefix cache hit: resume at (\d+) of (\d+)", text)]


def ask(port, log, messages, thinking=True, max_tokens=MAXTOK):
    body = {"model": "default", "messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
            "seed": 0, "logprobs": True}
    if not thinking:
        body["enable_thinking"] = False
    pos = log_size(log)
    r, wall = post(port, body)
    time.sleep(0.3)  # the log line of this request
    msg = r["choices"][0]["message"]
    lp = (r["choices"][0].get("logprobs") or {}).get("content") or []
    u = r.get("usage", {})
    rec = {
        "messages": messages, "thinking": thinking, "max_tokens": max_tokens,
        "content": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or "",
        "tokens": [x.get("token") for x in lp], "logprobs": [x.get("logprob") for x in lp],
        "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
        "prompt_time": u.get("total_prompt_time_sec"), "wall": wall,
        "resume": resumes_since(log, pos),
    }
    print(f"  prompt {rec['prompt_tokens']} tok, prompt time {rec['prompt_time']:.2f}s, wall {wall:.2f}s, "
          f"resume {rec['resume']}, out {len(rec['tokens'])} tok: {(rec['reasoning'] + '|' + rec['content'])[:70]!r}",
          flush=True)
    return rec


def as_history(rec):
    """The assistant message a client sends back: the reply content (the template drops the reasoning
    of turns before the last user query)."""
    return {"role": "assistant", "content": rec["content"]}


def as_tool_history(rec):
    """In a tool loop the template keeps the reasoning of the assistant turns after the last user query."""
    c = rec["content"]
    if rec["reasoning"]:
        c = "<think>\n" + rec["reasoning"] + "\n</think>\n\n" + c
    return {"role": "assistant", "content": c}


def cases():
    sys13 = open(os.path.join(HERE, "sys13k.txt")).read()
    return [
        # opencode, thinking off: turn 2 re-renders turn 1's reply without the generation prompt's
        # empty think block -> the longest resume point is the last chunk boundary (the exact grid)
        ("nothink", False, sys13, "Summarise what the code in the system text does in three sentences.",
         ["Now name the three most important functions and say why.", "Which of them would you test first?"],
         "plain"),
        # thinking on, a new user message: the reply's reasoning is dropped from the history
        ("think", True, sys13, "What does search_for_matching_cache return when nothing matches? Be brief.",
         ["And when does it skip a cached entry?"], "plain"),
        # thinking on, a tool loop: turn 2 = turn 1 + reply (with its reasoning) + a tool response;
        # the prompt's end (mid-chunk) or the whole sequence is a prefix
        ("tool", True, sys13, "Find where the KV cache length is changed. Say which file you would open first.",
         ["<tool_response>\nkv_cache/single_cache.rs: set_len, reset, append\n</tool_response>"], "tool"),
        # a tool loop whose assistant turn the client rewrote: only the prompt's end is shared, so the
        # resume point falls mid-chunk (the reference is a cold pass split at the same points)
        # opencode: a tool step, then a new user message (the live check that found the dropped
        # chunk-boundary point): turn 3 re-renders turn 1's reply without its reasoning
        ("loop", True, sys13, "Live check: which function restores recurrent state? Be brief.",
         ["<tool_response>\nkv_cache/hybrid_cache.rs: restore_recurrent_state\n</tool_response>",
          "Thanks. One more line on why it exists?"], "loop"),
        ("ptool", True, sys13, "Which function evicts cached entries? Answer after checking the code.",
         ["<tool_response>\nprefix_cacher.rs: evict_caches, evict_all_caches\n</tool_response>"], "fixed"),
    ]


def run_warm(port, log, out):
    res = []
    for name, thinking, system, u1, later, style in cases():
        print(f"case {name}", flush=True)
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": u1}]
        turns = [ask(port, log, msgs, thinking)]
        for u in later:
            prev = turns[-1]
            if style == "loop":
                hist = as_tool_history(prev) if len(turns) == 1 else as_history(prev)
            elif style == "fixed":
                hist = {"role": "assistant", "content": "<think>\nI should check the code first.\n</think>\n\nLet me search the files."}
            else:
                hist = as_tool_history(prev) if style == "tool" else as_history(prev)
            msgs = msgs + [hist, {"role": "user", "content": u}]
            turns.append(ask(port, log, msgs, thinking))
        res.append({"case": name, "turns": turns})
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)


def run_cold(port, log, inp, out, only):
    res = []
    for c in json.load(open(inp)):
        if only and c["case"] not in only:
            continue
        print(f"case {c['case']} (cold)", flush=True)
        turns = [None] + [ask(port, log, t["messages"], t["thinking"], t["max_tokens"]) for t in c["turns"][1:]]
        res.append({"case": c["case"], "turns": turns})
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)


def run_short(port, log, out, inp):
    prompts = [l.strip() for l in open(os.path.join(HERE, "..", "prompts-eval.txt")) if l.strip()][:20]
    res = []
    if inp:
        for c in json.load(open(inp)):
            t2 = c["turns"][1]
            res.append({"case": c["case"], "turns": [None, ask(port, log, t2["messages"], True, t2["max_tokens"])]})
    else:
        for i, p in enumerate(prompts):
            msgs = [{"role": "user", "content": p}]
            t1 = ask(port, log, msgs, True, 64)
            msgs = msgs + [as_tool_history(t1), {"role": "user", "content": "<tool_response>\nok\n</tool_response>"}]
            res.append({"case": f"short{i}", "turns": [t1, ask(port, log, msgs, True, 64)]})
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)


def run_big(port, log, out, sysfile, n):
    system = open(sysfile).read()
    res = []
    for k in range(n):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": f"Question {k}: summarise the system text in one line."}]
        res.append(ask(port, log, msgs, False, 32))
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)


def cmp(a, b):
    A = {c["case"]: c for c in json.load(open(a))}
    B = {c["case"]: c for c in json.load(open(b))}
    same = total = 0
    for name, cb in B.items():
        ca = A.get(name)
        if not ca:
            continue
        for i, tb in enumerate(cb["turns"]):
            if tb is None or i >= len(ca["turns"]) or ca["turns"][i] is None:
                continue
            ta = ca["turns"][i]
            total += 1
            eq = ta["tokens"] == tb["tokens"]
            same += eq
            n = next((j for j, (x, y) in enumerate(zip(ta["tokens"], tb["tokens"])) if x != y),
                     min(len(ta["tokens"]), len(tb["tokens"])))
            d = [abs(x - y) for x, y in zip(ta["logprobs"][:n], tb["logprobs"][:n]) if x is not None and y is not None]
            print(f"{name} turn {i + 1}: {'SAME' if eq else 'DIFF'} ({len(ta['tokens'])}/{len(tb['tokens'])} tok, "
                  f"first diff at {n if not eq else '-'}), resume A {ta.get('resume')} B {tb.get('resume')}, "
                  f"max |dlogprob| before it {max(d) if d else 0:.3g}, prompt time A {ta['prompt_time']:.2f}s B {tb['prompt_time']:.2f}s")
    print(f"GATE identity {os.path.basename(b)} vs {os.path.basename(a)}: {same} / {total}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "warm":
        run_warm(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "cold":
        run_cold(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6:])
    elif cmd == "short":
        run_short(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else None)
    elif cmd == "big":
        run_big(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]))
    elif cmd == "cmp":
        cmp(sys.argv[2], sys.argv[3])
