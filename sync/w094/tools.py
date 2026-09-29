#!/usr/bin/env python3
"""opencode-shaped tool round trip + Anthropic Messages smoke. usage: tools.py PORT"""
import json, sys, time, urllib.request
port = int(sys.argv[1])
def post(path, body):
    t = time.time()
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600)
    return json.load(r), time.time() - t
tools = [{"type": "function", "function": {"name": "read", "description": "Read a file from the workspace",
          "parameters": {"type": "object", "properties": {"filePath": {"type": "string", "description": "absolute path"}}, "required": ["filePath"]}}},
         {"type": "function", "function": {"name": "bash", "description": "Run a shell command",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
msgs = [{"role": "system", "content": "You are opencode, a coding agent. Use the tools to answer. Working directory: /work"},
        {"role": "user", "content": "Read /work/src/main.rs and tell me what it prints."}]
ok = True
for temp in (0.0, 0.6):
    r, dt = post("/v1/chat/completions", {"model": "default", "messages": msgs, "tools": tools, "tool_choice": "auto", "temperature": temp, "max_tokens": 1024})
    m = r["choices"][0]["message"]; calls = m.get("tool_calls") or []
    print(f"temp {temp}: finish={r['choices'][0]['finish_reason']} {dt:.1f}s tool_calls={[(c['function']['name'], c['function']['arguments']) for c in calls]} content={(m.get('content') or '')[:80]!r}")
    try:
        args = json.loads(calls[0]["function"]["arguments"]); good = calls[0]["function"]["name"] == "read" and args.get("filePath", "").endswith("main.rs")
    except Exception as e:
        good = False; print("  bad tool call:", e)
    ok &= good
    if temp == 0.0 and calls:
        c = calls[0]
        msgs2 = msgs + [{"role": "assistant", "content": m.get("content") or "", "tool_calls": calls},
                        {"role": "tool", "tool_call_id": c.get("id", "call_0"), "content": 'fn main() {\n    println!("hello from titan");\n}\n'}]
        r2, dt2 = post("/v1/chat/completions", {"model": "default", "messages": msgs2, "tools": tools, "temperature": 0.0, "max_tokens": 512})
        m2 = r2["choices"][0]["message"]
        ans = m2.get("content") or ""
        print(f"  after tool result: {dt2:.1f}s finish={r2['choices'][0]['finish_reason']} content={ans[:160]!r} usage={r2['usage'].get('prompt_tokens_details')}")
        ok &= "hello from titan" in ans
r, dt = post("/v1/messages", {"model": "default", "max_tokens": 256, "messages": [{"role": "user", "content": "Say hi in three words."}]})
print(f"anthropic /v1/messages: {dt:.1f}s type={r.get('type')} stop={r.get('stop_reason')} content={json.dumps(r.get('content'))[:200]}")
ok &= r.get("type") == "message"
print("TOOLS", "PASS" if ok else "FAIL")
