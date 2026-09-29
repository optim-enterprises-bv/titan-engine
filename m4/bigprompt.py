#!/usr/bin/env python3
"""Send one prompt of ~N tokens (chat) and report prompt tokens / success. usage: bigprompt.py PORT N"""
import json, sys, urllib.request
port, n = sys.argv[1], int(sys.argv[2])
chunk = "The titan engine tiers mixture-of-experts weights between the GPU and the CPU. "
text = chunk * (n // 16)  # ~16 tokens per chunk
body = {"model": "default", "messages": [{"role": "system", "content": text}, {"role": "user", "content": "Summarise the system text in one line."}], "max_tokens": 32, "temperature": 0}
try:
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
    print(f"ok prompt_tokens={r['usage']['prompt_tokens']} prompt_tok_s={r['usage'].get('avg_prompt_tok_per_sec')}")
except Exception as e:
    print("FAILED", e)
