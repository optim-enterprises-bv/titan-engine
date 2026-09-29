#!/usr/bin/env python3
"""Live check against the service on :int(sys.argv[1]): two opencode-shaped turns (tool loop), then a new user turn."""
import json, os, sys
sys.path.insert(0, os.path.expanduser("~/titan-engine/m4/pc"))
import pc_test as t
sysp = open(os.path.join(t.HERE, "sys13k.txt")).read()
msgs = [{"role": "system", "content": sysp}, {"role": "user", "content": "Live check: which function restores recurrent state? Be brief."}]
r1 = t.ask(int(sys.argv[1]), "/dev/null", msgs, True, 96)
msgs += [t.as_tool_history(r1), {"role": "user", "content": "<tool_response>\nkv_cache/hybrid_cache.rs: restore_recurrent_state\n</tool_response>"}]
r2 = t.ask(int(sys.argv[1]), "/dev/null", msgs, True, 96)
msgs += [t.as_history(r2), {"role": "user", "content": "Thanks. One more line on why it exists?"}]
r3 = t.ask(int(sys.argv[1]), "/dev/null", msgs, True, 96)
json.dump([r1, r2, r3], open(os.path.join(t.HERE, "out", sys.argv[2]), "w"), indent=1)
