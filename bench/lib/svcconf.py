#!/usr/bin/env python3
"""Print the service config of deploy/titan-mistral.service for bench servers, so the bench follows the unit.

    svcconf.py env            -> KEY=VALUE lines (Environment=, %h expanded)
    svcconf.py args PORT      -> ExecStart arguments after the binary, one per line, with `-p PORT`
    svcconf.py model          -> model dir and file (-m / -f), one per line"""
import os, shlex, sys

unit = os.path.expanduser("~/titan-engine/deploy/titan-mistral.service")
env, args = [], []
for line in open(unit):
    line = line.strip().replace("%h", os.path.expanduser("~"))
    if line.startswith("Environment="):
        env.append(line[len("Environment="):])
    elif line.startswith("ExecStart="):
        args = shlex.split(line[len("ExecStart="):])[1:]
# BENCH_EXTRA_ENV="K=V K2=V2": extra settings on top of the unit (A/B of a flag without a new unit)
env += os.environ.get("BENCH_EXTRA_ENV", "").split()
cmd = sys.argv[1]
if cmd == "env":
    print("\n".join(env))
elif cmd == "args":
    out, i = [], 0
    while i < len(args):
        if args[i] in ("-p", "--port"):
            out += ["-p", sys.argv[2]]
            i += 2
            continue
        out.append(args[i])
        i += 1
    print("\n".join(out))
elif cmd == "model":
    print(args[args.index("-m") + 1])
    print(args[args.index("-f") + 1])
