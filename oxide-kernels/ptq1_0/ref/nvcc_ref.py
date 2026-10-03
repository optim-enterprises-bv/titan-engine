#!/usr/bin/env python3
"""Rebuild the reference cubins + PTX of the sudoingX/llama.cpp bonsai2 v1.1 (ff41412) TUs the ptq1_0 crate
gates against, from the reference build's compile_commands.json (nvcc 13.3, g++-15, sm_120a, -use_fast_math),
retargeted from -c to --cubin / --ptx. Usage: nvcc_ref.py [tu ...] (default: all). Run under a 4G memory cap."""
import json, os, shlex, subprocess, sys
R = os.path.expanduser("~/titan-engine/ref/sudoingx-bonsai2")
OUT = os.path.dirname(os.path.abspath(__file__))
TUS = {
    "mmvq": "ggml/src/ggml-cuda/mmvq.cu",
    "quantize": "ggml/src/ggml-cuda/quantize.cu",
    "fwht": "ggml/src/ggml-cuda/fwht.cu",
    "mmq": "ggml/src/ggml-cuda/template-instances/mmq-instance-ptq1_0.cu",
    "convert": "ggml/src/ggml-cuda/convert.cu",
    "unary": "ggml/src/ggml-cuda/unary.cu",
}
cc = {e["file"]: e for e in json.load(open(f"{R}/build/compile_commands.json"))}
for name in (sys.argv[1:] or TUS):
    e = cc[f"{R}/{TUS[name]}"]
    args = shlex.split(e["command"])
    i = args.index("-c"); args = args[:i] + args[i + 2:]          # drop "-c file"
    j = args.index("-o"); args = args[:j] + args[j + 2:]          # drop "-o obj"
    args = [a for a in args if not a.startswith("--generate-code") and a != "-compress-mode=size"]
    for kind in ("cubin", "ptx"):
        cmd = args + ["-arch=sm_120a", f"--{kind}", f"{R}/{TUS[name]}", "-o", f"{OUT}/{name}.{kind}"]
        r = subprocess.run(cmd, cwd=e["directory"], capture_output=True, text=True)
        print(name, kind, "rc", r.returncode, r.stderr.strip()[-400:], flush=True)
