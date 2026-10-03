#!/usr/bin/env python3
"""export_ptx.py: split ptq1_0.ptx into the two modules mistral.rs embeds
(mistralrs-quant/src/gguf/q1_0_{mmvq,mmq}_oxide.ptx): decode (q8_1 quantizers + mmvq) and prefill
(mmq quantizers + mul_mat_q + stream-k fixup), so the decode module JIT-compiles in a fraction of
the time. Every non-entry item (.func bodies, extern shared declarations) is kept in both."""
import re, subprocess, sys, os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptq1_0.ptx")).read()
dst_dir = os.path.expanduser("~/titan-engine/mr-b2/mistralrs-quant/src/gguf")
lines = src.split("\n")
# top-level item starts: a line at column 0 beginning with '.' (directive) or '//' comment before it
starts = [i for i, l in enumerate(lines) if re.match(r"^\.(visible|entry|func|extern|global|shared|weak|const)\b", l)]
items = []  # (start, end, entry_name or None)
for n, s in enumerate(starts):
    e = starts[n + 1] if n + 1 < len(starts) else len(lines)
    m = re.match(r"^(?:\.visible\s+)?\.entry\s+(\w+)\(", lines[s])
    items.append((s, e, m.group(1) if m else None))
head = "\n".join(lines[: starts[0]])
def emit(keep, name):
    out = [head]
    n = 0
    for s, e, ent in items:
        if ent is None or keep(ent):
            out.append("\n".join(lines[s:e]))
            n += ent is not None
    path = os.path.join(dst_dir, name)
    open(path, "w").write("\n".join(out) + "\n")
    r = subprocess.run(["/usr/local/cuda/bin/ptxas", "-arch=sm_120a", "-O3", path, "-o", "/dev/null"], capture_output=True, text=True)
    print(f"{name}: {n} entries, {os.path.getsize(path)} bytes, ptxas rc={r.returncode} {r.stderr.strip()[:200]}")
    return r.returncode == 0
ok = emit(lambda e: e.startswith("ptq1_0_quantize_pt_") or e.startswith("ptq1_0_fwht") or e.startswith("ptq1_0_mmvq_"), "ptq1_0_mmvq_oxide.ptx")
ok &= emit(lambda e: e.startswith("ptq1_0_quantize_mmq_") or e.startswith("ptq1_0_mmq_"), "ptq1_0_mmq_oxide.ptx")
sys.exit(0 if ok else 1)
