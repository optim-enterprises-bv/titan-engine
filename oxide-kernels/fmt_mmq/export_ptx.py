#!/usr/bin/env python3
"""export_ptx.py [mistral.rs dir]: split fmt_mmq.ptx into the per-format prefill modules mistral.rs
embeds (mistralrs-quant/src/gguf/{iq4_nl,mxfp4,nvfp4,iq4_xs}_mmq_oxide.ptx: the format's activation
quantizers, mul_mat_q instances and stream-k fixups), so each JIT-compiles only what its model
uses. Every non-entry item (.func bodies, extern shared declarations) is kept in each module."""
import re, subprocess, sys, os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fmt_mmq.ptx")).read()
root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/titan-engine/mr-mmq")
dst_dir = os.path.join(root, "mistralrs-quant/src/gguf")
lines = src.split("\n")
starts = [i for i, l in enumerate(lines) if re.match(r"^\.(visible|entry|func|extern|global|shared|weak|const)\b", l)]
items = []  # (start, end, entry_name or None)
for n, s in enumerate(starts):
    e = starts[n + 1] if n + 1 < len(starts) else len(lines)
    m = re.match(r"^(?:\.visible\s+)?\.entry\s+(\w+)\(", lines[s])
    items.append((s, e, m.group(1) if m else None))
head = "\n".join(lines[: starts[0]])
ok = True
# IQ4_XS's prefill uses IQ4_NL's activation quantizer (quantize_mmq_q8_1<D4>); the gate mutants
# (`iq4_xs_mut*`) are not exported.
QUANT_FROM = {"iq4_xs": "iq4_nl_quantize_mmq_d4_"}
for fmt in ("iq4_nl", "mxfp4", "nvfp4", "iq4_xs"):
    out, n = [head], 0
    for s, e, ent in items:
        if ent is not None and "_mut" in ent:
            continue
        if ent is None or ent.startswith(fmt + "_") or (fmt in QUANT_FROM and ent.startswith(QUANT_FROM[fmt])):
            out.append("\n".join(lines[s:e]))
            n += ent is not None
    path = os.path.join(dst_dir, f"{fmt}_mmq_oxide.ptx")
    open(path, "w").write("\n".join(out) + "\n")
    r = subprocess.run(["/usr/local/cuda/bin/ptxas", "-arch=sm_120a", "-O3", path, "-o", "/dev/null"], capture_output=True, text=True)
    print(f"{path}: {n} entries, {os.path.getsize(path)} bytes, ptxas rc={r.returncode} {r.stderr.strip()[:200]}")
    ok &= r.returncode == 0 and n == 35
sys.exit(0 if ok else 1)
