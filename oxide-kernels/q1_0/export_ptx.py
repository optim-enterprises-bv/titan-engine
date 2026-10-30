#!/usr/bin/env python3
"""export_ptx.py [mistral.rs dir]: split q1_0.ptx into the modules titan embeds:
mistralrs-quant/src/gguf/q1_0_{mmvq,mmq}_oxide.ptx, decode (q8_1 quantizers + mmvq) and prefill
(mmq quantizers + mul_mat_q + stream-k fixup), so the decode module JIT-compiles in a fraction of
the time; and candle's candle-core/src/quantized/q1_0_oxide.ptx (dequantize + mmvq). Every non-entry
item (.func bodies, extern shared declarations) is kept in each. Gate mutants (`_mut`) are not exported."""
import re, subprocess, sys, os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "q1_0.ptx")).read()
root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/src/titan-engine/mistral.rs")
dst_dir = os.path.join(root, "mistralrs-quant/src/gguf")
candle_dir = os.path.join(os.path.dirname(os.path.abspath(root)), "candle/candle-core/src/quantized")
lines = src.split("\n")
# top-level item starts: a line at column 0 beginning with '.' (directive) or '//' comment before it
starts = [i for i, l in enumerate(lines) if re.match(r"^\.(visible|entry|func|extern|global|shared|weak|const)\b", l)]
items = []  # (start, end, entry_name or None)
for n, s in enumerate(starts):
    e = starts[n + 1] if n + 1 < len(starts) else len(lines)
    m = re.match(r"^(?:\.visible\s+)?\.entry\s+(\w+)\(", lines[s])
    items.append((s, e, m.group(1) if m else None))
head = "\n".join(lines[: starts[0]])
def emit(keep, name, where=None):
    out = [head]
    n = 0
    for s, e, ent in items:
        if ent is None or keep(ent):
            out.append("\n".join(lines[s:e]))
            n += ent is not None
    path = os.path.join(where or dst_dir, name)
    open(path, "w").write("\n".join(out) + "\n")
    r = subprocess.run(["/usr/local/cuda/bin/ptxas", "-arch=sm_120a", "-O3", path, "-o", "/dev/null"], capture_output=True, text=True)
    print(f"{name}: {n} entries, {os.path.getsize(path)} bytes, ptxas rc={r.returncode} {r.stderr.strip()[:200]}")
    return r.returncode == 0
ok = emit(lambda e: e.startswith("q1_0_quantize_q8_1_") or e.startswith("q1_0_mmvq_"), "q1_0_mmvq_oxide.ptx")
ok &= emit(lambda e: e.startswith("q1_0_quantize_mmq_") or e.startswith("q1_0_mmq_"), "q1_0_mmq_oxide.ptx")
if os.path.isdir(candle_dir):
    # the entries candle's llama_fmt path launches: dequantize and mmvq_1..8 (+ small_k)
    ok &= emit(lambda e: re.fullmatch(r"q1_0_dequant_(f32|f16|bf16)|q1_0_mmvq_[1-8]|q1_0_mmvq_1_small_k", e) is not None,
               "q1_0_oxide.ptx", candle_dir)
sys.exit(0 if ok else 1)
