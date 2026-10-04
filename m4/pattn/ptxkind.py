#!/usr/bin/env python3
"""ptxkind.py BASE.ptx NEW.ptx: ptxnorm.py's per-entry comparison (per-function numbering, comments, spacing and rustc crate
hashes normalised), with identical / differing / missing / new entries counted per kernel family, so a regenerated module
can be checked against what the change is expected to touch."""
import re, sys
from collections import Counter
sys.argv, args = sys.argv[:1], sys.argv[1:]
exec(open(__file__.replace("ptxkind.py", "ptxnorm.py")).read().split("a, b = entries")[0])
a, b = entries(args[0]), entries(args[1])
def fam(k):
    if "BatchDecodeWithPagedKVCacheKernelMLA" in k: return "decode-MLA"
    if "BatchDecodeWithPagedKVCacheKernel" in k:
        g = re.match(r".*ModeE0ELj(\d+)ELj\d+ELj\d+ELj\d+ELj(\d+)ELj\d+E", k)
        return f"decode-stages{g.group(1)}-group{g.group(2)}" if g else "decode"
    if "MergeStates" in k: return "merge"
    m = re.match(r"_ZN20mistralrs_flashinfer\d+(\w+?)_kernel", k)
    if m: return m.group(1)
    return re.sub(r"_(f32|f16|bf16|u8|fp8|e|a|\d+).*$", "", k)
for name, ks in [("identical", [k for k in a if k in b and a[k] == b[k]]), ("DIFFER", [k for k in a if k in b and a[k] != b[k]]),
                 ("missing", [k for k in a if k not in b]), ("new", [k for k in b if k not in a])]:
    c = Counter(fam(k) for k in ks)
    print(f"PTXKIND {args[1].split('/')[-1]} {name} {len(ks)}: " + ", ".join(f"{f} {n}" for f, n in sorted(c.items())))
