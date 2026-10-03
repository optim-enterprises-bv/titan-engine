#!/usr/bin/env python3
"""ptxcmp.py OLD.ptx NEW.ptx: per-entry comparison of two PTX modules (comments stripped; basic-block label
numbers, static shared-array numbers and Rust symbol names normalised).
Every .entry present in OLD must be byte-identical in NEW; entries only in NEW are listed."""
import re, sys


def norm(s):
    # basic-block label numbers are module-global and Rust v0 symbol names carry crate hashes /
    # back-reference indices: neither is code
    s = re.sub(r"\$L__BB\d+_", "$L__BB_", s)
    s = re.sub(r"__shared_mem_\d+", "__shared_mem_N", s)
    return re.sub(r"__dynamic_smem_\w+|\b_R[0-9A-Za-z_]+", "SYM", s)


def entries(path):
    s = norm(re.sub(r"//[^\n]*", "", open(path).read().replace("\r", "")))
    lines = s.split("\n")
    starts = [i for i, l in enumerate(lines) if re.match(r"^\.(visible|entry|func|extern|global|shared|weak|const)\b", l)]
    out = {}
    for n, st in enumerate(starts):
        e = starts[n + 1] if n + 1 < len(starts) else len(lines)
        m = re.match(r"^(?:\.visible\s+)?\.entry\s+(\w+)\(", lines[st])
        if m:
            out[m.group(1)] = "\n".join(x.rstrip() for x in lines[st:e]).strip()
    return out


a, b = entries(sys.argv[1]), entries(sys.argv[2])
same = [k for k in a if k in b and a[k] == b[k]]
diff = [k for k in a if k in b and a[k] != b[k]]
gone = [k for k in a if k not in b]
new = [k for k in b if k not in a]
print(f"PTXCMP {sys.argv[2].split('/')[-1]}: {len(same)}/{len(a)} existing entries identical, {len(diff)} differ, {len(gone)} missing, {len(new)} new")
for k in diff[:10]:
    print("  DIFFERS", k)
for k in gone[:10]:
    print("  MISSING", k)
print("  new:", " ".join(new[:8]), "..." if len(new) > 8 else "")
sys.exit(0 if not diff and not gone else 1)
