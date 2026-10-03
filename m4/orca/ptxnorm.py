#!/usr/bin/env python3
"""ptxnorm.py A.ptx B.ptx: per-entry comparison after renaming the compiler's per-function numbering
(__local_depotN, $L__BBN_M, __shared_mem_N, $L__tmpN) to first-use order and dropping // comments and spacing.
Prints identical / differing / missing / new entry counts."""
import re, sys
def entries(p):
    s = open(p).read().replace("\r", "")
    s = re.sub(r"//[^\n]*", "", s)
    parts = re.split(r"\n(?=(?:\.visible\s+)?\.entry\s)", s)
    out = {}
    for part in parts[1:]:
        m = re.match(r"(?:\.visible\s+)?\.entry\s+(\w+)", part); name = m.group(1)
        body = part.split("\n}\n")[0]
        maps, cnt = {}, {}
        def ren(mo):
            k = mo.group(0); kind = mo.group(1)
            if k not in maps:
                cnt[kind] = cnt.get(kind, 0) + 1; maps[k] = f"{kind}#{cnt[kind]}"
            return maps[k]
        body = re.sub(r"(__local_depot|\$L__BB|__shared_mem_|\$L__tmp)[0-9_]+", ren, body)
        body = re.sub(r"Cs[0-9A-Za-z]{8,}_", "Cs#_", body)  # rustc crate disambiguator (build path)
        out[name] = "\n".join(l.strip() for l in re.sub(r"[ \t]+", " ", body).split("\n")).strip()
    return out
a, b = entries(sys.argv[1]), entries(sys.argv[2])
same = [k for k in a if k in b and a[k] == b[k]]; diff = [k for k in a if k in b and a[k] != b[k]]
print(f"PTXNORM {sys.argv[2].split('/')[-1]}: {len(same)}/{len(a)} entries identical, {len(diff)} differ, {len([k for k in a if k not in b])} missing, {len([k for k in b if k not in a])} new {diff[:4]}")
