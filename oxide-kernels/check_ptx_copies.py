#!/usr/bin/env python3
"""Check that every PTX titan ships (include_str!'d under candle/ and mistral.rs/) is the build of a gated
oxide-kernels crate.

A shipped copy's crate is the crate PTX that defines all of the copy's kernel entries. The two are compared
as the multiset of their top-level blocks (functions, entries, globals) with comments dropped and branch labels
renumbered: the compiler does not keep function order stable between builds (and labels carry the function's
position), so a byte compare would flag equivalent builds.

    python3 oxide-kernels/check_ptx_copies.py        # from the titan-engine root; exit 1 on any mismatch
"""
import collections
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRATES = os.path.join(ROOT, "oxide-kernels")
ENTRY = re.compile(r"^\.visible \.entry ([A-Za-z0-9_$]+)", re.M)
BLOCK_START = re.compile(r"^(\.visible |\.extern |\.weak )?(\.entry|\.func|\.global|\.const|\.shared)\b")
# branch labels carry the function's position in the file (`$L__BB<fn>_<n>`), which follows the unstable order
LABEL = re.compile(r"\$L__BB\d+_")
# shared-memory symbols are numbered in emission order too
SHARED = re.compile(r"__shared_mem_\d+")


def blocks(path):
    """Top-level PTX blocks, comments stripped, as a Counter of their text."""
    out, cur, depth = collections.Counter(), [], 0
    for line in open(path, encoding="utf-8", errors="replace"):
        line = SHARED.sub("__shared_mem_", LABEL.sub("$L__BB_", line.split("//", 1)[0].rstrip()))
        if not line:
            continue
        if depth == 0 and not cur and not BLOCK_START.match(line):
            continue  # header (.version/.target/.address_size) and stray directives
        cur.append(line)
        depth += line.count("{") - line.count("}")
        if depth == 0 and (line.endswith("}") or line.endswith(";")):
            out["\n".join(cur)] += 1
            cur = []
    return out


def entries(path):
    return set(ENTRY.findall(open(path, encoding="utf-8", errors="replace").read()))


def shipped():
    """(copy path, referring source) for each include_str!'d .ptx under candle/ and mistral.rs/."""
    inc = re.compile(r'include_str!\("([^"]+\.ptx)"\)')
    found = {}
    for top in ("candle", "mistral.rs"):
        for d, dirs, files in os.walk(os.path.join(ROOT, top)):
            dirs[:] = [x for x in dirs if x != "target"]
            for f in files:
                if f.endswith(".rs"):
                    src = os.path.join(d, f)
                    for rel in inc.findall(open(src, encoding="utf-8", errors="replace").read()):
                        found.setdefault(os.path.normpath(os.path.join(d, rel)), src)
    return sorted(found.items())


def main():
    crates = {}
    for name in sorted(os.listdir(CRATES)):
        d = os.path.join(CRATES, name)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(".ptx"):
                    p = os.path.join(d, f)
                    crates[p] = entries(p)
    bad = 0
    for copy, src in shipped():
        rel = os.path.relpath(copy, ROOT)
        if not os.path.exists(copy):
            print(f"MISSING   {rel} (included by {os.path.relpath(src, ROOT)})")
            bad += 1
            continue
        want = entries(copy)
        owners = [p for p, e in crates.items() if want and want <= e]
        if not owners:
            print(f"NO CRATE  {rel}: no oxide-kernels crate defines its {len(want)} entries")
            bad += 1
            continue
        mine = blocks(copy)
        best = None
        for p in owners:
            theirs = blocks(p)
            only_copy = sum((mine - theirs).values())
            only_crate = sum((theirs - mine).values())
            if best is None or only_copy + only_crate < best[1] + best[2]:
                best = (p, only_copy, only_crate)
        p, only_copy, only_crate = best
        crate = os.path.relpath(p, ROOT)
        if only_copy == 0 and only_crate == 0:
            print(f"MATCH     {rel} == {crate}")
        elif only_copy == 0:
            # the crate builds more (e.g. gate-only entries) but every shipped block is its build
            print(f"MATCH     {rel} == {crate} (subset; crate has {only_crate} more blocks)")
        else:
            print(f"DIFFERS   {rel} vs {crate}: {only_copy} blocks only in the copy, {only_crate} only in the crate")
            bad += 1
    print(f"{bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
