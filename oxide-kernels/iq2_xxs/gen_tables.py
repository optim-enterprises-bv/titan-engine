#!/usr/bin/env python3
"""Regenerate the iq2_xxs table chains inside #[cuda_module].

ggml-common.h writes table values as hex AND decimal (kmask_iq2xs / ksigns_iq2xs are decimal),
so parse tokens generically instead of regexing only 0x... .
"""
import re, os

SRC = "~/ai/llama.cpp/ggml/src/ggml-common.h"
MAIN = "~/titan-engine/oxide-kernels/iq2_xxs/src/main.rs"

src = open(SRC).read()


def table(name):
    m = re.search(r"GGML_TABLE_BEGIN\([^,]+,\s*" + name + r",\s*(\d+)\)(.*?)GGML_TABLE_END\(\)", src, re.S)
    n = int(m.group(1))
    body = re.sub(r"//[^\n]*", "", m.group(2))
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    vals = [int(t, 0) for t in re.findall(r"(?:0x[0-9a-fA-F]+|\d+)", body)]
    assert len(vals) == n, (name, n, len(vals))
    return vals


G = table("iq2xxs_grid")
K = table("ksigns_iq2xs")
M = table("kmask_iq2xs")
assert len(G) == 256 and len(K) == 128 and len(M) == 8, (len(G), len(K), len(M))
assert all(b <= 0x2b for v in G for b in v.to_bytes(8, "little")), "grid byte out of range"
print("iq2xxs_grid %d, ksigns %d, kmask %d" % (len(G), len(K), len(M)))

lo = [v & 0xFFFFFFFF for v in G]
hi = [(v >> 32) & 0xFFFFFFFF for v in G]


def chain(name, vals, ty, fmt, comment):
    L = ["    /// %s" % comment,
         "    #[inline(always)]",
         "    pub fn %s(i: u32) -> %s {" % (name, ty),
         "        let mut v: %s = 0;" % ty]
    for k, val in enumerate(vals):
        if val:
            L.append("        if i == %du32 { v = %s; }" % (k, fmt % val))
    L += ["        v", "    }"]
    return "\n".join(L)


blocks = [
    chain("grid_lo", lo, "u32", "0x%08xu32",
          "`iq2xxs_grid[i]` low word, as a compare chain: a runtime-indexed Rust array spills to\n    /// local memory and a host-side table never reaches the PTX (PORTING.md #43)."),
    chain("grid_hi", hi, "u32", "0x%08xu32", "`iq2xxs_grid[i]` high word."),
    chain("ksign", K, "u8", "0x%02xu8", "`ksigns_iq2xs[128]` (dequantize only; mmvq uses `unpack_ksigns`)."),
    "    /// `kmask_iq2xs[8]`, indexed only by unrolled constants so it stays in registers.\n"
    "    pub const KMASK: [u8; 8] = [" + ", ".join("0x%02x" % m for m in M) + "];",
]
new_block = "\n".join(blocks)

s = open(MAIN).read()
start = s.index("    /// `iq2xxs_grid[i]` low word")
end = s.index("    pub const KMASK: [u8; 8] = ")
end = s.index("\n", end) + 1
s = s[:start] + new_block + "\n" + s[end:]
open(MAIN, "w").write(s)
print("chains rewritten; main.rs", os.path.getsize(MAIN), "bytes")
