#!/usr/bin/env python3
"""iq3s_grid[512] (ggml-common.h) -> src/iq3s_grid.rs (this kernel crate's global-memory table)
and, with an argument, <dir>/data_iq3s_grid.rs (candle's CPU dequantize table)."""
import re, sys, os
src = open("~/ai/llama.cpp/ggml/src/ggml-common.h").read()
m = re.search(r"GGML_TABLE_BEGIN\([^,]+,\s*iq3s_grid,\s*(\d+)\)(.*?)GGML_TABLE_END\(\)", src, re.S)
n = int(m.group(1)); body = re.sub(r"//[^\n]*", "", m.group(2))
vals = [int(t, 0) for t in re.findall(r"0x[0-9a-fA-F]+|\d+", body)]
assert n == 512 and len(vals) == 512 and all(0 <= v < 2**32 for v in vals)
text = "[\n" + "\n".join("    " + ", ".join("0x%08x" % v for v in vals[i:i + 8]) + "," for i in range(0, 512, 8)) + "\n]\n"
here = os.path.dirname(os.path.abspath(__file__))
open(os.path.join(here, "src/iq3s_grid.rs"), "w").write(text)
if len(sys.argv) > 1:
    open(os.path.join(sys.argv[1], "data_iq3s_grid.rs"), "w").write(text)
print("iq3s_grid: 512 u32 values")
