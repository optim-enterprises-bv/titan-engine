import os

p = "~/titan-engine/candle/candle-core/src/quantized/cuda.rs"
s = open(p).read()

# --- 1. module names actually used by the i-quant crates ---------------------
import re
for f in ("iq2_xxs", "iq3_xxs", "iq2_s", "iq4_xs"):
    main = "~/titan-engine/oxide-kernels/%s/src/main.rs" % f
    txt = open(main).read()
    mods = sorted(set(re.findall(r'"((?:titan_)?i[a-z0-9_]*)"', txt)))
    print(f, "->", [m for m in mods if "titan" in m or m.startswith("iq")][:4])
