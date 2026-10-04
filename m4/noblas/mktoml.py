#!/usr/bin/env python3
"""mktoml.py OUTDIR PORT [--no-graphs]: one from-config toml per [[models]] block of deploy/models.toml (the deployed
roster semantics: that block verbatim, default_model_id = it, server port PORT); prints the model names in order.
--no-graphs drops TITAN_CUDA_GRAPHS* (inventory runs: every decode step reaches the host-side dispatch)."""
import re, sys
src = open("~/titan-engine/deploy/models.toml").read()
out, port = sys.argv[1], sys.argv[2]
nog = "--no-graphs" in sys.argv
parts = re.split(r"(?m)^\[\[models\]\]\n", src)
head, blocks = parts[0], parts[1:]
head = re.sub(r"(?m)^port = \d+", f"port = {port}", head)
for b in blocks:
    # trailing comment lines belong to the next block's description
    name = re.search(r'(?m)^name = "([^"]+)"', b).group(1)
    if nog:
        b = "\n".join(l for l in b.splitlines() if not l.startswith("TITAN_CUDA_GRAPHS")) + "\n"
    h = re.sub(r'(?m)^default_model_id = .*$', f'default_model_id = "{name}"', head)
    open(f"{out}/{name}.toml", "w").write(h + "[[models]]\n" + b)
    print(name)
