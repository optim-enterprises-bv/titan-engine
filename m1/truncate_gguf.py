"""Write a GGUF keeping only the first N transformer blocks, every byte of them unchanged.
Used for titan-engine gates that need a MoE model whose experts all fit on the 5080."""
import os
import sys
sys.path.insert(0, os.path.expanduser("~/ai/llama.cpp/gguf-py"))
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType

src, dst, keep = sys.argv[1], sys.argv[2], int(sys.argv[3])
r = GGUFReader(src)
arch = r.fields["general.architecture"].contents()
w = GGUFWriter(dst, arch)
for name, f in r.fields.items():
    if name.startswith("GGUF.") or name == "general.architecture":
        continue
    if name == f"{arch}.block_count":
        w.add_uint32(name, keep)
        continue
    vt = f.types[0]
    if vt == GGUFValueType.ARRAY:
        w.add_array(name, f.contents())
    else:
        w.add_key_value(name, f.contents(), vt)
kept = 0
for t in r.tensors:
    if t.name.startswith("blk.") and int(t.name.split(".")[1]) >= keep:
        continue
    w.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)  # byte shape from the reader
    kept += 1
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(progress=False); w.close()
print(f"{kept} tensors, block_count={keep} -> {dst}")
