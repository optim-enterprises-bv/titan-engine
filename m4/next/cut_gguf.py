#!/usr/bin/env python3
"""Cut the first N trunk layers of a GGUF into a smaller GGUF of the same arch (block_count = N).

    python3 cut_gguf.py SRC DST N

Metadata is copied verbatim except {arch}.block_count; non-layer tensors and blk.0..N-1 are kept.
Works on a partially downloaded SRC as long as the kept tensors' bytes are present.
"""
import struct
import sys

ALIGN_DEFAULT = 32


def main(src, dst, n):
    f = open(src, "rb")
    rd = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))

    def rstr():
        (ln,) = rd("Q")
        return f.read(ln)

    def skip_value(t):
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if t in sizes:
            f.seek(sizes[t], 1)
        elif t == 8:
            rstr()
        elif t == 9:
            (et,) = rd("I")
            (cnt,) = rd("Q")
            if et in sizes:
                f.seek(sizes[et] * cnt, 1)
            else:
                for _ in range(cnt):
                    skip_value(et)
        else:
            raise ValueError(f"type {t}")

    magic = f.read(4)
    assert magic == b"GGUF", magic
    (version,) = rd("I")
    n_tensors, n_kv = rd("QQ")
    kvs = []  # (key, raw bytes of the whole kv)
    arch = None
    align = ALIGN_DEFAULT
    for _ in range(n_kv):
        start = f.tell()
        key = rstr().decode()
        (t,) = rd("I")
        vstart = f.tell()
        if key == "general.architecture":
            arch = rstr().decode()
        elif key == "general.alignment":
            (align,) = rd("I")
        else:
            skip_value(t)
        end = f.tell()
        f.seek(start)
        kvs.append((key, t, f.read(end - start)))
    infos = []
    for _ in range(n_tensors):
        name = rstr().decode()
        (nd,) = rd("I")
        dims = rd("Q" * nd)
        (typ,) = rd("I")
        (off,) = rd("Q")
        infos.append((name, dims, typ, off))
    data_start = (f.tell() + align - 1) // align * align

    def keep(name):
        if not name.startswith("blk."):
            return True
        return int(name.split(".")[1]) < n

    # byte sizes from the next offset (sorted), last tensor to EOF
    offs = sorted(o for _, _, _, o in infos)
    import os
    fsize = os.path.getsize(src)
    nxt = {o: (offs[i + 1] if i + 1 < len(offs) else fsize - data_start) for i, o in enumerate(offs)}

    kept = [i for i in infos if keep(i[0])]
    out = open(dst, "wb")
    w = lambda fmt, *v: out.write(struct.pack("<" + fmt, *v))
    out.write(b"GGUF")
    w("I", version)
    w("QQ", len(kept), len(kvs))
    for key, t, raw in kvs:
        if key == f"{arch}.block_count":
            kb = key.encode()
            out.write(struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 4) + struct.pack("<I", n))
        else:
            out.write(raw)
    new_off = 0
    plan = []
    for name, dims, typ, off in kept:
        size = nxt[off] - off  # includes the source's alignment padding; harmless
        nb = name.encode()
        w("Q", len(nb)); out.write(nb)
        w("I", len(dims)); w("Q" * len(dims), *dims)
        w("I", typ); w("Q", new_off)
        plan.append((off, size, new_off))
        new_off = (new_off + size + align - 1) // align * align
    pos = out.tell()
    out.write(b"\0" * ((pos + align - 1) // align * align - pos))
    dstart = out.tell()
    for off, size, no in plan:
        if data_start + off + size > fsize:
            sys.exit(f"source truncated: need {data_start + off + size}, have {fsize}")
        out.seek(dstart + no)
        f.seek(data_start + off)
        left = size
        while left:
            chunk = f.read(min(left, 64 << 20))
            out.write(chunk)
            left -= len(chunk)
    out.truncate()
    print(f"{arch}: kept {len(kept)}/{n_tensors} tensors, block_count {n}, {out.tell() >> 20} MiB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
