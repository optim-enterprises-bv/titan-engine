#!/usr/bin/env python3
"""Read the GGUF key/value header (no tensor data). usage: ggufmeta.py FILE [substr ...] -> JSON of matching keys.
Arrays longer than 16 elements are summarised as their length (the tokenizer vocab is skipped that way)."""
import json, struct, sys

T = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def rd(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def rstr(f):
    return f.read(rd(f, "<Q")).decode("utf-8", "replace")


def val(f, t):
    if t in T:
        return rd(f, T[t])
    if t == 8:
        return rstr(f)
    if t == 9:
        et, n = rd(f, "<I"), rd(f, "<Q")
        if n > 16:
            for _ in range(n):
                val(f, et)
            return f"<array of {n}>"
        return [val(f, et) for _ in range(n)]
    raise ValueError(f"gguf type {t}")


def meta(path, keys=()):
    out = {}
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF", "not a GGUF file"
        ver, ntens, nkv = rd(f, "<I"), rd(f, "<Q"), rd(f, "<Q")
        out["_gguf_version"], out["_tensor_count"] = ver, ntens
        for _ in range(nkv):
            k = rstr(f)
            v = val(f, rd(f, "<I"))
            if not keys or any(s in k for s in keys):
                out[k] = v
    return out


if __name__ == "__main__":
    print(json.dumps(meta(sys.argv[1], sys.argv[2:]), indent=1))
