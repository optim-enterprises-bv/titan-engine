#!/usr/bin/env python3
"""Mine the mul_mat_vec_q reference cubin out of a llama.cpp CUDA build.

    uv run --with zstandard python scripts/mine-mmvq-ref.py <libggml-cuda.so|ggml-cuda.dll>
    -> oxide-kernels/reference/llama/mmvq-sm120.cubin

Cubins are OS-independent, so a Windows release bundle works fine on Linux. Any build carrying
the MUL_MAT_ID decode kernel for Q1_0 / IQ4_NL / MXFP4 / NVFP4 will do -- the PrismML fork's sm_120
bundle is the one vendored here, because upstream at the NOTICE.md pin does not execute this kernel
on Blackwell (see the doc comment on llama_gate in src/main.rs).

Two template signatures are searched for: <type, ncols_dst, has_fusion, small_k, halve_iters> and
the newer form with an extra has_gate bool.
"""
"""Pick the fatbin payload for the *right* arch, not the first one that has the symbol."""
import struct, sys, re, zstandard as zstd
MAGIC = struct.pack('<I', 0xBA55ED50)
TYPES = [41, 20, 39, 40]
WANT_ARCH = "sm_120"

def payloads(d):
    pos = 0
    while True:
        pos = d.find(MAGIC, pos)
        if pos < 0: return
        try:
            _v,_h = struct.unpack_from('<HH', d, pos+4); fat, = struct.unpack_from('<Q', d, pos+8)
        except struct.error: pos += 4; continue
        p = pos+16; end = pos+fat if 0 < fat <= len(d)-pos else len(d)
        while p+24 <= end:
            try:
                kind,_ev = struct.unpack_from('<HH', d, p); hdr, = struct.unpack_from('<I', d, p+4); padded, = struct.unpack_from('<Q', d, p+8)
            except struct.error: break
            if hdr < 16 or hdr > 8192 or padded == 0 or p+hdr+padded > len(d): break
            raw = d[p+hdr:p+hdr+padded]
            try: body = zstd.ZstdDecompressor().decompress(raw, max_output_size=64<<20)
            except Exception: body = raw
            yield kind, body
            p += hdr+padded
        pos += 4

d = open(sys.argv[1],'rb').read()
cands = {}
for kind, body in payloads(d):
    arches = set(m.group(0).decode() for m in re.finditer(rb'sm_1?[0-9]{2}', body))
    for t in TYPES:
        if f"mul_mat_vec_qIL9ggml_type{t}ELi1ELb0ELb0ELb0ELb0EE".encode() in body:
            cands.setdefault(t, []).append((kind, arches, body))
for t, lst in sorted(cands.items()):
    print(f"type {t}: {len(lst)} candidate payloads")
    pick = None
    for kind, arches, body in lst:
        print(f"   kind={kind} arches={sorted(arches)} size={len(body)/1024:.0f}KiB")
        if WANT_ARCH in arches and kind == 2:
            pick = (kind, body)   # cubin first: the gate's spec is SASS, not JIT'd PTX
        elif WANT_ARCH in arches and pick is None:
            pick = (kind, body)
    if pick is None:
        # PTX (kind 1) JITs for the device regardless of the arch string it was generated for
        for kind, arches, body in lst:
            if kind == 1: pick = (kind, body); break
    if pick:
        kind, body = pick
        ext = "ptx" if kind==1 else "cubin"
        path = f"/tmp/refs120/{t}.{ext}"
        import os; os.makedirs("/tmp/refs120", exist_ok=True)
        open(path,'wb').write(body)
        print(f"   -> picked kind={kind} ({ext}) {len(body)/1024:.0f} KiB -> {path}")
