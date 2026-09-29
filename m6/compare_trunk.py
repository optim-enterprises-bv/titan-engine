import os
import struct, hashlib, sys
BS = {0:(1,4),1:(1,2),8:(32,34),12:(256,144),13:(256,176),14:(256,210),30:(1,2)}
def parse(path):
    f = open(path, 'rb')
    def rd(fmt):
        n = struct.calcsize(fmt); return struct.unpack('<'+fmt, f.read(n))
    def s(): n, = rd('Q'); return f.read(n).decode('utf8','replace')
    sc = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
    def val(t):
        if t in sc: return rd(sc[t])[0]
        if t == 8: return s()
        if t == 9:
            et, = rd('I'); n, = rd('Q')
            return [val(et) for _ in range(n)]
    magic = f.read(4); ver, = rd('I'); nt, = rd('Q'); nkv, = rd('Q')
    align = 32
    for _ in range(nkv):
        k = s(); t, = rd('I'); v = val(t)
        if k == 'general.alignment': align = v
    ts = {}
    for _ in range(nt):
        name = s(); nd, = rd('I'); dims = rd('Q'*nd); typ, = rd('I'); off, = rd('Q')
        ts[name] = (dims, typ, off)
    pos = f.tell(); base = (pos + align - 1)//align*align
    return f, ts, base
def size(dims, typ):
    n = 1
    for d in dims: n *= d
    b, t = BS[typ]; return n // b * t
fa, ta, ba = parse(os.path.expanduser("~/ai/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"))
fb, tb, bb = parse(os.path.expanduser("~/ai/models/Qwen3.6-35B-A3B-MTP/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"))
print("A", len(ta), "B", len(tb), "only in B:", sorted(set(tb)-set(ta))[:4], "... only in A:", sorted(set(ta)-set(tb)))
diff = 0; tot = 0
for n, (d, t, o) in ta.items():
    d2, t2, o2 = tb[n]
    if d != d2 or t != t2: print("meta diff", n); diff += 1; continue
    sz = size(d, t); tot += sz
    fa.seek(ba + o); fb.seek(bb + o2)
    ha, hb = hashlib.md5(), hashlib.md5(); left = sz
    while left:
        c = min(left, 1 << 24); ha.update(fa.read(c)); hb.update(fb.read(c)); left -= c
    if ha.digest() != hb.digest(): print("data diff", n); diff += 1
print("trunk tensors compared:", len(ta), "bytes", tot, "differing:", diff)
