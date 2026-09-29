#!/usr/bin/env python3
"""Compare the SASS of every kernel matching PATTERN in two cubins (names normalised: the
per-build INTERNAL_<hash> namespace is dropped). Prints SAME/DIFF per kernel."""
import re, subprocess, sys, os
T = os.path.expanduser('~/titan-engine/oxide-kernels/tools/cuobjdump')
def funcs(cubin):
    out = subprocess.run([T, '-sass', cubin], capture_output=True, text=True).stdout
    res = {}; cur = None
    for line in out.splitlines():
        m = re.match(r'\s*Function : (\S+)', line)
        if m:
            cur = re.sub(r'N\d+_INTERNAL_\w*?_cu_\w*?(?=(dequantize_|vec_dot_|mul_mat_|quantize_))', 'N_I_', m.group(1)); res[cur] = []; continue
        if cur and re.match(r'\s*/\*[0-9a-f]{4}\*/', line):
            ins = re.sub(r'/\*[0-9a-f]{4}\*/', '', line).split(';')[0].strip()
            ins = re.sub(r'`\([^)]*\)', '', ins)
            res[cur].append(ins)
    return res
a, b, pat = sys.argv[1], sys.argv[2], sys.argv[3]
fa, fb = funcs(a), funcs(b)
ka = sorted(k for k in fa if re.search(pat, k)); kb = sorted(k for k in fb if re.search(pat, k))
print(len(ka), 'kernels in A,', len(kb), 'in B')
for k in ka:
    if k not in fb: print('MISSING in B', k); continue
    print('SAME' if fa[k] == fb[k] else 'DIFF', len(fa[k]), k[:140])
