#!/usr/bin/env python3
"""Score out/NAME.hard.jsonl. HumanEval: the reply's first ```python block (+ the prompt's imports, or the prompt itself
when the block is only a body) + the test + check(entry_point), executed ONLY inside a sandbox: temp dir, systemd scope
MemoryMax=1G TasksMax=32, no network (unshare -rn), timeout 10 s, python3 -I. GSM8K: exact match of the number after
the last '####' (else the last number in the reply) against the reference.
usage: hardscore.py NAME [REF_NAME]  (with REF: paired comparison on the items both have)"""
import gzip, json, math, os, re, subprocess, sys, tempfile
D = os.path.dirname(os.path.abspath(__file__)) + "/data"
he = [json.loads(l) for l in gzip.open(f"{D}/HumanEval.jsonl.gz", "rt")]
gs = [json.loads(l) for l in open(f"{D}/gsm8k-test.jsonl")]

def num(s):
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        v = float(s); return int(v) if v == int(v) else v
    except ValueError:
        return None

def gsm_ok(reply, i):
    ref = num(gs[i]["answer"].split("####")[-1])
    m = re.findall(r"####\s*\$?\s*(-?[\d,]*\.?\d+)", reply)
    if not m:
        m = re.findall(r"-?[\d,]*\.?\d+", reply)
    return bool(m) and num(m[-1]) == ref

def he_program(reply, i):
    p = he[i]
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", reply, re.S)
    code = blocks[0] if blocks else reply
    if re.search(rf"def\s+{re.escape(p['entry_point'])}\s*\(", code):
        imports = "\n".join(l for l in p["prompt"].splitlines() if re.match(r"\s*(import|from)\s", l))
        prog = imports + "\n" + code
    else:
        prog = p["prompt"] + code
    return prog + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})\n"

def he_ok(reply, i):
    with tempfile.TemporaryDirectory() as d:
        open(f"{d}/prog.py", "w").write(he_program(reply, i))
        cmd = ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=1G", "-p", "MemorySwapMax=0", "-p", "TasksMax=32",
               "unshare", "-rn", "timeout", "-k", "2", "10", "python3", "-I", "prog.py"]
        try:
            r = subprocess.run(cmd, cwd=d, capture_output=True, timeout=30, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return False
        return r.returncode == 0

def score(name):
    res = {}
    for r in map(json.loads, open(f"out/{name}.hard.jsonl")):
        res[(r["kind"], r["idx"])] = (he_ok if r["kind"] == "he" else gsm_ok)(r["reply"], r["idx"])
    return res

def binom_p(a, b):  # exact two-sided McNemar on discordant pairs
    n = a + b
    if n == 0: return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n)

name = sys.argv[1]
s = score(name)
json.dump({f"{k}:{i}": v for (k, i), v in s.items()}, open(f"out/{name}.hard.score.json", "w"))
for kind, lab in (("he", "HumanEval pass@1"), ("gs", "GSM8K")):
    v = [x for (k, _), x in s.items() if k == kind]
    print(f"{name}: {lab} {sum(v)}/{len(v)}")
if len(sys.argv) > 2:
    ref = score(sys.argv[2])
    for kind, lab in (("he", "HumanEval"), ("gs", "GSM8K")):
        common = [k for k in s if k in ref and k[0] == kind]
        a = sum(ref[k] and not s[k] for k in common)
        b = sum(s[k] and not ref[k] for k in common)
        print(f"{lab} on {len(common)} common: {sys.argv[2]} {sum(ref[k] for k in common)}, {name} {sum(s[k] for k in common)}; "
              f"{sys.argv[2]}-only {a}, {name}-only {b}, net {b - a:+d}, exact McNemar p = {binom_p(a, b):.3f}")
