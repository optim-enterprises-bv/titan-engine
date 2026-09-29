#!/usr/bin/env python3
"""Compare the C ABI of every exported launcher with the extern "C" declarations mistral.rs and
candle actually call it through.

1. Extracts every `extern "C" { ... }` block (plus the file's macro_rules, which generate some
   declarations via paste!) from the Rust files of mistralrs-quant / mistralrs-core /
   mistralrs-paged-attn / candle-kernels that declare foreign launchers, drops them into a scratch
   crate (.mut/abi-decls) and macro-expands it (RUSTC_BOOTSTRAP=1 -Zunpretty=expanded).
2. Macro-expands this crate the same way and reads every `#[no_mangle] extern "C" fn`.
3. Classifies each parameter / return type by its x86-64 SysV class and width
   (pointers and 64-bit integers are both one INTEGER eightbyte; i32/u32 are one 32-bit INTEGER;
   f16/bf16/u16 by value are 16-bit INTEGER; f32 SSE; bool/u8/i8/fp8 8-bit INTEGER) and reports:
     MISMATCH  a different class/width or arity (silent corruption)
     note      same class, different Rust spelling (e.g. i64 vs *mut c_void stream, *const vs *mut,
               i32 vs u32): ABI-identical
     undeclared exported launchers no Rust code declares (never called by mistral.rs)
     unexported declarations with no exported twin (would be an unresolved symbol)
Exit status 1 on any MISMATCH or unexported declaration.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TE = HERE.parent.parent  # ~/titan-engine
SRC_DIRS = [
    TE / "mistral.rs/mistralrs-quant/src",
    TE / "mistral.rs/mistralrs-core/src",
    TE / "mistral.rs/mistralrs-paged-attn/src",
    TE / "candle/candle-kernels/src",
]
SCRATCH = HERE / ".mut/abi-decls"
ENV = dict(os.environ, RUSTC_BOOTSTRAP="1")


def block_end(s, i):
    """Index just past the brace matching s[i] == '{'."""
    depth = 0
    while True:
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1


def strip_comments(s):
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    return re.sub(r"//[^\n]*", "", s)


def extract_decls():
    mods = []
    for d in SRC_DIRS:
        for f in sorted(d.rglob("*.rs")):
            s = strip_comments(f.read_text())
            if 'extern "C"' not in s:
                continue
            macros, blocks = [], []
            for m in re.finditer(r"macro_rules!\s*\w+\s*\{", s):
                macros.append(s[m.start():block_end(s, m.end() - 1)])
            for m in re.finditer(r'(unsafe\s+)?extern\s+"C"\s*\{', s):
                blocks.append(s[m.start():block_end(s, m.end() - 1)])
            if not blocks:
                continue
            name = re.sub(r"\W", "_", str(f.relative_to(TE)))
            body = "\n".join(macros) + "\n" + "\n".join(
                f"pub mod b{i} {{ use super::*; {b} }}" for i, b in enumerate(blocks))
            mods.append((name, str(f.relative_to(TE)), body))
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "src").mkdir(exist_ok=True)
    (SCRATCH / "Cargo.toml").write_text(
        '[package]\nname = "abi-decls"\nversion = "0.0.0"\nedition = "2021"\n[workspace]\n'
        '[dependencies]\nhalf = "=2.7.1"\npaste = "=1.0.15"\nfloat8 = "=0.7.0"\n')
    prelude = ("#![allow(warnings, improper_ctypes)]\n"
               "pub use core::ffi::*;\npub use half::{bf16, f16};\npub use paste::paste;\npub use float8::F8E4M3;\n"
               "pub type CUstream = *mut c_void;\npub type cudaStream_t = *mut c_void;\n"
               "pub mod candle_core { pub mod cuda { pub mod cudarc { pub mod driver { pub mod sys {\n"
               "    pub type CUstream = *mut core::ffi::c_void; } } } } }\n")
    lib = prelude
    for name, rel, body in mods:
        lib += f"\n/// {rel}\n#[allow(unused_imports)]\npub mod {name} {{ use super::*; {body} }}\n"
    (SCRATCH / "src/lib.rs").write_text(lib)
    out = subprocess.run(["cargo", "rustc", "--offline", "-q", "--lib", "--", "-Zunpretty=expanded"], cwd=SCRATCH,
                         env=ENV, capture_output=True, text=True)
    if out.returncode:
        sys.exit("expanding the declarations failed:\n" + out.stderr[-4000:])
    return out.stdout


def expand_self():
    out = subprocess.run(["cargo", "rustc", "--release", "-q", "--lib", "--crate-type", "rlib", "--",
                          "-Zunpretty=expanded"], cwd=HERE, env=dict(ENV, CARGO_TARGET_DIR=str(HERE / "target/expand")),
                         capture_output=True, text=True)
    if out.returncode:
        sys.exit("expanding titan-oxide-ffi failed:\n" + out.stderr[-4000:])
    return out.stdout


def split_params(p):
    parts, depth, cur = [], 0, ""
    for c in p:
        if c in "(<[":
            depth += 1
        elif c in ")>]":
            depth -= 1
        if c == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += c
    if cur.strip():
        parts.append(cur)
    return [x.strip() for x in parts if x.strip()]


def ptype(param):
    t = param.split(":", 1)[1] if ":" in param else param
    return re.sub(r"\s+", " ", t).strip()


def classify(t):
    t = t.replace("::core::ffi::", "").replace("core::ffi::", "").replace("std::ffi::", "").replace("std::os::raw::", "")
    t = re.sub(r"\b\w+::", "", t).strip()
    if t in ("", "()"):
        return "void"
    if t.startswith("*") or t.startswith("&") or t.startswith("Option<") or t.startswith("unsafe extern") or \
            t.startswith("extern") or t in ("CUstream", "cudaStream_t", "CUdeviceptr"):
        return "INT64"
    if t in ("i64", "u64", "isize", "usize", "c_long", "c_ulong", "c_longlong", "c_ulonglong", "size_t"):
        return "INT64"
    if t in ("i32", "u32", "c_int", "c_uint"):
        return "INT32"
    if t in ("i16", "u16", "f16", "bf16", "c_short", "c_ushort"):
        return "INT16"
    if t in ("i8", "u8", "bool", "c_char", "c_uchar", "c_schar", "F8E4M3"):
        return "INT8"
    if t in ("f32", "c_float"):
        return "SSE32"
    if t in ("f64", "c_double"):
        return "SSE64"
    return "?" + t


# Symbols defined both in candle's libmoe.a and in a mistral.rs library; the binary links the
# mistral.rs library's definition (mistralrs-quant for the GGUF mmvq/mmq family,
# mistralrs-core for moe_gemm_wmma), and so does this crate.
SHADOWED_IN_BINARY = set()


FN_RE = re.compile(r"(?:pub(?:\([^)]*\))?\s+)?(?:unsafe\s+)?fn\s+(\w+)\s*\((.*?)\)\s*(?:->\s*([^;{]+?))?\s*;", re.S)


def parse_decls(text):
    decls = {}
    mods = [(m.start(), m.group(1)) for m in re.finditer(r"(?m)^pub mod (\w+_rs) \{", text)]
    # declarations live in `extern "C" { ... }` blocks of the expanded scratch crate
    for m in re.finditer(r'extern\s+"C"\s*\{', text):
        src = [n for (pos, n) in mods if pos < m.start()][-1]
        blk = text[m.end():block_end(text, m.end() - 1) - 1]
        for f in FN_RE.finditer(blk):
            name, params, ret = f.group(1), f.group(2), (f.group(3) or "").strip()
            sig = ([ptype(p) for p in split_params(params)], ret, src)
            decls.setdefault(name, []).append(sig)
    return decls


def parse_exports(text):
    ex = {}
    for m in re.finditer(r'#\[unsafe\(no_mangle\)\]\s*pub\s+unsafe\s+extern\s+"C"\s+fn\s+(\w+)\s*\(', text):
        i = m.end() - 1
        depth, j = 0, i
        while True:
            c = text[j]
            depth += c == "("
            depth -= c == ")"
            j += 1
            if depth == 0:
                break
        params = text[i + 1:j - 1]
        rest = text[j:text.index("{", j)]
        ret = rest.split("->", 1)[1].strip() if "->" in rest else ""
        ex[m.group(1)] = ([ptype(p) for p in split_params(params)], ret)
    return ex


def main():
    ref = HERE.parent / "reference"
    def syms(lib):
        out = subprocess.run(["nm", "-g", "--defined-only", str(lib)], capture_output=True, text=True).stdout
        return {l.split()[2] for l in out.splitlines() if len(l.split()) == 3 and l.split()[1] == "T"}
    moe = syms(ref / "candle-ffi/libmoe.a")
    SHADOWED_IN_BINARY.update(moe & (syms(ref / "mistralrs-quant/libmistralrsquant.a") | syms(ref / "mistralrs-core/libmistralrscuda.a")))
    decls = parse_decls(extract_decls())
    exports = parse_exports(expand_self())
    bad, notes, shadowed = [], [], []
    for name, sigs in sorted(decls.items()):
        if name not in exports:
            continue
        ep, er = exports[name]
        for dp, dr, src in sigs:
            if name in SHADOWED_IN_BINARY and src.startswith("candle_candle_kernels"):
                # candle's declaration of a symbol the linked binary resolves to the mistral.rs
                # library's definition: the ABI is whatever that definition has, before and after.
                if len(dp) != len(ep) or any(classify(a) != classify(b) for a, b in zip(dp, ep)):
                    shadowed.append(f"{name}: candle-kernels declares {len(dp)} params, the resolved definition "
                                    f"(and this crate) takes {len(ep)}")
                continue
            if len(dp) != len(ep):
                bad.append(f"{name}: arity {len(dp)} declared vs {len(ep)} exported\n    decl {dp}\n    ours {ep}")
                continue
            for k, (a, b) in enumerate(zip(dp, ep)):
                ca, cb = classify(a), classify(b)
                if ca != cb or ca.startswith("?"):
                    bad.append(f"{name}: param {k} declared `{a}` ({ca}) vs exported `{b}` ({cb})")
                elif a.replace(" ", "") != b.replace(" ", ""):
                    notes.append(f"{name}: param {k} `{a}` vs `{b}` ({ca})")
            ca, cb = classify(dr), classify(er)
            if ca != cb:
                bad.append(f"{name}: return declared `{dr or '()'}` ({ca}) vs exported `{er or '()'}` ({cb})")
            elif dr.replace(" ", "") != er.replace(" ", ""):
                notes.append(f"{name}: return `{dr}` vs `{er}` ({ca})")
    # declarations of symbols that belong to the four libraries but are not exported
    union = HERE / ".mut/union_symbols.txt"  # written by check_symbols.sh
    if not union.exists():
        sys.exit("run ./check_symbols.sh first (it writes .mut/union_symbols.txt)")
    target = set(union.read_text().split())
    unexported = sorted(n for n in decls if n in target and n not in exports)
    undeclared = sorted(n for n in exports if n not in decls)
    verbose = "--verbose" in sys.argv
    print(f"declarations parsed: {len(decls)} names; exported launchers: {len(exports)}; "
          f"declared+exported: {sum(1 for n in decls if n in exports)}")
    kinds = {}
    for n in notes:
        k = n.split(": ", 1)[1].split(" (")[0]
        k = re.sub(r"param \d+ ", "", k)
        kinds[k] = kinds.get(k, 0) + 1
    print(f"ABI-identical spelling differences: {len(notes)}")
    for k, v in sorted(kinds.items(), key=lambda x: -x[1]):
        print(f"  {v:4}x {k}")
    if verbose:
        for n in notes:
            print("  note", n)
    print(f"exported but declared by no Rust code (mistral.rs never calls them): {len(undeclared)}")
    if verbose or len(undeclared) < 60:
        print("  " + " ".join(undeclared))
    if shadowed:
        print(f"PRE-EXISTING shadowing hazards (same in the nvcc-linked binary; candle's declaration != the "
              f"definition the linker picks): {len(shadowed)}")
        for x in shadowed:
            print("  SHADOWED", x)
    for n in unexported:
        bad.append(f"{n}: declared in Rust and in the libraries' symbol set, but not exported")
    for b in bad:
        print("MISMATCH", b)
    print("ABI: PASS" if not bad else f"ABI: FAIL ({len(bad)})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
