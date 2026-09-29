#!/usr/bin/env python3
"""Build the opencode-shaped code-editing benchmark (titan-engine n-gram drafting).

20 prompts over real files of ~/titan-engine/mistral.rs (300-800 lines and under 15k characters: prefill runs at ~250 tok/s here, pinned to commit ktrace@da11fc3
via `git show`, so the set is reproducible): 7 "rewrite the whole file with change Y", 7 "apply this
diff and output the whole function", 6 "fix this compiler/clippy output, output the whole function".
Writes code-bench.json: [{"id", "kind", "file", "messages", "max_tokens"}].
usage: gen_code_bench.py [OUT]
"""
import os
import difflib, json, random, re, subprocess, sys

REPO = os.path.expanduser("~/titan-engine/mistral.rs")
REV = "da11fc3"
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/titan-engine/m6/ngram/code-bench.json")


def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], check=True, capture_output=True, text=True).stdout


def functions(src):
    """(name, start line, end line) of `fn` items whose body closes at the same indentation."""
    lines = src.split("\n")
    out = []
    for i, l in enumerate(lines):
        m = re.match(r"^(\s*)(pub(\([^)]*\))? )?(async )?(unsafe )?fn (\w+)", l)
        if not m:
            continue
        ind = m.group(1)
        sig = next((k for k in range(i, min(i + 30, len(lines))) if "{" in lines[k] or lines[k].rstrip().endswith(";")), None)
        if sig is None or "{" not in lines[sig]:
            continue  # declaration without a body
        for j in range(sig, min(i + 400, len(lines))):
            if lines[j] == ind + "}":
                out.append((m.group(6), i, j))
                break
    return out


def main():
    files = []
    for f in git("ls-tree", "-r", "--name-only", REV).split():
        if not f.endswith(".rs") or not f.startswith(("mistralrs-core/src", "mistralrs-quant/src", "mistralrs-server-core/src")):
            continue
        src = git("show", f"{REV}:{f}")
        n = src.count("\n")
        if 300 <= n <= 800 and len(src) < 15000:
            files.append((f, src))
    rng = random.Random(20260928)
    rng.shuffle(files)
    prompts, used = [], set()
    kinds = ["file"] * 7 + ["diff"] * 7 + ["fix"] * 6
    for kind in kinds:
        while True:
            f, src = files.pop()
            fns = [x for x in functions(src) if 12 <= x[2] - x[1] <= 90]
            if len(fns) >= 2 and f not in used:
                break
        used.add(f)
        lines = src.split("\n")
        name, a, b = rng.choice(fns)
        head = f"Here is the file `{f}`:\n\n```rust\n{src}\n```\n\n"
        if kind == "file":
            new = name + "_impl" if not name.endswith("_impl") else name + "2"
            ask = (f"Rename the function `{name}` to `{new}` and update every use of it in this file. "
                   "Output the complete updated file in one ```rust block, with no other text.")
            max_tokens = 1024
        elif kind == "diff":
            body = lines[a:b + 1]
            ind = re.match(r"^(\s*)", body[0]).group(1) + "    "
            changed = body[:1] + [f'{ind}tracing::trace!("{name}: enter");'] + body[1:]
            # rename one local `let` binding if there is one
            lets = [m.group(1) for l in body for m in [re.match(r"^\s*let (?:mut )?([a-z_][a-z0-9_]{2,})\b", l)] if m]
            if lets:
                v = lets[0]
                changed = [re.sub(rf"\b{v}\b", v + "_v", l) for l in changed]
            diff = "\n".join(difflib.unified_diff(body, changed, f"a/{f}", f"b/{f}", lineterm="", n=3))
            ask = (f"Apply this diff to the file:\n\n```diff\n{diff}\n```\n\n"
                   f"Output the whole function `{name}` after the change in one ```rust block, with no other text.")
            max_tokens = 768
        else:
            ask = (f"`cargo clippy` reports:\n\n```\nwarning: this function has too many lines\n"
                   f"  --> {f}:{a + 1}:1\n   |\n{a + 1:>3} | {lines[a].strip()}\n   |\n"
                   f"   = help: for further information visit https://rust-lang.github.io/rust-clippy/master/index.html#too_many_lines\n```\n\n"
                   f"Silence it with `#[allow(clippy::too_many_lines)]` on `{name}` and output the whole function "
                   "`{name}` (attribute included) in one ```rust block, with no other text.").replace("{name}", name)
            max_tokens = 768
        prompts.append({
            "id": len(prompts), "kind": kind, "file": f, "fn": name, "lines": src.count("\n"),
            "messages": [{"role": "user", "content": head + ask}], "max_tokens": max_tokens,
        })
    json.dump(prompts, open(OUT, "w"), indent=1)
    chars = sum(len(p["messages"][0]["content"]) for p in prompts)
    print(f"{len(prompts)} prompts, {chars} chars (~{chars // 3.3:.0f} tokens), lines "
          f"{[p['lines'] for p in prompts]}")


main()
