#!/usr/bin/env bash
# Proves that this crate exports exactly the C launcher symbols mistral.rs links from the four nvcc
# static libraries, resolved as the linked binary resolves them.
#
# Target set = union of `nm -g --defined-only <lib> | awk '$2=="T"' | grep -v '^_Z'` over libmoe.a,
# libmistralrsquant.a, libmistralrscuda.a, libmistralrspagedattention.a (duplicates count once:
# the binary uses one definition), MINUS the host stubs of `extern "C" __global__` kernels, i.e.
# symbols that are also a kernel (`.text.<name>` section) in the device code of those libraries.
# Such a stub only exists to be called through `<<<>>>` from CUDA C++; no Rust code declares them
# (listed by name below). Everything else must be exported by the crate, and nothing more.
#
# Exported set = `T` symbols of this crate's own objects in the built rlib/staticlib that are not
# Rust-mangled (_R/_ZN...17h) and not the internal `titan_oxide_ffi_*` embedding symbols (hidden).
# Usage: ./check_symbols.sh [--verbose]   (after cargo build --release)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ref=$here/../reference
libs=("$ref/candle-ffi/libmoe.a" "$ref/mistralrs-quant/libmistralrsquant.a" "$ref/mistralrs-core/libmistralrscuda.a"
      "$ref/mistralrs-paged-attn-094/libmistralrspagedattention.a")
tmp=$(mktemp -d -p "$here/.mut" chk.XXXX)
trap 'rm -rf "$tmp"' EXIT
for l in "${libs[@]}"; do
  nm -g --defined-only "$l" 2>/dev/null | awk '$2=="T"{print $3}' | grep -v '^_Z' | sort -u > "$tmp/$(basename "$l").syms"
done
sort -u "$tmp"/*.syms > "$tmp/union"
# kernels of the device code: extract every fatbin ELF's section names with cuobjdump if present,
# else from the reference cubins shipped next to the libraries.
cuobj=$(command -v cuobjdump || true); [ -z "$cuobj" ] && [ -x "$here/../tools/cuobjdump" ] && cuobj=$here/../tools/cuobjdump
: > "$tmp/kernels"
if [ -n "$cuobj" ]; then
  for l in "${libs[@]}"; do "$cuobj" -elf "$l" 2>/dev/null | grep -o '\.text\.[A-Za-z_0-9$]*' | sed 's/^\.text\.//' >> "$tmp/kernels" || true; done
fi
if [ ! -s "$tmp/kernels" ]; then
  for f in "$ref"/*/*.cubin; do readelf -SW "$f" 2>/dev/null | grep -o '\.text\.[^ ]*' | sed 's/^\.text\.//'; done >> "$tmp/kernels"
fi
sort -u -o "$tmp/kernels" "$tmp/kernels"
comm -12 "$tmp/union" "$tmp/kernels" > "$tmp/stubs"
comm -23 "$tmp/union" "$tmp/stubs" > "$tmp/target"
cp "$tmp/union" "$here/.mut/union_symbols.txt"; cp "$tmp/stubs" "$here/.mut/kernel_stubs.txt"  # for abi_check.py

check() { # <artifact>
  local art=$1
  # only this crate's own objects (the staticlib also carries std / compiler_builtins)
  nm -A -g --defined-only "$art" 2>/dev/null | awk '$(NF-1)=="T" && $1 ~ /titan_oxide_ffi-/ {print $NF}' \
    | grep -v -E '^_R|^_ZN.*17h[0-9a-f]{16}E$|^titan_oxide_ffi_' | sort -u > "$tmp/have"
  local missing extra
  missing=$(comm -23 "$tmp/target" "$tmp/have"); extra=$(comm -13 "$tmp/target" "$tmp/have")
  echo "$(basename "$art"): exports $(wc -l < "$tmp/have") C symbols; target $(wc -l < "$tmp/target");" \
       "missing $( [ -z "$missing" ] && echo 0 || echo "$missing" | wc -l ); extra $( [ -z "$extra" ] && echo 0 || echo "$extra" | wc -l )"
  [ -n "$missing" ] && { echo "MISSING:"; echo "$missing" | sed 's/^/  /'; }
  [ -n "$extra" ] && { echo "EXTRA:"; echo "$extra" | sed 's/^/  /'; }
  [ -z "$missing" ] && [ -z "$extra" ]
}

echo "union of the four libraries: $(wc -l < "$tmp/union") C symbols" \
     "($(cat "$tmp"/*.a.syms | wc -l) with duplicates: $(cat "$tmp"/*.a.syms | sort | uniq -d | wc -l) defined twice)"
echo "  of which extern \"C\" __global__ kernel host stubs (not launchers, excluded): $(wc -l < "$tmp/stubs")"
echo "  launchers to export: $(wc -l < "$tmp/target")"
# no Rust source in mistral.rs / candle may declare a kernel stub as an extern fn
decl=$(grep -rhoE --include='*.rs' 'fn [A-Za-z_0-9]+\s*\(' "$here/../../mistral.rs" "$here/../../candle" 2>/dev/null \
       | grep -v '/target/' | sed -E 's/fn ([A-Za-z_0-9]+).*/\1/' | sort -u | comm -12 - "$tmp/stubs" || true)
[ -n "$decl" ] && echo "  note: Rust fns named like a kernel stub (check they are not extern decls): $(echo $decl)"
[ "${1:-}" = "--verbose" ] && { echo "kernel stubs:"; sed 's/^/  /' "$tmp/stubs"; }
rc=0
for art in "$here/target/release/libtitan_oxide_ffi.rlib" "$here/target/release/libtitan_oxide_ffi.a"; do
  [ -f "$art" ] && { check "$art" || rc=1; }
done
[ $rc = 0 ] && echo "SYMBOLS: PASS" || echo "SYMBOLS: FAIL"
exit $rc
