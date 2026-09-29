#!/usr/bin/env bash
# Copies the four reference static libraries into .ref/ with every exported C launcher symbol
# renamed <prefix><name> (rm_ libmoe.a, rq_ libmistralrsquant.a, rc_ libmistralrscuda.a,
# rp_ libmistralrspagedattention.a), so the gate example can link the real C launchers next to
# this crate's same-named Rust exports. Only needed for the gate; the library never uses .ref/.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ref=$here/../reference
mkdir -p "$here/.ref"
one() { # <src .a> <prefix> <out name>
  local src=$1 pre=$2 out=$here/.ref/lib$3.a map
  map=$(mktemp -p "$here/.ref")
  nm -g --defined-only "$src" 2>/dev/null | awk '$2=="T"{print $3}' | grep -v '^_Z' | sort -u | awk -v p="$pre" '{print $1" "p$1}' > "$map"
  objcopy --redefine-syms="$map" "$src" "$out"
  echo "$(wc -l < "$map") symbols renamed ${pre}* -> $out"
  rm -f "$map"
}
one "$ref/candle-ffi/libmoe.a" rm_ ref_moe
one "$ref/mistralrs-quant/libmistralrsquant.a" rq_ ref_quant
one "$ref/mistralrs-core/libmistralrscuda.a" rc_ ref_core
one "$ref/mistralrs-paged-attn/libmistralrspagedattention.a" rp_ ref_pa
