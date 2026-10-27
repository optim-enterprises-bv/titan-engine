# titan-variants

Variant + route codegen for titan-engine's oxide kernels, ported from the two mechanisms that make
`trymirai/uzu`'s kernel layer hold together:

| uzu | here |
|---|---|
| `crates/uzu-engine/build/metal/variant_combinations.rs` (prune the variant product) | `constrained_combinations` in `src/main.rs`, same search order, same output order |
| `build/metal/wrapper.rs` (emit one entry point per surviving combination) | `[kernel]` templates + axes in `specs/*.toml` |
| `build/common/traitgen.rs` (one trait per kernel from the parsed signature) | not ported yet — see "next" below |
| `src/backends/metal/kernel/matmul/qmv/routes.rs` (measured per-device dispatch table) | `route` mode, `RouteRow { .., measured_by }` |

## Modes

```
titan-variants gen    <spec>   # emit the #[kernel] region (stdout)
titan-variants verify <spec>   # regenerate and diff names + text against an existing crate
titan-variants prune  <spec>   # naive product vs viable combinations
titan-variants route  <spec>   # emit route()/RouteRow from [[route.row]] entries
```

`verify` is the point: a spec is only accepted once it reproduces a crate's existing generated region
byte for byte. Both current specs pass:

```
specs/candle-mmvq.toml  240 entry points, 240 lines, 0 diffs
specs/mmvq-moe.toml      34 entry points, 136 lines, 0 diffs  (42 naive -> 34 viable)
```

## Spec shape

```toml
constraints = ['variant.suffix != "_b1_llama" || fmt.ftz == false']   # root level, before any table

[kernel]
args = "..."                 # substituted as {args} in entry
call = "(...)"               # substituted as {call}
name_template = "{fmt.name}_moe{variant.suffix}"
entry = '''    #[kernel] pub unsafe fn {name}({args}) {{ {variant.body}{call} }}'''
begin = "// GENERATED KERNELS BEGIN"   # marker pair in the target crate
end = "// GENERATED KERNELS END"
target = "../../oxide-kernels/mmvq-moe/src/main.rs"

[[axis]]
name = "fmt"
[[axis.value]]
name = "q4k"
ftz = false
```

Fields are addressable as `{<axis>.<field>}` anywhere in the templates, including inside another
axis value's own text (a variant's `doc` or `body` can reference `{fmt.name}`). `{{`/`}}` emit literal
braces.

## Route rows

`specs/route-5080.toml` carries the first measured rows (2026-10-27): decode attention eager below
1024 keys / flash-decode above, and the Q3_K single-row matvec tie-break. `route` emits
`out/route_5080.rs` (`RouteRow { .. keys_lo, keys_hi, measured_by }` + `route()`); a 5-assertion
harness in `/tmp/routetest` covers exact-device hit, sub-threshold fallback, same-arch agreement,
unknown arch -> `None`, and an uncovered `m` -> `None`.

`keys` is a dispatch axis because the flash-decode decision is a function of KV depth, not rows.


Every row requires `measured_by`. The generator refuses to emit a table row it cannot attribute, so
the table cannot turn into folklore the way an env-var default does. Selection follows uzu's
`route()`: exact device, else same arch only when all measured rows agree, else `None` (generic
kernel) — never a guess.

## Next, in order

1. `traitgen` equivalent: generate a `trait <Kernel>Kernel` from the same spec so the oxide kernel and
   its `src/cpu.rs` twin cannot drift in signature or coverage.
2. Replace the remaining per-crate `gen_kernels.py` files crate by crate (`verify` proves each step).
3. Feed `route` from `ab-*.sh` output instead of hand-copied rows.
