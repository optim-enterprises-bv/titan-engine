# Upstream bases

| directory | upstream | base | how to see our changes |
|---|---|---|---|
| `mistral.rs/` | https://github.com/EricLBuehler/mistral.rs | tag `v0.9.4` | clone upstream, check out `v0.9.4`, copy this directory over it, `git diff` |
| `candle/` | https://github.com/huggingface/candle | commit `a9667ca` | same, against `a9667ca` |
| `oxide-kernels/` | (new) cuda-oxide translations of candle, mistral.rs and llama.cpp kernels | llama.cpp `acecd56` for the ported ggml kernels | each crate's `gate` compares against the upstream nvcc build |
| cuda-oxide (not vendored) | https://github.com/NVlabs/cuda-oxide | commit `b9847e9` + `cuda-oxide-fast/0001-*.patch` | the patch file |

Directory layout matters: `mistral.rs/Cargo.toml` patches candle from `../candle`, and the `oxide` feature takes
`titan-oxide-ffi` from `../../oxide-kernels` relative to each crate. Keep the three directories side by side.
