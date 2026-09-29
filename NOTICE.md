# Third-party notices

titan-engine contains, modifies or translates code from the projects below. Each keeps its own licence. Full licence
texts are in the files named here; the MIT and Apache-2.0 texts are also reproduced at the end of this file.

| project | where it appears in this repository | licence | copyright |
|---|---|---|---|
| [mistral.rs](https://github.com/EricLBuehler/mistral.rs) v0.9.4 | `mistral.rs/` (fork, branch `titan-094`); `oxide-kernels/mistralrs-*` and `oxide-kernels/titan-oxide-ffi` are cuda-oxide (Rust) translations of its CUDA kernels and host launchers | MIT (`mistral.rs/LICENSE`) | Copyright (c) 2024 Eric Buehler |
| [candle](https://github.com/huggingface/candle) at a9667ca | `candle/` (fork); `oxide-kernels/candle-*` are translations of `candle-kernels` | Apache-2.0 OR MIT (`candle/LICENSE-APACHE`, `candle/LICENSE-MIT`) | Copyright the candle authors (Hugging Face and contributors) |
| [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) at acecd56 | quantized GEMV/MMQ/MMQ-MoE kernels ported for bit-identity: `oxide-kernels/{fmt_mmq,q1_0,iq4_nl,mxfp4,nvfp4,mmvq-moe,mmvq-rows,candle-mmq,candle-mmvq,candle-quantized,candle-moe}`, their embedded PTX in `mistral.rs/mistralrs-quant/src/gguf/*_oxide.ptx` and `candle/candle-core/src/quantized/*_oxide.ptx`, the launchers in `mistral.rs/mistralrs-quant/src/gguf/fast_mmq.rs` / `fast_mmvq.rs`; GGUF block formats and CPU dot products in `mistral.rs/mistralrs-quant/src/gguf/titan_cpu.rs`; `bench/kbench` harness links llama.cpp | MIT | Copyright (c) 2023-2026 The ggml authors |
| [vLLM](https://github.com/vllm-project/vllm) (via mistral.rs) | paged-attention kernels translated in `oxide-kernels/mistralrs-paged-attn-{a,b}` | Apache-2.0 | Copyright (c) 2023, The vLLM team |
| [FasterTransformer](https://github.com/NVIDIA/FasterTransformer) (via vLLM / mistral.rs) | attention utility code in the same paged-attention translations | Apache-2.0 | Copyright (c) 2020-2023, NVIDIA CORPORATION |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) (via mistral.rs) | decode-attention kernels translated in `oxide-kernels/mistralrs-paged-attn-b` | Apache-2.0 | Copyright (c) 2023 by FlashInfer team |
| [Marlin](https://github.com/IST-DASLab/marlin) (via mistral.rs) | `oxide-kernels/mistralrs-quant-b/marlin-kernels` | Apache-2.0 | Copyright (C) Marlin.2024 Elias Frantar |
| [cuda-oxide](https://github.com/NVlabs/cuda-oxide) at b9847e9 | build-time dependency (`cuda-device`, `cuda-host` crates and the `rustc-codegen-cuda` backend) that compiles `oxide-kernels` to PTX; `cuda-oxide-fast/` holds a patch to it; the committed PTX may contain inlined code from its device crates | Apache-2.0 | Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES |

Modifications: every file listed above that titan-engine changed or translated differs from its upstream. The
changes are the difference against the upstream revision named in the table (see `UPSTREAM.md` for how to produce
the diff). Translated kernels keep the entry names and argument ABI of the original and are gated bit-identical
against it; that is why they are derivative works and carry the original licence.

Ideas, not code: several techniques (a mapped-memory doorbell for CPU experts, one CPU pass per expert miss,
prefill expert streaming, and others) were first seen described in [Strata](https://github.com/Niko1221/Strata).
Strata publishes no licence; no Strata code, text or identifiers were copied. The implementations here are
independent and were written from the descriptions.

Models are not part of this repository. Model weights (Qwen, gpt-oss, Bonsai and others) are distributed by their
authors under their own licences.

---

## MIT licence text (mistral.rs, llama.cpp/ggml, candle's MIT option)

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Apache License 2.0 (vLLM, FasterTransformer, FlashInfer, Marlin, cuda-oxide, candle's Apache option)

The full text is in `candle/LICENSE-APACHE` (identical to https://www.apache.org/licenses/LICENSE-2.0.txt).
