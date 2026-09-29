
# titan-engine

Mixture-of-experts LLM inference for one consumer GPU that is too small for the model: a fork of
[mistral.rs](https://github.com/EricLBuehler/mistral.rs) (v0.9.4) with tiered experts, prefill expert streaming,
multi-token prediction and a hybrid prefix cache, and GPU kernels written in Rust and compiled to PTX with NVIDIA's
[cuda-oxide](https://github.com/NVlabs/cuda-oxide), so the build does not invoke nvcc.

It was built and measured on one machine: an RTX 5080 Laptop GPU (16 GB, 150 W cap), a Core Ultra 9 285HX (24 cores)
and 30 GB of RAM. It serves one user at a time. Treat it as a research engine with a reproducible benchmark, not a
product.

## Results

Single-user numbers from one machine, measured 2026-09-29: ours (build `f601844a6`) against llama.cpp at `acecd56`.
llama.cpp ran its best configuration from a sweep (`--n-cpu-moe 16` (19 with its MTP draft), `-t 6`, `-ub 2048`,
`-fa on`). Prompts are cold (a nonce at the start defeats any prompt cache), decoding is greedy, one request at a time.
Numbers are medians of repeated runs; raw per-run JSON is in [`release/bench/`](release/bench/).

### Qwen3.6-35B-A3B, UD-Q4_K_XL (22.4 GB model on a 16 GB GPU)

| | ours, MTP=2 | ours, MTP off | llama.cpp, MTP draft | llama.cpp, no MTP |
|---|---:|---:|---:|---:|
| decode tok/s (8 short prompts) | **133.0** | 103.4 | 83.7 | 75.7 |
| time to first token, short prompt | **98 ms** | 103 ms | 172 ms | 152 ms |
| prompt tok/s, 4k tokens | 2262 | **2314** | 1539 | 1771 |
| prompt tok/s, 13k tokens | 2576 | **2667** | 1564 | 1711 |
| prompt tok/s, 28k tokens | **2391** | 2384 | 1614 | - |
| prompt tok/s, 40k tokens | 2057 | 2355 | - | - |
| q100 sanity check | 95 | 95 | 96 | - |

Decode is 1.59x llama.cpp with MTP on both sides (1.37x without); prompt processing is 1.47x at 4k, 1.65x at 13k and
1.48x at 28k. A repeated prompt that hits the prefix cache starts generating in 0.4-0.9 s instead of recomputing it.

### Other models (preliminary, not from the controlled campaign)

| model | ours decode tok/s | llama.cpp decode tok/s | note |
|---|---:|---:|---|
| Qwen3-Next-80B-A3B Q4_K_M (48.5 GB) | 41.5 | 15.2 | experts streamed from NVMe; prompt speed is NVMe/page-cache bound on 30 GB RAM and varies run to run |
| gpt-oss-120b MXFP4 (63.4 GB) | 4.4 | 2.8 | older build, 8-prompt set |

What the numbers do not say:
- They are one laptop GPU, one user and one request at a time. There is no batching of concurrent users, and no
  server GPU was used.
- The expert-offload wins depend on this machine's RAM and NVMe. On a card that holds the whole model, the tiering
  does nothing.
- q100 is a sanity check that the output is not broken: 100 short questions, greedy, scored by regex. It is not a
  quality benchmark.

<!-- CLAIMS -->

## How it works

### Tiered experts with a CPU twin
Only a fraction of each MoE layer's experts fits in VRAM. `TITAN_TIERED_GPU_FRACTION=auto` sizes that fraction from
the free memory after the dense weights, KV and a reserve. A hotness profile picks which experts stay resident
(`TITAN_TIERED_PROFILE`, made from calibration prompts). An expert a token needs that is not on the GPU (a miss) is
computed on the CPU in the same step. The CPU kernels are a "twin" of the GPU kernels: they emulate the GPU's
reduction order, so a tiered forward and an all-resident forward produce the same text, byte for byte. As of
2026-09-29 the CPU path may also use faster, numerically different kernels; those are gated statistically (KL,
perplexity and task scores). Host experts are either owned in RAM or read through the GGUF mmap for models larger
than RAM (the 80B and 120B), with a router-lookahead prefetch.

### Doorbell and one-pass CPU experts
With `TITAN_DOORBELL=1`, the expert ids and the quantized input of a layer are published to pinned, mapped host
memory. A host thread answers, and the GPU waits in-stream, so there is no host synchronisation in the MoE layer and
the step stays CUDA-graph capturable. With `TITAN_CPU_ONEPASS=1`, a miss runs gate/up, SwiGLU, the q8_1
requantisation and down on the host in one pass, with no GPU round trip between the two projections. The CPU work
runs on a spinning fork-join pool (waking a sleeping pool per pass cost more than the pass).

### Prefill expert streaming with MMQ-MoE
For a prompt chunk big enough to need most experts, `TITAN_PFS=1` streams the non-resident experts through a ring of
device staging buffers on a copy stream. The grouped expert matmul then runs on the GPU over resident and staged
experts, using a port of llama.cpp's MMQ kernels (MMQ-MoE). Before this, prefill on the tiered path was bound by the
CPU experts.

### Batched MoE decode
Decode and MTP-verify batches (up to 8 rows) use kernels shaped like llama.cpp's `mul_mat_vec_q` MoE grid, ported to
Rust/PTX. Each column is reduced the way the batch-1 kernel does it, so the output bits do not depend on the batch
size.

### Flash-decoding attention for GQA
Decode attention over long contexts uses a split-K flash-decoding kernel. It is on by default from 1024 keys
(`TITAN_ATTN_FLASH`). From 4096 keys, eager GQA attention runs grouped per KV head instead of materialising
`repeat_kv` copies.

### Multi-token prediction (MTP)
For Qwen3.6 GGUFs that carry the MTP head (`nextn_predict_layers`), `TITAN_MTP=N` drafts N tokens with it and
verifies them in one batched forward. Greedy output is the same as without MTP; the release gate checks that.

### Hybrid prefix cache
Qwen3.6 and Qwen3-Next mix Gated DeltaNet (recurrent) layers with attention layers, so a prompt prefix cannot be
resumed from the KV cache alone. When a sequence finishes, its attention KV, the GDN states at resume points and the
MTP state are kept in host memory. A new prompt that shares a prefix resumes from the longest saved point. For agent
loops that is the difference between re-reading a 13k-token system prompt and not.

### Chunked prefill
Unpaged prompt passes run in chunks (default 512 tokens). KV, RoPE and the mask continue at the chunk offset, the GDN
convolution carries state across chunks, and MTP catches up per chunk. KV is reserved in 8192-token buckets, so long
prompts do not fragment the device pool.

### nvcc-free build with cuda-oxide
The kernels in the service path live in `oxide-kernels/`. They are Rust `#[kernel]` functions compiled to PTX by
cuda-oxide's rustc backend: GGUF dequantisation, the Q8_0 and K-quant matvecs, the expert GEMVs, MMQ prefill tiles,
the flash-decode kernel and the doorbell. The PTX is committed, so building the engine needs Rust and the CUDA
libraries and headers (cuBLAS, NVRTC, driver), but no nvcc. Rebuilding the kernels themselves needs cuda-oxide; see
`cuda-oxide-fast/README.md`. Ported kernels are checked against llama.cpp's output bits by the benchmark's kernel tier.

## Supported models

Measured in this release:
- Qwen3.6-35B-A3B (`qwen35moe`): UD-Q4_K_XL, MXFP4_MOE, with or without the MTP head;
- Qwen3-Next-80B-A3B (`qwen3next`): Q4_K_M, host experts through mmap;
- gpt-oss-20b and gpt-oss-120b (`gpt-oss`): MXFP4;
- Bonsai-27B (`qwen35`, dense): Q1_0.

Tiered expert formats (GPU kernel and CPU twin): Q4_K, Q5_K, Q6_K, IQ4_NL, MXFP4, NVFP4, Q1_0. Other
architectures work as they do in upstream mistral.rs v0.9.4, without the titan paths.

## Build and run

Requirements:
- Linux x86-64 with AVX2;
- an NVIDIA GPU with compute capability 12.0 (Blackwell; the PTX is built for sm_120), and the driver;
- the CUDA 13 libraries and headers (no nvcc is invoked);
- Rust stable;
- [Git LFS](https://git-lfs.com): the largest embedded PTX file (`oxide-kernels/mistralrs-quant-c/mistralrs_quant_c.ptx`,
  98 MB) is stored in LFS. Run `git lfs install` before cloning; a clone without LFS gets a pointer file there and the
  build fails.

```bash
git lfs install
git clone https://github.com/optim-enterprises-bv/titan-engine && cd titan-engine/mistral.rs
CUDA_HOME=/usr/local/cuda CUDA_COMPUTE_CAP=120 CUDARC_CUDA_VERSION=13030 \
TITAN_OXIDE_DIR=$PWD/../oxide-kernels CARGO_BUILD_JOBS=2 \
  cargo build --release -p mistralrs-cli --features oxide
```

`TITAN_OXIDE_DIR` makes candle drop its nvcc-built kernels too. The build is memory-hungry: on a 30 GB machine, keep
`CARGO_BUILD_JOBS=2`. <!-- DRAFT NOTE: verify this exact recipe from a clean clone of the staging tree (AUDIT.md B5)
before publishing; the local builds used a CUDA_HOME with only include/ and lib64/ to prove nvcc is not called. -->

Serve Qwen3.6-35B-A3B the way the benchmark does (this is `deploy/titan-mistral.service`):

```bash
TITAN_TIERED=1 TITAN_TIERED_GPU_FRACTION=auto TITAN_TIERED_PROFILE=m4/profile-q35.txt \
TITAN_MTP=2 TITAN_PFS=1 TITAN_DOORBELL=1 TITAN_CPU_ONEPASS=1 TITAN_CUDA_GRAPHS=1 \
  ./target/release/mistralrs serve -p 1234 --paged-attn off --max-seq-len 65536 \
  --format gguf -m <dir> -f Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
```

It exposes an OpenAI-compatible API on port 1234. The per-model configurations used in the benchmark are in
`release/bench/campaign.json`, and the exact command lines are in `release/bench/results.md`.

To reproduce the benchmark, see `release/bench/campaign.sh`. It needs the models, llama.cpp at `acecd56`, and the GPU
to itself for about several hours.

## Limitations

- **Single user.** One request at a time. `--paged-attn off` is required on the titan paths. There is no continuous
  batching of concurrent requests.
- **One GPU class.** It was developed and tuned on a 16 GB Blackwell laptop GPU at a 150 W cap, and the PTX targets
  sm_120. Other cards are untested.
- **A 30 GB RAM machine.** The 80B and 120B models do not fit in RAM. Their host experts are read through the page
  cache, so their speed depends on the NVMe and on what else is using memory.
- **Model coverage.** The titan paths cover the architectures listed above. Anything else runs on upstream
  mistral.rs code.
- **Bit-identity is a gate, not a guarantee.** The release gates check byte-identical greedy output against frozen
  references for the 35B model, with and without MTP. Other models were checked against llama.cpp at the first-token
  and top-k level, not byte for byte.

## Credits

- [mistral.rs](https://github.com/EricLBuehler/mistral.rs) (MIT): the engine this is a fork of.
- [candle](https://github.com/huggingface/candle) (Apache-2.0 OR MIT): the tensor library, forked at `a9667ca`.
- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT): the quantisation formats, and the kernels that
  `oxide-kernels` ports (MMQ, mmvq, MoE grids, the flash-decode structure). It is also the baseline we measure
  against.
- [cuda-oxide](https://github.com/NVlabs/cuda-oxide) (Apache-2.0, NVIDIA): the Rust-to-PTX compiler for our kernels.
- [Qwen](https://github.com/QwenLM) (Qwen3.6, Qwen3-Next), [OpenAI](https://github.com/openai/gpt-oss) (gpt-oss) and
  PrismML (Bonsai): the models. Their weights are not part of this repository.
- [Strata](https://github.com/Niko1221/Strata): the mapped-memory doorbell, the single CPU pass per expert miss and
  the prefill staging ring were first described there. These are ideas, not code: Strata publishes no licence, and
  none of its code, text or identifiers were used. The implementations here were written independently.

Third-party notices are in `NOTICE.md`, and upstream bases in `UPSTREAM.md`.

## Licence

The code written for titan-engine is MIT-licensed (see `LICENSE`). The forked trees keep their upstream licences:
mistral.rs is MIT, candle is Apache-2.0 OR MIT, and code ported from llama.cpp/ggml is MIT with its copyright notice
kept. See `NOTICE.md`.
