# IQ3_S (GGML type 21) gates, 2026-10-02

Kernel: oxide-kernels/iq3_s (branch iq3s, 22df5e4): 1085 launches, 15.6 MB, 0 differing bytes vs llama.cpp
acecd56 cubins; CPU oracle (libggml-base dequantize_row_iq3_s) equal; 4/4 mutants detected (out/gate-iq3s.log).
candle CPU dequantize (iquant.rs iq3s_block, extracted verbatim by cpucheck/make.py) == llama.cpp on 4096 blocks
(out/cpucheck.log).

End-to-end, Qwen3.5-9B (qwen35 dense) requantized with llama-quantize IQ3_M (no imatrix needed), llama.cpp vs
mistral.rs (mr-iq3s, --features oxide), 8 prompts, greedy 64 tokens + teacher-forced top-5 at every prefix of
llama.cpp's greedy continuation (e2e.py):
| model | greedy identical | TF top-1 | llama top-1 in mistral top-5 | mean / max abs dlogp |
| IQ3_M, token_embd IQ3_S (embiq3s) | 2/8 | 441/452 (97.6%) | 452/452 | 0.046 / 0.54 |
| IQ3_M, token_embd Q8_0 (iq3m) | 2/8 | 500/509 (98.2%) | 509/509 | 0.054 / 0.78 |
| control: source Q8_0, no IQ3_S (q8ctl) | 6/8 | 510/512 (99.6%) | 512/512 | 0.0095 / 0.099 |

Regression (window2 log): 35B MTP=2 40/40 vs m6/out/h-off.json. MTP off: the q35-prof.json source file
(~/ai/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf, via sync/w094/q35root) no longer exists; on the MTP-dir file this
build is 40/40 identical to the deployed bin/mistralrs-titan-swap and to m6/out/mtpfile-off.json (both 5/40 vs q35-prof).
