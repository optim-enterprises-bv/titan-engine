# M4 layer-diff: Qwen3.6-35B-A3B, mistral.rs (nvcc-free, tiered auto) vs llama.cpp acecd56 (--n-cpu-moe 18)

Prompt pass of "The capital of France is" (5 tokens); llama-eval-callback l_out-N vs TITAN_LAYER_DUMP.
Both predict " Paris".

- Layers 0-30: max sampled |diff| / max |value| between 0.08% and 0.74%, falling with depth; attention layers
  (3, 7, ..., 27) track exactly like GDN layers: no layer-type-specific error.
- Layers 31-39: relative error 3-16%. The hidden state's magnitude drops ~8x at layer 31 in BOTH engines
  (mean |v| 0.64 -> 0.08), and routing near-ties flip.
- Routing: in 12 of 40 layers one to three (token, expert) choices of llama.cpp's top-8 are absent from
  mistral.rs's top-8, always a lowest-ranked (near-tie) slot; layer 31 has one (token 0, expert 15).
  Order within the top-8 differs more often and does not affect the output.
- Known noise source: mistral.rs runs GDN layers in bf16 on CUDA (upstream choice), llama.cpp in f32.

Verdict: no systematic per-layer error; differences are cross-engine numeric noise plus near-tie routing flips.
Files: m4/layerdiff.py, m4/layerdiff-run.sh, m4/route-run.sh, m4/out/layers-35b.txt, m4/out/route-35b.trace.
