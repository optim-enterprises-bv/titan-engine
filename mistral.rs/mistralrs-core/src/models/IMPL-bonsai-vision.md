# Bonsai / Offrail vision (Qwen3-VL ViT on the GGUF qwen35 PTQ1_0 text path)

Native multimodal for the Bonsai 2 / Offrail GGUF models (arch `qwen35`, text served by
`quantized_qwen35_moe.rs`). The projector is a standard Qwen3-VL ViT in llama.cpp clip-GGUF form.
Goal: titan serves image+text for `offrail-27b` without PrismML's bundled llama.cpp.

## Verified vision spec (from Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf)
- `clip.projector_type = qwen3vl_merger`, `clip.vision.spatial_merge_size = 2`, `projection_dim = 5120`.
- ViT: depth 27, hidden 1152, ffn 4304, heads 16, patch 16, image_size 768, in_chans 3,
  temporal_patch_size 2, layer_norm_eps 1e-6, gelu, image_mean/std = 0.5/0.5.
- `is_deepstack_layers`: ALL FALSE (0/27). No deepstack injection needed - plain tower -> merger.
- Learned absolute pos embed `v.position_embd.weight [1152, 2304]` (2304 = (768/16)^2 = 48x48).

## GGUF clip tensor -> titan Qwen3VLVisionModel name map
- `v.patch_embd.weight [16,16,3,1152]` + `v.patch_embd.weight.1` (temporal frame 2) + `.bias`
  -> `patch_embed.proj` (conv3d; two temporal slices concatenated on in_chans*t).
- `v.position_embd.weight` -> `pos_embed` (learned, interpolate to the image's patch grid).
- `v.blk.N.ln1/ln2` -> `blocks.N.norm1/norm2`; `attn_qkv` -> `blocks.N.attn.qkv`;
  `attn_out` -> `blocks.N.attn.proj`; `ffn_up` -> `blocks.N.mlp.linear_fc1`;
  `ffn_down` -> `blocks.N.mlp.linear_fc2` (all with bias).
- `v.post_ln` -> `merger.norm` (pre-merge LN on 1152).
- merger: reshape 2x2 spatial merge -> 4608; `mm.0 [4608,4608]+bias` -> `merger.linear_fc1`;
  gelu; `mm.2 [4608,5120]+bias` -> `merger.linear_fc2`. Output 5120 = text hidden.
- deepstack_merger_list: none (deepstack off).

## Build pieces
1. GGUF clip -> VarBuilder bridge: dequantize the mmproj tensors (Q8_0/F16/F32) to BF16 and expose
   them under the Qwen3VLVisionModel names above, so `Qwen3VLVisionModel::new` loads unchanged.
   Reuse titan's GGUF dequant (mistralrs_quant gguf archive -> QTensor::dequantize).
2. Image preprocessing: reuse `vision_models::qwen3_vl::inputs_processor` / `Qwen3VLProcessor`
   (smart-resize to a multiple of patch*merge, normalize 0.5/0.5, patchify, produce grid_thw).
3. Text splice (the invasive part, do last, behind a flag so text-only is unchanged):
   add an input-embeds entry to `quantized_qwen35_moe` forward: embed token ids, then
   `masked_scatter` the merger output into the image-placeholder token positions; mrope/3D position
   ids for the vision span (Qwen3-VL mrope_section). Must thread through the forward variants that a
   vision request actually uses (eager/chunked prefill); graph-captured decode runs post-image so can
   stay text-only for the generated tokens.
4. Resolver family: in `gguf/multimodal_vision_registry.rs` add `family_from_names` arm for
   projector `qwen3vl_merger` + architecture `qwen35` -> new `MultimodalLoaderType::Qwen35Vl`.
5. Loader: new multimodal loader that builds the vision tower (piece 1) + the quantized text model
   and runs piece 3. Chat template: Qwen3-VL image tokens (`<|vision_start|><|image_pad|>...`).

## Validation
Reference answer: PrismML bundled binary (`bonsai2-27b/.../serve-16gb-mtp-vision.sh`) on
`bonsai2-27b/vision_check.png`. titan output must be comparably correct (not byte-identical).

## Status
Spec verified against the GGUF. Vision tower + bridge + processor are non-invasive and come first;
the text-path input-embeds splice is the risky multi-session piece and is gated so the text path
(production) is untouched when no image is present.
