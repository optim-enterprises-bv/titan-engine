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

## Remaining design (fully mapped; all reference impls located)
DONE + validated (non-invasive, committed): vision tower loader (models/bonsai_vision.rs) and
image processor (bonsai_image_to_patches) - image -> [N,5120] embeds proven on vision_check.png.

Backbone is already VL-aware: quantized_qwen35_moe uses IMRoPE and feeds `text_positions` today
(t==h==w). For images, supply 3D (t,h,w) positions; no new rope math needed in the kernel.

Remaining (cross-cutting, multi-session, touches shared prod infra - do gated so text is identical):
1. Shared ModelForwardContext (pipeline/mod.rs): add Option fields `image_embeds: Option<Tensor>`
   and `vision_position_ids: Option<Tensor>` (None = today's text path, byte-identical). Thread
   through its `new()` and construction sites.
2. 3D rope-index: port qwen3_vl/mod.rs:112-221 (image_token_id spans, spatial-merge grid -> t/h/w
   positions) into a shared helper; feed as vision_position_ids when an image is present.
3. Embed splice: in quantized_qwen35_moe the token-embed happens per forward variant
   (lines ~3243, 4206, and 2848/3056/3431/3592/3747/3907/4053). Add one helper
   `embed_with_images(ids, ctx)` = tok_embeddings.forward then scatter ctx.image_embeds into the
   image_token_id positions; call it in the variants a vision request uses (eager/chunked prefill;
   decode is post-image text, unchanged). Gate on ctx.image_embeds.is_some().
4. Vision pipeline + resolver: family arm in gguf/multimodal_vision_registry.rs for projector
   `qwen3vl_merger` + arch `qwen35` -> new MultimodalLoaderType; loader builds the tower
   (load_bonsai_vision_tower) + processor, runs the model with image_embeds in ctx. Chat template:
   Qwen3-VL `<|vision_start|><|image_pad|>...<|vision_end|>`; image_token_id from the GGUF.
5. Cap processor max_pixels ~768^2 to match Bonsai's training (default oversizes: 9072 patches seen).
6. Validate vs the bundled binary (serve-16gb-mtp-vision.sh) on vision_check.png.

## Status update: all compute bricks built + validated (committed)
Done & tested: vision tower loader, image processor, image-embed splice into the qwen35 forward
(embed_with_images + ctx.image_embeds), vision_mrope_positions [3,n], mrope_cos_sin, and the GGUF
constants. All in models/bonsai_vision.rs + the shared ModelForwardContext field. Text path untouched.

### GGUF constants (verified from Offrail-27B-12G.gguf)
image_pad=248056, vision_start=248053, vision_end=248054; rope.dimension_sections=[11,11,10];
rope_dim=64, freq_base=1e7; arch qwen35, heads 24/kv 4, embd 5120, 65 blocks (64 + 1 MTP).

### Remaining = one coupled serve-path block (gguf_titan.rs + friends), validate on GPU:
A. Loader: get the mmproj path from `paths` (ignored today per gguf_titan.rs:181); if present and
   arch==qwen35, load_bonsai_vision_tower + preprocess config, store tower on TitanGGUFPipeline,
   call model.set_image_token_id(Some(BONSAI_IMAGE_PAD_TOKEN)).
B. Inputs: override PreProcessingMixin to return the Qwen3VL image inputs processor so requests'
   images reach the pipeline (ModelInputs must carry pixel_values/grid for the titan path).
C. forward_inputs: when images present, bonsai_image_to_patches -> tower -> embeds; set
   ctx.with_image_embeds(Some(embeds)); build vision_mrope_positions and feed them as the rope
   positions (needs a rope-apply using mrope_cos_sin in the qwen35 attention when vision positions
   are set - isolated to quantized_qwen35_moe, NOT the shared RotaryEmbedding).
D. Chat template: Qwen3-VL `<|vision_start|>` + image_pad*num_merged_tokens + `<|vision_end|>`.
E. Validate: `mistralrs serve ... --mmproj Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` + an image request,
   compare to bonsai2-27b/.../serve-16gb-mtp-vision.sh on vision_check.png.

This block is interdependent and only meaningful once it generates a correct image answer on GPU.
