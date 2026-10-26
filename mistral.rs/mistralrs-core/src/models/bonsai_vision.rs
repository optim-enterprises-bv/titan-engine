//! Load the Bonsai / Offrail vision tower (a Qwen3-VL ViT) from a llama.cpp clip `mmproj` GGUF.
//! Bridges the clip tensor names/layout onto titan's `Qwen3VLVisionModel` so the existing tower
//! runs unchanged. Deepstack is off for Bonsai 2 (0/27 layers), so this is a plain tower -> merger.
//! See models/IMPL-bonsai-vision.md for the verified spec.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_quant::{GgufArchive, ShardedSafeTensors};

use crate::layers::Activation;
use crate::vision_models::preprocessor_config::PreProcessorConfig;
use crate::vision_models::qwen3_vl::config::VisionConfig;
use crate::vision_models::qwen3_vl::inputs_processor::bonsai_image_to_patches;
use crate::vision_models::qwen3_vl::vision::Qwen3VLVisionModel;

const DEPTH: usize = 27;

/// Preprocess one image into the `(patches, grid_thw)` the Bonsai tower consumes, using the shared
/// Qwen3-VL preprocessing (smart-resize, 0.5/0.5 normalize, patchify) so the patch ordering matches.
pub(crate) fn preprocess_bonsai_image(
    image: image::DynamicImage,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let cfg = PreProcessorConfig {
        patch_size: Some(16),
        merge_size: Some(2),
        temporal_patch_size: Some(2),
        image_mean: Some([0.5, 0.5, 0.5]),
        image_std: Some([0.5, 0.5, 0.5]),
        ..Default::default()
    };
    bonsai_image_to_patches(image, &cfg, device)
}

fn deq(arch: &GgufArchive, name: &str, dev: &Device) -> Result<Tensor> {
    let q = arch.load_qtensor(name, dev)?;
    q.dequantize(dev)?.to_dtype(DType::BF16)
}

/// Build the Qwen3-VL vision tower from a Bonsai clip `mmproj` GGUF. Weights are dequantized to BF16
/// and exposed under the names `Qwen3VLVisionModel::new` expects.
pub(crate) fn load_bonsai_vision_tower(
    mmproj_path: &Path,
    device: &Device,
) -> Result<Qwen3VLVisionModel> {
    let arch = GgufArchive::open_file(mmproj_path)?;
    let mut t: HashMap<String, Tensor> = HashMap::new();

    // Patch embed: two temporal frames [1152,3,16,16] -> conv3d weight [1152,3,2,16,16].
    let p0 = deq(&arch, "v.patch_embd.weight", device)?;
    let p1 = deq(&arch, "v.patch_embd.weight.1", device)?;
    t.insert(
        "patch_embed.proj.weight".to_string(),
        Tensor::stack(&[&p0, &p1], 2)?,
    );
    t.insert(
        "patch_embed.proj.bias".to_string(),
        deq(&arch, "v.patch_embd.bias", device)?,
    );

    // Learned absolute position embedding [num_position_embeddings, hidden] = [2304, 1152].
    t.insert(
        "pos_embed.weight".to_string(),
        deq(&arch, "v.position_embd.weight", device)?,
    );

    for i in 0..DEPTH {
        let pairs = [
            (format!("blocks.{i}.norm1.weight"), format!("v.blk.{i}.ln1.weight")),
            (format!("blocks.{i}.norm1.bias"), format!("v.blk.{i}.ln1.bias")),
            (format!("blocks.{i}.norm2.weight"), format!("v.blk.{i}.ln2.weight")),
            (format!("blocks.{i}.norm2.bias"), format!("v.blk.{i}.ln2.bias")),
            (format!("blocks.{i}.attn.qkv.weight"), format!("v.blk.{i}.attn_qkv.weight")),
            (format!("blocks.{i}.attn.qkv.bias"), format!("v.blk.{i}.attn_qkv.bias")),
            (format!("blocks.{i}.attn.proj.weight"), format!("v.blk.{i}.attn_out.weight")),
            (format!("blocks.{i}.attn.proj.bias"), format!("v.blk.{i}.attn_out.bias")),
            (format!("blocks.{i}.mlp.linear_fc1.weight"), format!("v.blk.{i}.ffn_up.weight")),
            (format!("blocks.{i}.mlp.linear_fc1.bias"), format!("v.blk.{i}.ffn_up.bias")),
            (format!("blocks.{i}.mlp.linear_fc2.weight"), format!("v.blk.{i}.ffn_down.weight")),
            (format!("blocks.{i}.mlp.linear_fc2.bias"), format!("v.blk.{i}.ffn_down.bias")),
        ];
        for (dst, src) in pairs {
            t.insert(dst, deq(&arch, &src, device)?);
        }
    }

    // Merger: v.post_ln -> norm, mm.0 -> linear_fc1 (gelu), mm.2 -> linear_fc2 (-> out_hidden 5120).
    for (dst, src) in [
        ("merger.norm.weight", "v.post_ln.weight"),
        ("merger.norm.bias", "v.post_ln.bias"),
        ("merger.linear_fc1.weight", "mm.0.weight"),
        ("merger.linear_fc1.bias", "mm.0.bias"),
        ("merger.linear_fc2.weight", "mm.2.weight"),
        ("merger.linear_fc2.bias", "mm.2.bias"),
    ] {
        t.insert(dst.to_string(), deq(&arch, src, device)?);
    }

    let vb = ShardedSafeTensors::wrap(t, DType::BF16, device.clone());
    let cfg = VisionConfig {
        depth: DEPTH,
        hidden_size: 1152,
        out_hidden_size: 5120,
        hidden_act: Activation::Gelu,
        intermediate_size: 4304,
        num_heads: 16,
        in_chans: 3,
        patch_size: 16,
        spatial_merge_size: 2,
        temporal_patch_size: 2,
        num_position_embeddings: 2304,
        deepstack_visual_indexes: vec![],
    };
    Qwen3VLVisionModel::new(&cfg, vb)
}

/// Qwen3-VL 3D (t,h,w) rope positions for a single-image sequence, `[3, n]` i64. Text spans use
/// equal t==h==w (which reduces to the backbone's partial-NEOX rope); the image placeholder block
/// gets grid (t,h,w) positions offset by the preceding text length. `grid` is post-merge (t, gh, gw).
/// Mirrors qwen3_vl/mod.rs get_rope_index for one image, one sequence, all tokens valid.
pub(crate) fn vision_mrope_positions(
    ids: &[u32],
    image_token_id: u32,
    grid: (usize, usize, usize),
    device: &Device,
) -> Result<Tensor> {
    let n = ids.len();
    let (gt, gh, gw) = grid;
    let (mut p0, mut p1, mut p2) = (vec![0i64; n], vec![0i64; n], vec![0i64; n]);
    match ids.iter().position(|&t| t == image_token_id) {
        None => {
            for i in 0..n {
                p0[i] = i as i64;
                p1[i] = i as i64;
                p2[i] = i as i64;
            }
        }
        Some(start) => {
            let count = gt * gh * gw;
            if start + count > n {
                candle_core::bail!("image placeholder run {count} exceeds sequence at {start}/{n}");
            }
            for i in 0..start {
                p0[i] = i as i64;
                p1[i] = i as i64;
                p2[i] = i as i64;
            }
            let base = start as i64;
            let mut idx = start;
            for t in 0..gt {
                for h in 0..gh {
                    for w in 0..gw {
                        p0[idx] = base + t as i64;
                        p1[idx] = base + h as i64;
                        p2[idx] = base + w as i64;
                        idx += 1;
                    }
                }
            }
            let next = base + gt.max(gh).max(gw) as i64;
            for (off, i) in (idx..n).enumerate() {
                let v = next + off as i64;
                p0[i] = v;
                p1[i] = v;
                p2[i] = v;
            }
        }
    }
    let mut flat = p0;
    flat.extend(p1);
    flat.extend(p2);
    Tensor::from_vec(flat, (3, n), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Runs the real Bonsai mmproj tower on a dummy patch grid. Requires the model on disk.
    #[test]
    #[ignore = "needs the Bonsai mmproj GGUF on disk"]
    fn loads_and_runs_bonsai_tower() -> Result<()> {
        let path =
            Path::new("/home/dingo/models/bonsai2-27b/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf");
        let dev = Device::Cpu;
        let tower = load_bonsai_vision_tower(path, &dev)?;
        // grid 1x4x4 = 16 patches; each patch is in_chans*temporal*patch*patch = 3*2*16*16 = 1536.
        let xs = Tensor::randn(0f32, 1f32, (16, 1536), &dev)?.to_dtype(DType::BF16)?;
        let grid = Tensor::from_vec(vec![1u32, 4, 4], (1, 3), &dev)?;
        let (embeds, _deepstack) = tower.forward(&xs, &grid)?;
        // 16 patches / (2*2 merge) = 4 merged tokens, out_hidden 5120.
        assert_eq!(embeds.dims(), &[4, 5120]);
        Ok(())
    }

    #[test]
    fn mrope_positions_text_and_image() -> Result<()> {
        let dev = Device::Cpu;
        // text-only: every position is [i,i,i]
        let text = vision_mrope_positions(&[5, 6, 7, 8], 999, (1, 2, 2), &dev)?;
        assert_eq!(text.dims(), &[3, 4]);
        let tv = text.to_vec2::<i64>()?;
        assert_eq!(tv[0], vec![0, 1, 2, 3]);
        assert_eq!(tv[1], vec![0, 1, 2, 3]);
        assert_eq!(tv[2], vec![0, 1, 2, 3]);

        // 2 text, 4 image (grid 1x2x2), 2 text
        let ids = [5u32, 5, 42, 42, 42, 42, 6, 6];
        let p = vision_mrope_positions(&ids, 42, (1, 2, 2), &dev)?;
        let v = p.to_vec2::<i64>()?;
        // text before: 0,1 ; image base=2: t all 2; h = 2,2,3,3 ; w = 2,3,2,3 ; text after from 4
        assert_eq!(v[0], vec![0, 1, 2, 2, 2, 2, 4, 5]); // t
        assert_eq!(v[1], vec![0, 1, 2, 2, 3, 3, 4, 5]); // h
        assert_eq!(v[2], vec![0, 1, 2, 3, 2, 3, 4, 5]); // w
        Ok(())
    }

    // Real image -> processor -> tower, end to end on CPU. Requires the Bonsai files on disk.
    #[test]
    #[ignore = "needs the Bonsai mmproj + vision_check.png on disk"]
    fn bonsai_image_through_tower() -> Result<()> {
        let dev = Device::Cpu;
        let img = image::open("/home/dingo/models/bonsai2-27b/vision_check.png")
            .map_err(|e| candle_core::Error::msg(format!("open image: {e}")))?;
        let (patches, grid) = preprocess_bonsai_image(img, &dev)?;
        let patches = patches.to_dtype(DType::BF16)?;
        let g = grid.to_vec2::<u32>()?;
        let expect = (g[0][0] * g[0][1] * g[0][2]) as usize / 4; // spatial merge 2x2

        let path =
            Path::new("/home/dingo/models/bonsai2-27b/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf");
        let tower = load_bonsai_vision_tower(path, &dev)?;
        let (embeds, _deepstack) = tower.forward(&patches, &grid)?;
        println!(
            "bonsai vision: grid {:?} -> patches {:?} -> embeds {:?}",
            g[0],
            patches.dims(),
            embeds.dims()
        );
        assert_eq!(embeds.dim(0)?, expect);
        assert_eq!(embeds.dim(1)?, 5120);
        assert!(embeds.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?.iter().all(|v| v.is_finite()));
        Ok(())
    }
}
