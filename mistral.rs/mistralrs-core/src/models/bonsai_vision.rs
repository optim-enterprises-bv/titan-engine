//! Load the Bonsai / Offrail vision tower (a Qwen3-VL ViT) from a llama.cpp clip `mmproj` GGUF.
//! Bridges the clip tensor names/layout onto titan's `Qwen3VLVisionModel` so the existing tower
//! runs unchanged. Deepstack is off for Bonsai 2 (0/27 layers), so this is a plain tower -> merger.
//! See models/IMPL-bonsai-vision.md for the verified spec.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_quant::{GgufArchive, ShardedSafeTensors};

use crate::layers::Activation;
use crate::vision_models::qwen3_vl::config::VisionConfig;
use crate::vision_models::qwen3_vl::vision::Qwen3VLVisionModel;

const DEPTH: usize = 27;

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
}
