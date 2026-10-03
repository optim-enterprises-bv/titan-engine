//! titan: text-only Gemma 4 GGUF config, synthesized from GGUF metadata or taken from config.json minus its towers.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file::Value as GgufValue;
use serde_json::{json, Map, Value as JsonValue};

use crate::vision_models::gemma4::config::Gemma4Config;

const ARCHITECTURE: &str = "gemma4";
const MULTIMODAL_ARCHITECTURE: &str = "Gemma4ForConditionalGeneration";
const DEFAULT_GLOBAL_ROPE_THETA: f64 = 1_000_000.0;
const DEFAULT_LOCAL_ROPE_THETA: f64 = 10_000.0;
/// `rope_freqs` entries at or above this are llama.cpp's "do not rotate" marker (1e30).
const ROPE_FREQ_MASKED: f32 = 1e20;

/// Shapes in archive order (row-major: the GGUF `ne` reversed), keyed by GGUF tensor name.
pub(crate) type TensorShapes = HashMap<String, Vec<usize>>;

pub(crate) fn prepare_gemma4_text_config(
    external: Option<&str>,
    metadata: &HashMap<String, GgufValue>,
    shapes: &TensorShapes,
    rope_freqs: Option<&[f32]>,
) -> Result<String> {
    anyhow::ensure!(
        metadata_string(metadata, "general.architecture")?.eq_ignore_ascii_case(ARCHITECTURE),
        "Gemma 4 config preparation requires `gemma4` GGUF architecture"
    );
    let value = match external {
        Some(config) => text_only_external_config(config, tokenizer_vocab_size(metadata)?)?,
        None => synthesize_config(metadata, shapes, rope_freqs)?,
    };
    let config = serde_json::to_string(&value)?;
    let parsed: Gemma4Config = serde_json::from_str(&config)
        .context("Gemma 4 text configuration is incompatible with the native loader")?;
    anyhow::ensure!(
        parsed.vision_config.is_none() && parsed.audio_config.is_none(),
        "text-only Gemma 4 must not carry vision or audio towers"
    );
    let text = &parsed.text_config;
    anyhow::ensure!(
        text.num_hidden_layers == required_usize(metadata, "gemma4.block_count")?
            && text.hidden_size == required_usize(metadata, "gemma4.embedding_length")?
            && text.vocab_size == tokenizer_vocab_size(metadata)?
            && text.layer_types.len() == text.num_hidden_layers,
        "Gemma 4 config does not match the GGUF model (layers {}, hidden {}, vocab {}, layer_types {})",
        text.num_hidden_layers,
        text.hidden_size,
        text.vocab_size,
        text.layer_types.len()
    );
    Ok(config)
}

/// Keep the external config's text tower; the projector is not loaded on this route.
fn text_only_external_config(config: &str, vocab_size: usize) -> Result<JsonValue> {
    let mut value: JsonValue =
        serde_json::from_str(config).context("External Gemma 4 model config is not valid JSON")?;
    let object = value
        .as_object_mut()
        .context("External Gemma 4 model config must be a JSON object")?;
    object.remove("vision_config");
    object.remove("audio_config");
    object.insert("quantization_config".to_string(), JsonValue::Null);
    let text = if object.contains_key("text_config") {
        match object.get_mut("text_config") {
            Some(JsonValue::Object(text)) => text,
            other => bail!("Gemma 4 `text_config` must be a JSON object, got {other:?}"),
        }
    } else {
        object
    };
    text.insert("vocab_size".to_string(), JsonValue::from(vocab_size));
    text.insert("quantization_config".to_string(), JsonValue::Null);
    Ok(value)
}

fn synthesize_config(
    metadata: &HashMap<String, GgufValue>,
    shapes: &TensorShapes,
    rope_freqs: Option<&[f32]>,
) -> Result<JsonValue> {
    let n_layers = required_usize(metadata, "gemma4.block_count")?;
    anyhow::ensure!(n_layers > 0, "Gemma 4 GGUF has zero layers");
    let hidden_size = required_usize(metadata, "gemma4.embedding_length")?;
    let intermediate_size = uniform(
        "feed_forward_length",
        &per_layer_usize(metadata, "gemma4.feed_forward_length", n_layers)?,
    )?;
    let num_attention_heads = uniform(
        "attention.head_count",
        &per_layer_usize(metadata, "gemma4.attention.head_count", n_layers)?,
    )?;
    let kv_heads = per_layer_usize(metadata, "gemma4.attention.head_count_kv", n_layers)?;
    let sliding = sliding_layers(metadata, n_layers)?;
    let full_layers = (0..n_layers).filter(|&l| !sliding[l]).collect::<Vec<_>>();
    let swa_layers = (0..n_layers).filter(|&l| sliding[l]).collect::<Vec<_>>();
    anyhow::ensure!(
        !swa_layers.is_empty() && !full_layers.is_empty(),
        "Gemma 4 GGUF needs both sliding and full attention layers"
    );
    let num_key_value_heads = uniform(
        "head_count_kv on sliding layers",
        &swa_layers.iter().map(|&l| kv_heads[l]).collect::<Vec<_>>(),
    )?;
    let num_global_key_value_heads = uniform(
        "head_count_kv on full layers",
        &full_layers.iter().map(|&l| kv_heads[l]).collect::<Vec<_>>(),
    )?;

    let head_dim = required_usize(metadata, "gemma4.attention.key_length_swa")?;
    let value_dim_swa = required_usize(metadata, "gemma4.attention.value_length_swa")?;
    let global_head_dim = required_usize(metadata, "gemma4.attention.key_length")?;
    let value_dim = required_usize(metadata, "gemma4.attention.value_length")?;
    anyhow::ensure!(
        head_dim == value_dim_swa && global_head_dim == value_dim,
        "Gemma 4 native loader requires equal key and value head dimensions \
         (sliding {head_dim}/{value_dim_swa}, full {global_head_dim}/{value_dim})"
    );

    // K = V on full layers: those layers ship no attn_v and project V with attn_k.
    let has_v = |l: usize| shapes.contains_key(&format!("blk.{l}.attn_v.weight"));
    anyhow::ensure!(
        swa_layers.iter().all(|&l| has_v(l)),
        "Gemma 4 GGUF sliding layers must all have `attn_v`"
    );
    let attention_k_eq_v = match full_layers.iter().filter(|&&l| has_v(l)).count() {
        0 => true,
        n if n == full_layers.len() => false,
        _ => bail!("Gemma 4 GGUF has `attn_v` on some full-attention layers but not others"),
    };
    // Without K = V the text tower gives full layers the sliding KV head count.
    anyhow::ensure!(
        attention_k_eq_v || num_global_key_value_heads == num_key_value_heads,
        "Gemma 4 GGUF full layers have attn_v and {num_global_key_value_heads} KV heads, but \
         sliding layers have {num_key_value_heads}; provide the original config.json"
    );
    for &l in &full_layers {
        let k = shape(shapes, &format!("blk.{l}.attn_k.weight"))?;
        anyhow::ensure!(
            k[0] == num_global_key_value_heads * global_head_dim,
            "Gemma 4 `blk.{l}.attn_k.weight` has {} rows, expected {} x {}",
            k[0],
            num_global_key_value_heads,
            global_head_dim
        );
    }

    let rope_dims = required_usize(metadata, "gemma4.rope.dimension_count")?;
    let rope_dims_swa = optional_usize(metadata, "gemma4.rope.dimension_count_swa")?
        .unwrap_or(head_dim);
    anyhow::ensure!(
        rope_dims == global_head_dim && rope_dims_swa == head_dim,
        "Gemma 4 GGUF RoPE dimensions ({rope_dims}/{rope_dims_swa}) must equal the head \
         dimensions ({global_head_dim}/{head_dim}); partial rotation is encoded in `rope_freqs`"
    );
    let partial_rotary_factor = partial_rotary_factor(rope_freqs, global_head_dim)?;

    let sliding_window = required_usize(metadata, "gemma4.attention.sliding_window")?;
    let layer_types = sliding
        .iter()
        .map(|&s| {
            if s {
                "sliding_attention"
            } else {
                "full_attention"
            }
        })
        .collect::<Vec<_>>();
    let vocab_size = tokenizer_vocab_size(metadata)?;
    let tie_word_embeddings = !shapes.contains_key("output.weight");

    let enable_moe_block = shapes.contains_key("blk.0.ffn_gate_inp.weight");
    let mut moe = Map::new();
    if enable_moe_block {
        let num_experts = required_usize(metadata, "gemma4.expert_count")?;
        let top_k = required_usize(metadata, "gemma4.expert_used_count")?;
        let expert_ff = required_usize(metadata, "gemma4.expert_feed_forward_length")?;
        for l in 0..n_layers {
            let fused = format!("blk.{l}.ffn_gate_up_exps.weight");
            if let Some(s) = shapes.get(&fused) {
                anyhow::ensure!(
                    s.len() == 3 && s[0] == num_experts && s[1] == 2 * expert_ff,
                    "Gemma 4 `{fused}` has shape {s:?}, expected [{num_experts}, {}, hidden]",
                    2 * expert_ff
                );
            }
            let router = shape(shapes, &format!("blk.{l}.ffn_gate_inp.weight"))?;
            anyhow::ensure!(
                router[0] == num_experts,
                "Gemma 4 `blk.{l}.ffn_gate_inp.weight` routes to {} experts, expected {num_experts}",
                router[0]
            );
        }
        moe.insert("num_experts".into(), json!(num_experts));
        moe.insert("top_k_experts".into(), json!(top_k));
        moe.insert("expert_intermediate_size".into(), json!(expert_ff));
    }

    let ple = optional_usize(metadata, "gemma4.embedding_length_per_layer_input")?.unwrap_or(0);
    anyhow::ensure!(
        ple == 0,
        "Gemma 4 GGUF uses per-layer input embeddings ({ple}); provide the original config.json"
    );

    let mut text = json!({
        "attention_bias": false,
        "attention_k_eq_v": attention_k_eq_v,
        "enable_moe_block": enable_moe_block,
        "final_logit_softcapping": optional_f64(metadata, "gemma4.final_logit_softcapping")?,
        "global_head_dim": global_head_dim,
        "head_dim": head_dim,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": hidden_size,
        "hidden_size_per_layer_input": null,
        "intermediate_size": intermediate_size,
        "layer_types": layer_types,
        "max_position_embeddings": required_usize(metadata, "gemma4.context_length")?,
        "num_attention_heads": num_attention_heads,
        "num_global_key_value_heads": num_global_key_value_heads,
        "num_hidden_layers": n_layers,
        "num_key_value_heads": num_key_value_heads,
        "num_kv_shared_layers":
            optional_usize(metadata, "gemma4.attention.shared_kv_layers")?.unwrap_or(0),
        "quantization_config": null,
        "rms_norm_eps": required_f64(metadata, "gemma4.attention.layer_norm_rms_epsilon")?,
        // the global (full-attention) RoPE reads the top-level theta
        "rope_theta": optional_f64(metadata, "gemma4.rope.freq_base")?
            .unwrap_or(DEFAULT_GLOBAL_ROPE_THETA),
        "rope_parameters": {
            "full_attention": {
                "rope_type": "proportional",
                "partial_rotary_factor": partial_rotary_factor,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": optional_f64(metadata, "gemma4.rope.freq_base_swa")?
                    .unwrap_or(DEFAULT_LOCAL_ROPE_THETA),
            },
        },
        "sliding_window": sliding_window,
        "tie_word_embeddings": tie_word_embeddings,
        "vocab_size": vocab_size,
    });
    text.as_object_mut()
        .expect("text config is an object")
        .extend(moe);
    Ok(json!({
        "architectures": [MULTIMODAL_ARCHITECTURE],
        "model_type": ARCHITECTURE,
        "text_config": text,
    }))
}

// llama.cpp encodes proportional RoPE as rope_freqs = 1.0 on rotated pairs and 1e30 (frequency ~0) on the rest
fn partial_rotary_factor(rope_freqs: Option<&[f32]>, head_dim: usize) -> Result<f64> {
    let Some(freqs) = rope_freqs else {
        return Ok(1.0);
    };
    anyhow::ensure!(
        freqs.len() * 2 == head_dim,
        "Gemma 4 `rope_freqs` has {} entries, expected {}",
        freqs.len(),
        head_dim / 2
    );
    let rotated = freqs.iter().take_while(|&&f| f == 1.0).count();
    anyhow::ensure!(
        freqs[rotated..].iter().all(|&f| f >= ROPE_FREQ_MASKED),
        "Gemma 4 `rope_freqs` must be 1.0 for a prefix of pairs and masked (>= 1e20) after it"
    );
    Ok((2 * rotated) as f64 / head_dim as f64)
}

// sliding_window_pattern is a bool array (true = sliding) or a period n (n - 1 sliding layers, then one full)
fn sliding_layers(metadata: &HashMap<String, GgufValue>, n_layers: usize) -> Result<Vec<bool>> {
    let key = "gemma4.attention.sliding_window_pattern";
    match metadata.get(key) {
        Some(GgufValue::Array(values)) => {
            anyhow::ensure!(
                values.len() == n_layers,
                "GGUF `{key}` has {} entries for {n_layers} layers",
                values.len()
            );
            values
                .iter()
                .map(|value| match value {
                    GgufValue::Bool(b) => Ok(*b),
                    other => value_usize(other)
                        .map(|v| v != 0)
                        .with_context(|| format!("GGUF `{key}` entries must be booleans")),
                })
                .collect()
        }
        Some(value) => {
            let period = value_usize(value)
                .filter(|&p| p > 0)
                .with_context(|| format!("GGUF `{key}` must be a positive integer or array"))?;
            Ok((0..n_layers).map(|l| (l + 1) % period != 0).collect())
        }
        None => bail!("GGUF metadata is missing `{key}`"),
    }
}

fn per_layer_usize(
    metadata: &HashMap<String, GgufValue>,
    key: &str,
    n_layers: usize,
) -> Result<Vec<usize>> {
    match metadata.get(key) {
        Some(GgufValue::Array(values)) => {
            anyhow::ensure!(
                values.len() == n_layers,
                "GGUF `{key}` has {} entries for {n_layers} layers",
                values.len()
            );
            values
                .iter()
                .map(|v| value_usize(v).with_context(|| format!("GGUF `{key}` entries must be integers")))
                .collect()
        }
        Some(_) => Ok(vec![required_usize(metadata, key)?; n_layers]),
        None => bail!("GGUF metadata is missing `{key}`"),
    }
}

fn uniform(label: &str, values: &[usize]) -> Result<usize> {
    let first = *values
        .first()
        .with_context(|| format!("Gemma 4 `{label}` is empty"))?;
    anyhow::ensure!(
        values.iter().all(|&v| v == first),
        "Gemma 4 `{label}` varies across layers ({values:?}); provide the original config.json"
    );
    Ok(first)
}

fn shape<'a>(shapes: &'a TensorShapes, name: &str) -> Result<&'a [usize]> {
    shapes
        .get(name)
        .map(Vec::as_slice)
        .with_context(|| format!("Gemma 4 GGUF is missing tensor `{name}`"))
}

fn tokenizer_vocab_size(metadata: &HashMap<String, GgufValue>) -> Result<usize> {
    match metadata.get("tokenizer.ggml.tokens") {
        Some(GgufValue::Array(tokens)) if !tokens.is_empty() => Ok(tokens.len()),
        Some(_) => bail!("GGUF metadata `tokenizer.ggml.tokens` must be a nonempty array"),
        None => bail!("GGUF metadata is missing `tokenizer.ggml.tokens`"),
    }
}

fn required_usize(metadata: &HashMap<String, GgufValue>, key: &str) -> Result<usize> {
    optional_usize(metadata, key)?.with_context(|| format!("GGUF metadata is missing `{key}`"))
}

fn optional_usize(metadata: &HashMap<String, GgufValue>, key: &str) -> Result<Option<usize>> {
    metadata
        .get(key)
        .map(|value| {
            value_usize(value)
                .with_context(|| format!("GGUF metadata `{key}` must be a nonnegative integer"))
        })
        .transpose()
}

fn value_usize(value: &GgufValue) -> Option<usize> {
    let value = match value {
        GgufValue::U8(value) => *value as u64,
        GgufValue::U16(value) => *value as u64,
        GgufValue::U32(value) => *value as u64,
        GgufValue::U64(value) => *value,
        GgufValue::I8(value) if *value >= 0 => *value as u64,
        GgufValue::I16(value) if *value >= 0 => *value as u64,
        GgufValue::I32(value) if *value >= 0 => *value as u64,
        GgufValue::I64(value) if *value >= 0 => *value as u64,
        _ => return None,
    };
    usize::try_from(value).ok()
}

fn required_f64(metadata: &HashMap<String, GgufValue>, key: &str) -> Result<f64> {
    optional_f64(metadata, key)?.with_context(|| format!("GGUF metadata is missing `{key}`"))
}

fn optional_f64(metadata: &HashMap<String, GgufValue>, key: &str) -> Result<Option<f64>> {
    metadata
        .get(key)
        .map(|value| match value {
            GgufValue::F32(value) => Ok(*value as f64),
            GgufValue::F64(value) => Ok(*value),
            _ => bail!("GGUF metadata `{key}` must be a floating-point number"),
        })
        .transpose()
}

fn metadata_string<'a>(metadata: &'a HashMap<String, GgufValue>, key: &str) -> Result<&'a str> {
    match metadata.get(key) {
        Some(GgufValue::String(value)) => Ok(value),
        Some(_) => bail!("GGUF metadata `{key}` must be a string"),
        None => bail!("GGUF metadata is missing `{key}`"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LAYERS: usize = 12;

    /// REDCELL-shaped metadata (gemma4 26B-A4B), shrunk to 12 layers.
    fn metadata(moe: bool) -> HashMap<String, GgufValue> {
        let sliding = (0..LAYERS).map(|l| (l + 1) % 6 != 0).collect::<Vec<_>>();
        let mut m: HashMap<String, GgufValue> = HashMap::from([
            ("general.architecture".into(), GgufValue::String("gemma4".into())),
            ("gemma4.block_count".into(), GgufValue::U32(LAYERS as u32)),
            ("gemma4.context_length".into(), GgufValue::U32(262144)),
            ("gemma4.embedding_length".into(), GgufValue::U32(2816)),
            ("gemma4.feed_forward_length".into(), GgufValue::U32(2112)),
            ("gemma4.attention.head_count".into(), GgufValue::U32(16)),
            (
                "gemma4.attention.head_count_kv".into(),
                GgufValue::Array(
                    sliding.iter().map(|&s| GgufValue::U32(if s { 8 } else { 2 })).collect(),
                ),
            ),
            ("gemma4.rope.freq_base".into(), GgufValue::F32(1e6)),
            ("gemma4.rope.freq_base_swa".into(), GgufValue::F32(1e4)),
            ("gemma4.attention.layer_norm_rms_epsilon".into(), GgufValue::F32(1e-6)),
            ("gemma4.attention.key_length".into(), GgufValue::U32(512)),
            ("gemma4.attention.value_length".into(), GgufValue::U32(512)),
            ("gemma4.attention.key_length_swa".into(), GgufValue::U32(256)),
            ("gemma4.attention.value_length_swa".into(), GgufValue::U32(256)),
            ("gemma4.rope.dimension_count".into(), GgufValue::U32(512)),
            ("gemma4.rope.dimension_count_swa".into(), GgufValue::U32(256)),
            ("gemma4.final_logit_softcapping".into(), GgufValue::F32(30.0)),
            ("gemma4.attention.sliding_window".into(), GgufValue::U32(1024)),
            ("gemma4.attention.shared_kv_layers".into(), GgufValue::U32(0)),
            ("gemma4.embedding_length_per_layer_input".into(), GgufValue::U32(0)),
            (
                "gemma4.attention.sliding_window_pattern".into(),
                GgufValue::Array(sliding.iter().map(|&s| GgufValue::Bool(s)).collect()),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                GgufValue::Array((0..16).map(|i| GgufValue::String(format!("t{i}"))).collect()),
            ),
        ]);
        if moe {
            m.insert("gemma4.expert_count".into(), GgufValue::U32(128));
            m.insert("gemma4.expert_used_count".into(), GgufValue::U32(8));
            m.insert("gemma4.expert_feed_forward_length".into(), GgufValue::U32(704));
        }
        m
    }

    fn shapes(moe: bool) -> TensorShapes {
        let mut s = TensorShapes::from([
            ("token_embd.weight".into(), vec![16, 2816]),
            ("rope_freqs.weight".into(), vec![256]),
        ]);
        for l in 0..LAYERS {
            let full = (l + 1) % 6 == 0;
            s.insert(format!("blk.{l}.attn_k.weight"), vec![if full { 1024 } else { 2048 }, 2816]);
            if !full {
                s.insert(format!("blk.{l}.attn_v.weight"), vec![2048, 2816]);
            }
            if moe {
                s.insert(format!("blk.{l}.ffn_gate_inp.weight"), vec![128, 2816]);
                s.insert(format!("blk.{l}.ffn_gate_up_exps.weight"), vec![128, 1408, 2816]);
            }
        }
        s
    }

    fn rope_freqs() -> Vec<f32> {
        let mut f = vec![1.0f32; 64];
        f.extend(std::iter::repeat_n(1e30f32, 192));
        f
    }

    #[test]
    fn synthesizes_redcell_shaped_moe_config() {
        let config =
            prepare_gemma4_text_config(None, &metadata(true), &shapes(true), Some(&rope_freqs()))
                .unwrap();
        let v: JsonValue = serde_json::from_str(&config).unwrap();
        let t = &v["text_config"];
        assert_eq!(v["architectures"], json!(["Gemma4ForConditionalGeneration"]));
        assert_eq!(t["num_key_value_heads"], 8);
        assert_eq!(t["num_global_key_value_heads"], 2);
        assert_eq!(t["head_dim"], 256);
        assert_eq!(t["global_head_dim"], 512);
        assert_eq!(t["attention_k_eq_v"], true);
        assert_eq!(t["rope_theta"], 1e6);
        assert_eq!(t["rope_parameters"]["sliding_attention"]["rope_theta"], 1e4);
        assert_eq!(t["rope_parameters"]["full_attention"]["partial_rotary_factor"], 0.25);
        assert_eq!(t["layer_types"][5], "full_attention");
        assert_eq!(t["layer_types"][4], "sliding_attention");
        assert_eq!(t["enable_moe_block"], true);
        assert_eq!(t["num_experts"], 128);
        assert_eq!(t["top_k_experts"], 8);
        assert_eq!(t["expert_intermediate_size"], 704);
        assert_eq!(t["tie_word_embeddings"], true);
        assert_eq!(t["vocab_size"], 16);
        let parsed: Gemma4Config = serde_json::from_str(&config).unwrap();
        assert_eq!(parsed.text_config.partial_rotary_factor(), 0.25);
        assert_eq!(parsed.text_config.rope_local_base_freq(), 1e4);
        assert_eq!(parsed.text_config.expert_intermediate_size(), Some(704));
        assert!(parsed.vision_config.is_none());
    }

    #[test]
    fn synthesizes_dense_config_from_integer_pattern() {
        let mut m = metadata(false);
        m.insert("gemma4.attention.sliding_window_pattern".into(), GgufValue::U32(6));
        let config =
            prepare_gemma4_text_config(None, &m, &shapes(false), Some(&rope_freqs())).unwrap();
        let v: JsonValue = serde_json::from_str(&config).unwrap();
        assert_eq!(v["text_config"]["enable_moe_block"], false);
        assert_eq!(v["text_config"]["layer_types"][11], "full_attention");
        assert!(v["text_config"].get("num_experts").is_none());
    }

    #[test]
    fn rejects_mixed_k_eq_v() {
        let mut s = shapes(false);
        s.insert("blk.5.attn_v.weight".into(), vec![1024, 2816]);
        let err = prepare_gemma4_text_config(None, &metadata(false), &s, Some(&rope_freqs()))
            .unwrap_err();
        assert!(err.to_string().contains("some full-attention layers"));
    }

    #[test]
    fn rejects_non_prefix_rope_freqs() {
        let mut f = rope_freqs();
        f[100] = 1.0;
        let err = prepare_gemma4_text_config(None, &metadata(false), &shapes(false), Some(&f))
            .unwrap_err();
        assert!(err.to_string().contains("rope_freqs"));
    }

    #[test]
    fn external_config_drops_towers() {
        let external = json!({
            "architectures": ["Gemma4ForConditionalGeneration"],
            "text_config": {
                "hidden_size": 2816, "intermediate_size": 2112, "num_hidden_layers": LAYERS,
                "sliding_window": 1024,
                "layer_types": (0..LAYERS).map(|l| if (l + 1) % 6 == 0 { "full_attention" } else { "sliding_attention" }).collect::<Vec<_>>(),
            },
            "vision_config": {"hidden_size": 768},
        });
        let config = prepare_gemma4_text_config(
            Some(&external.to_string()),
            &metadata(false),
            &shapes(false),
            None,
        )
        .unwrap();
        let v: JsonValue = serde_json::from_str(&config).unwrap();
        assert!(v.get("vision_config").is_none());
        assert_eq!(v["text_config"]["vocab_size"], 16);
    }
}
