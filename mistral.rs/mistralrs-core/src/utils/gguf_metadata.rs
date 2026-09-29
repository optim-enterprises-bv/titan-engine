use akin::akin;
use anyhow::ensure;
use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::DType;
use std::collections::HashMap;
use std::fs;
use tracing::warn;

use crate::attention::ATTENTION_CHUNK_SIZE;
use crate::gguf::Content;
use crate::matformer::MatformerSliceConfig;
use crate::paged_attention::ModelConfigLike;
use crate::pipeline::AutoDeviceMapParams;
use crate::pipeline::DeviceMappedModelLoader;
use crate::GGUFArchitecture;

#[derive(Debug)]
pub struct ContentConfig {
    max_seq_len: usize,
    hidden_size: usize,
    num_attn_heads: usize,
    num_kv_heads: usize,
    num_layers: usize,
    key_length: Option<usize>,
    value_length: Option<usize>,
}

#[allow(clippy::cast_possible_truncation)]
impl<'a, R: std::io::Seek + std::io::Read> From<&Content<'a, R>> for ContentConfig {
    fn from(value: &Content<'a, R>) -> Self {
        let metadata = value.get_metadata();
        let arch = metadata["general.architecture"].to_string().unwrap();
        Self {
            max_seq_len: metadata[&format!("{arch}.context_length")]
                .to_u64()
                .unwrap() as usize,
            hidden_size: metadata[&format!("{arch}.embedding_length")]
                .to_u64()
                .unwrap() as usize,
            num_attn_heads: metadata[&format!("{arch}.attention.head_count")]
                .to_u64()
                .unwrap() as usize,
            num_kv_heads: metadata[&format!("{arch}.attention.head_count_kv")]
                .to_u64()
                .unwrap() as usize,
            num_layers: metadata[&format!("{arch}.block_count")].to_u64().unwrap() as usize
                - crate::models::quantized_qwen35_moe::nextn_predict_layers(metadata, &arch),
            key_length: metadata
                .get(&format!("{arch}.attention.key_length"))
                .map(|x| x.to_u64().unwrap() as usize),
            value_length: metadata
                .get(&format!("{arch}.attention.value_length"))
                .map(|x| x.to_u64().unwrap() as usize),
        }
    }
}

impl ModelConfigLike for ContentConfig {
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn num_attn_heads(&self) -> usize {
        self.num_attn_heads
    }
    fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }
    fn num_layers(&self) -> usize {
        self.num_layers
    }
    fn k_head_dim(&self) -> usize {
        self.key_length
            .unwrap_or(self.hidden_size / self.num_attn_heads)
    }
    fn v_head_dim(&self) -> usize {
        self.value_length
            .unwrap_or(self.hidden_size / self.num_attn_heads)
    }
}

pub struct ContentMetadata<'a> {
    pub path_prefix: &'a str,
    pub metadata: &'a HashMap<String, gguf_file::Value>,
}

impl ContentMetadata<'_> {
    // Retrieve a prop the struct needs by querying the metadata content:
    pub fn get_value<T: TryFromValue>(&self, field_name: &str) -> Result<T, anyhow::Error> {
        let prop_key = format!("{prefix}.{field_name}", prefix = self.path_prefix);
        let value = self.metadata.get(&prop_key).cloned();

        // Unwrap the inner value of the `Value` enum via trait method,
        // otherwise format error with prop key as context:
        value
            .try_value_into()
            .or_else(|e| anyhow::bail!("`{prop_key}` `{e}`"))
    }

    // Retrieve a prop the struct needs by querying the metadata content:
    pub fn get_option_value<T: TryFromValue>(
        &self,
        field_name: &str,
    ) -> Result<Option<T>, anyhow::Error> {
        let prop_key = format!("{prefix}.{field_name}", prefix = self.path_prefix);
        let value = self.metadata.get(&prop_key).cloned();

        // Unwrap the inner value of the `Value` enum via trait method,
        // otherwise format error with prop key as context:
        value
            .map(|v| {
                v.try_value_into()
                    .or_else(|e| anyhow::bail!("`{prop_key}` `{e}`"))
            })
            .map_or(Ok(None), |res| res.map(Some))
    }

    // Fail early - Catch all missing mandatory keys upfront:
    pub fn has_required_keys(&self, fields: &[&str]) -> Result<()> {
        let mut all_props_are_present = true;

        for field_name in fields {
            let prop_key = format!("{prefix}.{field_name}", prefix = self.path_prefix);

            if !self.metadata.contains_key(&prop_key) {
                all_props_are_present = false;
                warn!("Expected GGUF metadata to have key: `{prop_key}`");
            }
        }

        ensure!(all_props_are_present, "Tokenizer is missing required props");
        Ok(())
    }

    // Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#required
    pub fn verify_arch(&self, expected_arch: &str) -> Result<()> {
        let actual_arch: String = self
            .metadata
            .get("general.architecture")
            .cloned()
            .try_value_into()?;

        anyhow::ensure!(
            actual_arch == expected_arch,
            "Expected `{expected_arch}` architecture, got `{actual_arch}`."
        );

        Ok(())
    }

    pub fn verify_arch_any(&self, expected_archs: &[&str]) -> Result<()> {
        let actual_arch: String = self
            .metadata
            .get("general.architecture")
            .cloned()
            .try_value_into()?;

        anyhow::ensure!(
            expected_archs.iter().any(|arch| *arch == actual_arch),
            "Expected one of `{expected_archs:?}` architectures, got `{actual_arch}`."
        );

        Ok(())
    }
}

// These traits below are a workaround for converting candles GGUF `Value` enum type wrapper.
// A better upstream approach would instead be to provide serialize/deserialize support?
pub trait TryFromValue {
    fn try_from_value(value: gguf_file::Value) -> Result<Self, candle_core::Error>
    where
        Self: Sized;
}

// Value wrapped types, each has a different conversion method:
// NOTE: Type conversion methods internally bail with "not a <into type> <input value>"
// https://docs.rs/candle-core/latest/candle_core/quantized/gguf_file/enum.Value.html#variants
akin! {
    let &types = [String, bool, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64];
    let &to_type = [
        value.to_string().cloned(),
        value.to_bool(),
        value.to_f32(),
        value.to_f64(),
        value.to_i8(),
        value.to_i16(),
        value.to_i32(),
        value.to_i64(),
        value.to_u8(),
        value.to_u16(),
        value.to_u32(),
        value.to_u64(),
    ];

    impl TryFromValue for *types {
        fn try_from_value(value: gguf_file::Value) -> Result<Self, candle_core::Error> {
            *to_type.or_else(|_| candle_core::bail!("value is not a `*types`"))
        }
    }
}

// Vec<Value> to Vec<T> from above types:
impl<T: TryFromValue> TryFromValue for Vec<T> {
    fn try_from_value(value_vec: gguf_file::Value) -> Result<Self, candle_core::Error> {
        value_vec
            .to_vec()
            .or_else(|_| candle_core::bail!("value is not a `Vec`"))?
            .clone()
            .into_iter()
            .map(|item| T::try_from_value(item))
            .collect()
    }
}

pub trait TryValueInto<T>: Sized {
    fn try_value_into(self) -> Result<T, candle_core::Error>;
}

impl<T: TryFromValue> TryValueInto<T> for gguf_file::Value {
    fn try_value_into(self) -> Result<T, candle_core::Error> {
        T::try_from_value(self)
    }
}

impl<T: TryFromValue> TryValueInto<T> for Option<gguf_file::Value> {
    fn try_value_into(self) -> Result<T, candle_core::Error> {
        match self {
            Some(value) => value.try_value_into(),
            None => candle_core::bail!("Expected `Option<gguf_file::Value>` to contain a value"),
        }
    }
}

macro_rules! tensor_info_size_in_bytes {
    ($t:expr) => {
        $t.shape.elem_count() / $t.ggml_dtype.block_size() * $t.ggml_dtype.type_size()
    };
    ($t:expr, $ty:expr) => {
        $t.shape.elem_count() * $ty.size_in_bytes()
    };
}

/// Tensors of a qwen35moe nextn (MTP) block besides the decoder-layer ones.
const QWEN35MOE_NEXTN_TENSORS: &[&str] = &[
    "nextn.eh_proj.weight",
    "nextn.enorm.weight",
    "nextn.hnorm.weight",
    "nextn.shared_head_norm.weight",
    "nextn.embed_tokens.weight",
    "nextn.shared_head_head.weight",
];

const QWEN35MOE_LAYER_TENSORS: &[&str] = &[
    "attn_norm.weight",
    "post_attention_norm.weight",
    "attn_qkv.weight",
    "attn_gate.weight",
    "ssm_a",
    "ssm_alpha.weight",
    "ssm_beta.weight",
    "ssm_ba.weight",
    "ssm_conv1d.weight",
    "ssm_dt.bias",
    "ssm_norm.weight",
    "ssm_out.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    "attn_output.weight",
    "ffn_gate_inp.weight",
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
    "ffn_gate_inp_shexp.weight",
    "ffn_gate_shexp.weight",
    "ffn_up_shexp.weight",
    "ffn_down_shexp.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
];

const GPT_OSS_LAYER_TENSORS: &[&str] = &[
    "attn_norm.weight",
    "post_attention_norm.weight",
    "attn_q.weight",
    "attn_q.bias",
    "attn_k.weight",
    "attn_k.bias",
    "attn_v.weight",
    "attn_v.bias",
    "attn_output.weight",
    "attn_output.bias",
    "attn_sinks.weight",
    "ffn_gate_inp.weight",
    "ffn_gate_inp.bias",
    "ffn_gate_exps.weight",
    "ffn_gate_exps.bias",
    "ffn_up_exps.weight",
    "ffn_up_exps.bias",
    "ffn_down_exps.weight",
    "ffn_down_exps.bias",
];

pub struct GgufDeviceMapLoaderInner<'a, 'f> {
    pub model: &'a Content<'f, fs::File>,
    pub arch: GGUFArchitecture,
}

impl GgufDeviceMapLoaderInner<'_, '_> {
    /// qwen35 nextn (MTP) blocks counted in block_count after the trunk layers.
    fn nextn_layers(&self) -> usize {
        match self.arch {
            GGUFArchitecture::Qwen35 | GGUFArchitecture::Qwen35MoE | GGUFArchitecture::Qwen3Next => {
                crate::models::quantized_qwen35_moe::nextn_predict_layers(
                    self.model.get_metadata(),
                    &self.arch.to_string(),
                )
            }
            _ => 0,
        }
    }

    /// Arches sized tensor by tensor (heterogeneous layers or titan tiered experts).
    fn layer_tensors(&self) -> Option<&'static [&'static str]> {
        match self.arch {
            GGUFArchitecture::Qwen35 | GGUFArchitecture::Qwen35MoE | GGUFArchitecture::Qwen3Next => {
                Some(QWEN35MOE_LAYER_TENSORS)
            }
            GGUFArchitecture::GptOss => Some(GPT_OSS_LAYER_TENSORS),
            _ => None,
        }
    }

    /// Device bytes of the qwen35moe MTP block when `TITAN_MTP` loads it: the whole block, its
    /// experts included (they stay stock, fully on the main GPU).
    fn mtp_block_bytes(&self) -> Result<usize> {
        if crate::models::quantized_qwen35_moe::mtp_draft_len() == 0 {
            return Ok(0);
        }
        self.nextn_block_bytes()
    }

    /// Bytes of the file's nextn (MTP) block, loaded or not.
    fn nextn_block_bytes(&self) -> Result<usize> {
        if self.nextn_layers() == 0 {
            return Ok(0);
        }
        let block_count =
            self.model.get_metadata()[&format!("{}.block_count", self.arch)].to_u32()? as usize;
        let il = block_count - self.nextn_layers();
        let mut bytes = 0;
        for t in QWEN35MOE_LAYER_TENSORS.iter().chain(QWEN35MOE_NEXTN_TENSORS) {
            let n = format!("blk.{il}.{t}");
            if self.model.has_tensor(&n) {
                bytes += tensor_info_size_in_bytes!(self.model.tensor_info(&n)?);
            }
        }
        Ok(bytes)
    }
}

impl DeviceMappedModelLoader for GgufDeviceMapLoaderInner<'_, '_> {
    fn mapped_max_act_size_elems(
        &self,
        _config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };
        let num_heads = self.model.get_metadata()[&format!("{}.attention.head_count", self.arch)]
            .to_u32()? as usize;
        Ok(max_batch_size * num_heads * max_seq_len.min(&ATTENTION_CHUNK_SIZE))
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        _config: &str,
        _dtype: DType,
        _weight_pack_factor: usize,
        _quantization: Option<&crate::pipeline::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let size_in_bytes = match self.arch {
            GGUFArchitecture::Llama | GGUFArchitecture::Mistral3 => {
                let token_embd = tensor_info_size_in_bytes!(
                    self.model.tensor_info("token_embd.weight")?,
                    DType::F32
                );
                let output_norm = tensor_info_size_in_bytes!(
                    self.model.tensor_info("output_norm.weight")?,
                    DType::F32
                );
                let output = if !self.model.has_tensor("output.weight") {
                    tensor_info_size_in_bytes!(self.model.tensor_info("token_embd.weight")?)
                } else {
                    tensor_info_size_in_bytes!(self.model.tensor_info("output.weight")?)
                };
                token_embd + output_norm + output
            }
            GGUFArchitecture::Phi2 => {
                let token_embd = tensor_info_size_in_bytes!(
                    self.model.tensor_info("token_embd.weight")?,
                    DType::F32
                );
                let output_norm =
                    tensor_info_size_in_bytes!(
                        self.model.tensor_info("output_norm.weight")?,
                        DType::F32
                    ) + tensor_info_size_in_bytes!(self.model.tensor_info("output_norm.bias")?);
                let output = if !self.model.has_tensor("output.weight") {
                    tensor_info_size_in_bytes!(self.model.tensor_info("token_embd.weight")?)
                } else {
                    tensor_info_size_in_bytes!(self.model.tensor_info("output.weight")?)
                };
                token_embd + output_norm + output
            }
            GGUFArchitecture::Phi3 => {
                let token_embd = tensor_info_size_in_bytes!(
                    self.model.tensor_info("token_embd.weight")?,
                    DType::F32
                );
                let output_norm = tensor_info_size_in_bytes!(
                    self.model.tensor_info("output_norm.weight")?,
                    DType::F32
                );
                let output = if !self.model.has_tensor("output.weight") {
                    tensor_info_size_in_bytes!(self.model.tensor_info("token_embd.weight")?)
                } else {
                    tensor_info_size_in_bytes!(self.model.tensor_info("output.weight")?)
                };
                token_embd + output_norm + output
            }
            GGUFArchitecture::Qwen2 | GGUFArchitecture::Qwen3 | GGUFArchitecture::Qwen3MoE => {
                let token_embd = tensor_info_size_in_bytes!(
                    self.model.tensor_info("token_embd.weight")?,
                    DType::F32
                );
                let output_norm = tensor_info_size_in_bytes!(
                    self.model.tensor_info("output_norm.weight")?,
                    DType::F32
                );
                let output = if !self.model.has_tensor("output.weight") {
                    tensor_info_size_in_bytes!(self.model.tensor_info("token_embd.weight")?)
                } else {
                    tensor_info_size_in_bytes!(self.model.tensor_info("output.weight")?)
                };
                token_embd + output_norm + output
            }
            GGUFArchitecture::Qwen35 | GGUFArchitecture::Qwen35MoE | GGUFArchitecture::Qwen3Next => {
                // token_embd stays quantized in host memory (see quantized_qwen35_moe::QEmbedding)
                let output_norm = tensor_info_size_in_bytes!(
                    self.model.tensor_info("output_norm.weight")?,
                    DType::F32
                );
                let output = if !self.model.has_tensor("output.weight") {
                    tensor_info_size_in_bytes!(self.model.tensor_info("token_embd.weight")?)
                } else {
                    tensor_info_size_in_bytes!(self.model.tensor_info("output.weight")?)
                };
                output_norm + output + self.mtp_block_bytes()?
            }
            GGUFArchitecture::GptOss => {
                // token_embd stays quantized in host memory (quantized_qwen35_moe::QEmbedding)
                let output_norm = tensor_info_size_in_bytes!(
                    self.model.tensor_info("output_norm.weight")?,
                    DType::F32
                );
                output_norm + tensor_info_size_in_bytes!(self.model.tensor_info("output.weight")?)
            }
            GGUFArchitecture::Starcoder2 => {
                let token_embd = tensor_info_size_in_bytes!(
                    self.model.tensor_info("token_embd.weight")?,
                    DType::F32
                );
                let output_norm =
                    tensor_info_size_in_bytes!(
                        self.model.tensor_info("output_norm.weight")?,
                        DType::F32
                    ) + tensor_info_size_in_bytes!(self.model.tensor_info("output_norm.bias")?);
                let output = if !self.model.has_tensor("output.weight") {
                    tensor_info_size_in_bytes!(self.model.tensor_info("token_embd.weight")?)
                } else {
                    tensor_info_size_in_bytes!(self.model.tensor_info("output.weight")?)
                };
                token_embd + output_norm + output
            }
            _ => unimplemented!(),
        };
        Ok(size_in_bytes)
    }
    fn num_layers(&self, _config: &str) -> Result<usize> {
        let block_count =
            self.model.get_metadata()[&format!("{}.block_count", self.arch)].to_u32()? as usize;
        Ok(block_count - self.nextn_layers())
    }
    fn kv_cache_layer_fraction(&self, config: &str) -> Result<f64> {
        if !matches!(self.arch, GGUFArchitecture::Qwen35 | GGUFArchitecture::Qwen35MoE | GGUFArchitecture::Qwen3Next) {
            return Ok(1.0);
        }
        // qwen35 hybrids: only every `full_attention_interval`-th trunk layer holds a KV cache; the
        // Gated DeltaNet layers keep a fixed-size recurrent state. (MTP blocks are not mapped layers.)
        let interval = self
            .model
            .get_metadata()
            .get(&format!("{}.full_attention_interval", self.arch))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(4)
            .max(1) as usize;
        let trunk = self.num_layers(config)?;
        let attn = (0..trunk).filter(|i| (i + 1) % interval == 0).count();
        Ok(attn as f64 / trunk.max(1) as f64)
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        _dtype: DType,
        _weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        if let Some(layer_tensors) = self.layer_tensors() {
            use mistralrs_quant::TieredExperts;
            // dense qwen35 has no experts to tier
            if matches!(self.arch, GGUFArchitecture::Qwen35MoE | GGUFArchitecture::Qwen3Next | GGUFArchitecture::GptOss)
                && TieredExperts::wants_auto_plan()
            {
                // Split every layer tensor into tierable experts vs everything else, then size the
                // expert share from the free VRAM on the main device.
                let (mut tiered, mut other) = (0usize, 0usize);
                for i in 0..self.num_layers(config)? {
                    for t in layer_tensors {
                        let n = format!("blk.{i}.{t}");
                        if !self.model.has_tensor(&n) {
                            continue;
                        }
                        let info = self.model.tensor_info(&n)?;
                        let bytes = tensor_info_size_in_bytes!(info);
                        if n.ends_with("_exps.weight") && TieredExperts::supports(info.ggml_dtype) {
                            tiered += bytes;
                        } else {
                            other += bytes;
                        }
                    }
                }
                // non_mapped_size_in_bytes includes the MTP block when TITAN_MTP loads it (stock experts); the plan
                // counts the block either way, so MTP on and off get the same placement and, with CPU experts that
                // differ from the GPU in rounding, the same output
                other += self.non_mapped_size_in_bytes(config, _dtype, _weight_pack_factor, None, _matformer_config)?;
                if crate::models::quantized_qwen35_moe::mtp_draft_len() == 0 {
                    other += self.nextn_block_bytes()?;
                }
                let dev = candle_core::Device::new_cuda(0)?;
                let free = crate::MemoryUsage.query(&dev)?.available();
                TieredExperts::plan_auto(tiered, other, free);
            }
            // Layers are heterogeneous (GDN vs full attention, mixed expert quants): sum each.
            return (0..self.num_layers(config)?)
                .map(|i| {
                    layer_tensors
                        .iter()
                        .map(|t| format!("blk.{i}.{t}"))
                        .filter(|n| self.model.has_tensor(n))
                        .map(|n| {
                            let info = self.model.tensor_info(&n)?;
                            let bytes = tensor_info_size_in_bytes!(info);
                            // titan tiered experts: only the GPU-resident share counts against the device.
                            Ok(if n.ends_with("_exps.weight") {
                                (bytes as f64 * mistralrs_quant::TieredExperts::gpu_share(info.ggml_dtype)) as usize
                            } else {
                                bytes
                            })
                        })
                        .sum::<Result<usize>>()
                })
                .collect();
        }
        let size_in_bytes = match self.arch {
            GGUFArchitecture::Llama => {
                let attn_norm = tensor_info_size_in_bytes!(
                    self.model.tensor_info("blk.0.attn_norm.weight")?,
                    DType::F32
                );
                let ffn_norm = tensor_info_size_in_bytes!(
                    self.model.tensor_info("blk.0.ffn_norm.weight")?,
                    DType::F32
                );

                let attn_q =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_q.weight")?);
                let attn_k =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_k.weight")?);
                let attn_v =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_v.weight")?);
                let attn_output = tensor_info_size_in_bytes!(self
                    .model
                    .tensor_info("blk.0.attn_output.weight")?);

                // MoE or Mlp
                #[allow(clippy::cast_possible_truncation)]
                let n_expert = self
                    .model
                    .get_metadata()
                    .get("expert_count")
                    .map(|x| x.to_u64().unwrap() as usize)
                    .unwrap_or(0);
                let moe_or_mlp = if n_expert <= 1 {
                    let ffn_gate = tensor_info_size_in_bytes!(self
                        .model
                        .tensor_info("blk.0.ffn_gate.weight")?);
                    let ffn_up = tensor_info_size_in_bytes!(self
                        .model
                        .tensor_info("blk.0.ffn_up.weight")?);
                    let ffn_down = tensor_info_size_in_bytes!(self
                        .model
                        .tensor_info("blk.0.ffn_down.weight")?);
                    ffn_gate + ffn_up + ffn_down
                } else {
                    let mut moe_count = 0;
                    moe_count += tensor_info_size_in_bytes!(self
                        .model
                        .tensor_info("blk.0.ffn_gate_inp.weight")?);
                    match self.model.tensor_info("blk.0.ffn_gate_exps.weight") {
                        Ok(feed_forward_gate_exps) => {
                            moe_count += tensor_info_size_in_bytes!(feed_forward_gate_exps);
                            moe_count += tensor_info_size_in_bytes!(self
                                .model
                                .tensor_info("blk.0.ffn_down_exps.weight")?);
                            moe_count += tensor_info_size_in_bytes!(self
                                .model
                                .tensor_info("blk.0.ffn_up_exps.weight")?);
                        }
                        Err(_) => {
                            for i in 0..n_expert {
                                moe_count += tensor_info_size_in_bytes!(self
                                    .model
                                    .tensor_info(&format!("blk.0.ffn_gate.{i}.weight"),)?);
                                moe_count += tensor_info_size_in_bytes!(self
                                    .model
                                    .tensor_info(&format!("blk.0.ffn_down.{i}.weight"),)?);
                                moe_count += tensor_info_size_in_bytes!(self
                                    .model
                                    .tensor_info(&format!("blk.0.ffn_up.{i}.weight"))?);
                            }
                        }
                    }

                    moe_count
                };
                attn_norm + ffn_norm + attn_q + attn_k + attn_v + attn_output + moe_or_mlp
            }
            GGUFArchitecture::Phi2 => {
                let attn_norm = tensor_info_size_in_bytes!(
                    self.model.tensor_info("blk.0.attn_norm.weight")?,
                    DType::F32
                ) + tensor_info_size_in_bytes!(self
                    .model
                    .tensor_info("blk.0.attn_norm.bias")?);

                let attn_qkv =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_qkv.weight")?);
                let attn_output = tensor_info_size_in_bytes!(self
                    .model
                    .tensor_info("blk.0.attn_output.weight")?);

                let ffn_up =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.ffn_up.weight")?);
                let ffn_down =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.ffn_down.weight")?);

                attn_norm + attn_qkv + attn_output + ffn_up + ffn_down
            }
            GGUFArchitecture::Phi3 => {
                let attn_norm = tensor_info_size_in_bytes!(
                    self.model.tensor_info("blk.0.attn_norm.weight")?,
                    DType::F32
                );
                let ffn_norm = tensor_info_size_in_bytes!(
                    self.model.tensor_info("blk.0.ffn_norm.weight")?,
                    DType::F32
                );

                let attn_qkv =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_qkv.weight")?);
                let attn_output = tensor_info_size_in_bytes!(self
                    .model
                    .tensor_info("blk.0.attn_output.weight")?);

                let ffn_up =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.ffn_up.weight")?);
                let ffn_down =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.ffn_down.weight")?);

                attn_norm + ffn_norm + attn_qkv + attn_output + ffn_up + ffn_down
            }
            GGUFArchitecture::Qwen2 | GGUFArchitecture::Qwen3 | GGUFArchitecture::Qwen3MoE => {
                let attn_norm = tensor_info_size_in_bytes!(
                    self.model.tensor_info("blk.0.attn_norm.weight")?,
                    DType::F32
                );
                let ffn_norm = tensor_info_size_in_bytes!(
                    self.model.tensor_info("blk.0.ffn_norm.weight")?,
                    DType::F32
                );

                let mut attn_q =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_q.weight")?);
                if let GGUFArchitecture::Qwen2 = self.arch {
                    attn_q +=
                        tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_q.bias")?);
                }
                let mut attn_k =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_k.weight")?);
                if let GGUFArchitecture::Qwen2 = self.arch {
                    attn_k +=
                        tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_k.bias")?);
                }

                let mut attn_v =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_v.weight")?);
                if let GGUFArchitecture::Qwen2 = self.arch {
                    attn_v +=
                        tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_v.bias")?);
                }

                let attn_output = tensor_info_size_in_bytes!(self
                    .model
                    .tensor_info("blk.0.attn_output.weight")?);

                let ffn_gate = if let GGUFArchitecture::Qwen3MoE = self.arch {
                    tensor_info_size_in_bytes!(self
                        .model
                        .tensor_info("blk.0.ffn_gate_exps.weight")?)
                } else {
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.ffn_gate.weight")?)
                };

                let ffn_up = if let GGUFArchitecture::Qwen3MoE = self.arch {
                    tensor_info_size_in_bytes!(self
                        .model
                        .tensor_info("blk.0.ffn_up_exps.weight")?)
                } else {
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.ffn_up.weight")?)
                };

                let ffn_down = if let GGUFArchitecture::Qwen3MoE = self.arch {
                    tensor_info_size_in_bytes!(self
                        .model
                        .tensor_info("blk.0.ffn_down_exps.weight")?)
                } else {
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.ffn_down.weight")?)
                };

                attn_norm
                    + ffn_norm
                    + attn_q
                    + attn_k
                    + attn_v
                    + attn_output
                    + ffn_gate
                    + ffn_up
                    + ffn_down
            }
            GGUFArchitecture::Starcoder2 => {
                let attn_norm = tensor_info_size_in_bytes!(
                    self.model.tensor_info("blk.0.attn_norm.weight")?,
                    DType::F32
                ) + tensor_info_size_in_bytes!(self
                    .model
                    .tensor_info("blk.0.attn_norm.bias")?);
                let ffn_norm = tensor_info_size_in_bytes!(
                    self.model.tensor_info("blk.0.ffn_norm.weight")?,
                    DType::F32
                ) + tensor_info_size_in_bytes!(self
                    .model
                    .tensor_info("blk.0.ffn_norm.bias")?);

                let attn_q = tensor_info_size_in_bytes!(self
                    .model
                    .tensor_info("blk.0.attn_q.weight")?)
                    + tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_q.bias")?);
                let attn_k = tensor_info_size_in_bytes!(self
                    .model
                    .tensor_info("blk.0.attn_k.weight")?)
                    + tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_k.bias")?);
                let attn_v = tensor_info_size_in_bytes!(self
                    .model
                    .tensor_info("blk.0.attn_v.weight")?)
                    + tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_v.bias")?);
                let attn_output = tensor_info_size_in_bytes!(self
                    .model
                    .tensor_info("blk.0.attn_output.weight")?)
                    + tensor_info_size_in_bytes!(self
                        .model
                        .tensor_info("blk.0.attn_output.bias")?);

                let ffn_up = tensor_info_size_in_bytes!(self
                    .model
                    .tensor_info("blk.0.ffn_up.weight")?)
                    + tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.ffn_up.bias")?);
                let ffn_down = tensor_info_size_in_bytes!(self
                    .model
                    .tensor_info("blk.0.ffn_down.weight")?)
                    + tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.ffn_down.bias")?);

                attn_norm + ffn_norm + attn_q + attn_k + attn_v + attn_output + ffn_up + ffn_down
            }
            GGUFArchitecture::Mistral3 => {
                let attn_norm = tensor_info_size_in_bytes!(
                    self.model.tensor_info("blk.0.attn_norm.weight")?,
                    DType::F32
                );

                let attn_q =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_q.weight")?);
                let attn_k =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_k.weight")?);
                let attn_v =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_v.weight")?);

                let attn_output = tensor_info_size_in_bytes!(self
                    .model
                    .tensor_info("blk.0.attn_output.weight")?);

                let ffn_norm = tensor_info_size_in_bytes!(
                    self.model.tensor_info("blk.0.ffn_norm.weight")?,
                    DType::F32
                );
                let ffn_up =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.ffn_up.weight")?);
                let ffn_down =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.ffn_down.weight")?);
                let ffn_gate =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.ffn_gate.weight")?);

                attn_norm
                    + attn_q
                    + attn_k
                    + attn_v
                    + attn_output
                    + ffn_norm
                    + ffn_up
                    + ffn_down
                    + ffn_gate
            }

            _ => unimplemented!(),
        };
        Ok(vec![size_in_bytes; self.num_layers(config)?])
    }
    fn model_config(&self, _config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let model_config_metadata: ContentConfig = self.model.into();
        Ok(Box::new(model_config_metadata))
    }
}
