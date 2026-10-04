#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Spark2.5: dense GQA with per-head sigmoid output gates and 3:1 sliding/full attention layers.

use std::{collections::HashMap, sync::Arc};

use candle_core::{DType, Device, Module, Result, Tensor, D};
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder,
};
use serde::{Deserialize, Serialize};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::{AttentionMask, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{embedding, Activation, CausalMasker, Mlp, RmsNorm, RotaryEmbedding, Sdpa},
    layers_masker::{CausalMaskConfig, PastKvLenCache},
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, ModelForwardContext, NormalCache, NormalCacheType,
        NormalLoadingMetadata, NormalModel,
    },
    serde_default_fn,
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

serde_default_fn!(bool, default_true, true);
serde_default_fn!(f64, default_partial_rotary_factor, 1.0);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    SlidingAttention,
    FullAttention,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GateActivation {
    #[default]
    Sigmoid,
    Silu,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeLayerParameters {
    pub rope_theta: f64,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeParameters {
    pub full_attention: RopeLayerParameters,
    pub sliding_attention: RopeLayerParameters,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub sliding_window: usize,
    pub layer_types: Vec<LayerType>,
    pub rope_parameters: RopeParameters,
    pub hidden_act: Activation,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default = "default_true")]
    pub headwise_attn_output_gate: bool,
    #[serde(default)]
    pub gate_attn_act_mode: GateActivation,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    pub quantization_config: Option<QuantizedConfig>,
}

impl Config {
    fn is_sliding(&self, layer_idx: usize) -> bool {
        self.layer_types[layer_idx] == LayerType::SlidingAttention
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.layer_types.len() != self.num_hidden_layers {
            candle_core::bail!(
                "Spark2.5 config has {} layer_types for {} layers",
                self.layer_types.len(),
                self.num_hidden_layers
            );
        }
        if !self.headwise_attn_output_gate {
            candle_core::bail!("Spark2.5 without the head-wise attention output gate is not supported");
        }
        for params in [
            &self.rope_parameters.full_attention,
            &self.rope_parameters.sliding_attention,
        ] {
            let rot = rotary_dim(self.head_dim, params.partial_rotary_factor);
            if rot == 0 || rot > self.head_dim || !rot.is_multiple_of(2) {
                candle_core::bail!(
                    "Spark2.5 partial_rotary_factor {} gives an invalid rotary dim {rot} for head dim {}",
                    params.partial_rotary_factor,
                    self.head_dim
                );
            }
        }
        Ok(())
    }

    fn cache_types(&self) -> Vec<NormalCacheType> {
        (0..self.num_hidden_layers)
            .map(|layer_idx| {
                if self.is_sliding(layer_idx) {
                    NormalCacheType::SlidingWindow {
                        window: self.sliding_window,
                    }
                } else {
                    NormalCacheType::Normal {
                        max_seq_len: self.max_position_embeddings,
                    }
                }
            })
            .collect()
    }
}

fn rotary_dim(head_dim: usize, partial_rotary_factor: f64) -> usize {
    (head_dim as f64 * partial_rotary_factor).round() as usize
}

// HF rotates the first rot_dim dims NEOX-style with inv_freq = theta^(-2i/rot_dim) and passes the rest through
fn rotary_embedding(
    params: &RopeLayerParameters,
    cfg: &Config,
    device: &Device,
    is_gptx: bool,
    dtype: DType,
) -> Result<RotaryEmbedding> {
    RotaryEmbedding::new_partial(
        params.rope_theta as f32,
        rotary_dim(cfg.head_dim, params.partial_rotary_factor),
        cfg.max_position_embeddings,
        device,
        is_gptx,
        dtype,
    )
}

/// Query rows per block of a sliding layer's banded prompt attention (keys per block: up to this + window - 1).
const SWA_BLOCK: usize = 512;
/// Query heads per KV head the flash-prefill kernel takes.
const FLASH_PREFILL_REP: usize = 8;

// CausalMasker builds both (rows, past + rows) masks on the host, ~0.3 s per 9k-token prompt
fn host_masks() -> bool {
    static ON: mistralrs_quant::titan_cfg::GenCell<bool> = mistralrs_quant::titan_cfg::GenCell::new();
    *ON.get_or_init(|| mistralrs_quant::titan_cfg::var("TITAN_SPARK_HOST_MASK").is_ok_and(|v| v == "1"))
}

/// The causal and the sliding-window additive prompt masks, (rows, past + rows), built on `device`: the same
/// values as `CausalMasker` (key j of query row i is visible iff past + i - window < j <= past + i).
fn prompt_masks(
    rows: usize,
    past: usize,
    window: usize,
    dtype: DType,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let kv = past + rows;
    let pos = Tensor::arange(past as f32, kv as f32, device)?.reshape((rows, 1))?;
    let key = Tensor::arange(0f32, kv as f32, device)?.reshape((1, kv))?;
    let future = key.broadcast_gt(&pos)?;
    let old = (key + window as f64)?.broadcast_le(&pos)?;
    let ninf = Tensor::new(f32::NEG_INFINITY, device)?.to_dtype(dtype)?.broadcast_as((rows, kv))?;
    let zero = Tensor::zeros((), dtype, device)?.broadcast_as((rows, kv))?;
    let full = future.where_cond(&ninf, &zero)?;
    let sliding = future.maximum(&old)?.where_cond(&ninf, &zero)?;
    Ok((full, sliding))
}

fn flash_swa_on() -> bool {
    static ON: mistralrs_quant::titan_cfg::GenCell<bool> = mistralrs_quant::titan_cfg::GenCell::new();
    *ON.get_or_init(|| mistralrs_quant::titan_cfg::var("TITAN_SPARK_FLASH_SWA").map(|v| v != "0").unwrap_or(true))
}

fn banded_on() -> bool {
    static ON: mistralrs_quant::titan_cfg::GenCell<bool> = mistralrs_quant::titan_cfg::GenCell::new();
    *ON.get_or_init(|| mistralrs_quant::titan_cfg::var("TITAN_SPARK_BANDED").map(|v| v != "0").unwrap_or(true))
}

struct Attention {
    q_k_v_proj: Arc<dyn QuantMethod>,
    g_proj: Arc<dyn QuantMethod>,
    out_proj: Arc<dyn QuantMethod>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    gate_act: GateActivation,
    rotary_emb: Arc<RotaryEmbedding>,
    is_sliding: bool,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl Attention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        layer_idx: usize,
        vb: ShardedVarBuilder,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        if comm.world_size() != 1 {
            candle_core::bail!("Spark2.5 fused q_k_v_proj does not support tensor parallelism");
        }
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let q_k_v_proj = ColumnParallelLayer::new(
            cfg.hidden_size,
            q_dim + 2 * kv_dim,
            &cfg.quantization_config,
            cfg.attention_bias,
            comm,
            vb.pp("q_k_v_proj"),
        )?;
        let g_proj = ColumnParallelLayer::new(
            cfg.hidden_size,
            cfg.num_attention_heads,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("g_proj"),
        )?;
        let out_proj = RowParallelLayer::new(
            q_dim,
            cfg.hidden_size,
            &cfg.quantization_config,
            cfg.attention_bias,
            comm,
            vb.pp("out_proj"),
        )?;
        let is_sliding = cfg.is_sliding(layer_idx);
        Ok(Self {
            q_k_v_proj,
            g_proj,
            out_proj,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            gate_act: cfg.gate_attn_act_mode,
            rotary_emb,
            is_sliding,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
                softcap: None,
                softmax_scale: 1.0 / (cfg.head_dim as f32).sqrt(),
                sliding_window: is_sliding.then_some(cfg.sliding_window),
                sinks: None,
            },
        })
    }

    // prompt passes from TITAN_ATTN_FLASH_PREFILL_MIN keys take the flash-prefill kernel (sliding layers with its key
    // window, TITAN_SPARK_FLASH_SWA=0: the banded eager path, TITAN_SPARK_BANDED=0: the whole mask)
    fn attend(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: &AttentionMask,
        flash_params: Option<&FlashParams>,
    ) -> Result<Tensor> {
        let (b, h, rows, d) = q.dims4()?;
        let (kvh, kv_len) = (k.dim(1)?, k.dim(2)?);
        let AttentionMask::Custom(m) = mask else {
            return Sdpa.run_attention(q, k, v, mask, flash_params, &self.sdpa_params);
        };
        let window = self.sdpa_params.sliding_window;
        if rows > 1
            && FLASH_PREFILL_REP.is_multiple_of(h / kvh)
            && crate::attention::flash_prefill_wanted(rows, kv_len)
            && (window.is_none() || flash_swa_on())
        {
            let rep = h / kvh;
            let q5 = q.contiguous()?.reshape((b, kvh, rep, rows, d))?;
            let q8 = Tensor::cat(&vec![&q5; FLASH_PREFILL_REP / rep], 2)?.reshape((
                b,
                kvh * FLASH_PREFILL_REP,
                rows,
                d,
            ))?;
            let p = SdpaParams {
                n_kv_groups: FLASH_PREFILL_REP,
                softcap: None,
                softmax_scale: self.sdpa_params.softmax_scale,
                sliding_window: None,
                sinks: None,
            };
            if crate::attention::flash_prefill_supported(&q8, k, v, &p) {
                return crate::attention::flash_prefill_window(&q8, k, v, &p, window.unwrap_or(0))?
                    .reshape((b, kvh, FLASH_PREFILL_REP, rows, d))?
                    .narrow(2, 0, rep)?
                    .reshape((b, h, rows, d));
            }
        }
        if let Some(w) = window {
            if rows > 1 && m.dims() == [rows, kv_len] && kv_len > w + SWA_BLOCK && banded_on() {
                let past = kv_len - rows;
                let mut out = Vec::with_capacity(rows.div_ceil(SWA_BLOCK));
                for off in (0..rows).step_by(SWA_BLOCK) {
                    let len = SWA_BLOCK.min(rows - off);
                    // row `past + off + i` sees keys in (pos - w, pos]
                    let k0 = (past + off + 1).saturating_sub(w);
                    let kl = past + off + len - k0;
                    let mb = m.narrow(0, off, len)?.narrow(1, k0, kl)?.contiguous()?;
                    out.push(Sdpa.run_attention(
                        &q.narrow(2, off, len)?,
                        &k.narrow(2, k0, kl)?,
                        &v.narrow(2, k0, kl)?,
                        &AttentionMask::Custom(mb),
                        flash_params,
                        &self.sdpa_params,
                    )?);
                }
                return Tensor::cat(&out, 2);
            }
        }
        Sdpa.run_attention(q, k, v, mask, flash_params, &self.sdpa_params)
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: &AttentionMask,
        sliding_attention_mask: &AttentionMask,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;

        let qkv = self.q_k_v_proj.forward(xs)?;
        let q = qkv.narrow(D::Minus1, 0, q_dim)?;
        let k = qkv.narrow(D::Minus1, q_dim, kv_dim)?;
        let v = qkv.narrow(D::Minus1, q_dim + kv_dim, kv_dim)?;
        let (q, k, v) = if q_len != 1 {
            (
                q.reshape((b_sz, q_len, self.num_heads, self.head_dim))?
                    .transpose(1, 2)?,
                k.reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                    .transpose(1, 2)?,
                v.reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                    .transpose(1, 2)?,
            )
        } else {
            (
                q.reshape((b_sz, self.num_heads, q_len, self.head_dim))?,
                k.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?,
                v.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?,
            )
        };

        let rope_positions = ctx
            .text_positions(q.device(), q.dim(2)?)?
            .ok_or_else(|| candle_core::Error::msg("missing RoPE positions"))?;
        let (q, k) = self.rotary_emb.forward(&q, &k, rope_positions)?;
        let metadata = ctx.paged_layer(layer_idx);
        let mask = if self.is_sliding {
            sliding_attention_mask
        } else {
            attention_mask
        };

        // The paged path takes this layer's mask too: a sliding layer gets the sliding-window mask. (It used to
        // get the full causal mask, which PagedAttention applies as given to a whole first prompt chunk, up to
        // max_num_batched_tokens = 4096 rows, so sliding layers attended past their 512-token window.)
        let attn_output = match &self.paged_attn {
            Some(paged_attn) => match metadata {
                Some(((key_cache, value_cache), input_metadata)) => paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    mask,
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    Some(ctx.flash_params()),
                )?,
                None => {
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    assert!(!matches!(mask, AttentionMask::None));
                    paged_attn.forward(
                        &q,
                        &k,
                        &v,
                        mask,
                        None,
                        None,
                        &input_metadata,
                        &self.sdpa_params,
                        Some(ctx.flash_params()),
                    )?
                }
            },
            None => {
                let (k, v) = kv_cache.append(&k, &v)?;
                self.attend(&q, &k, &v, mask, Some(ctx.flash_params()))?
            }
        };
        let attn_output = if !matches!(attention_mask, AttentionMask::None) {
            attn_output.transpose(1, 2)?
        } else {
            attn_output
        };
        let attn_output = attn_output.reshape((b_sz, q_len, self.num_heads, self.head_dim))?;
        crate::attention::titan_dump_rows(layer_idx, &attn_output)?;

        let gate = self.g_proj.forward(xs)?;
        let gate = match self.gate_act {
            GateActivation::Sigmoid => candle_nn::ops::sigmoid(&gate)?,
            GateActivation::Silu => gate.silu()?,
        };
        let gate = gate
            .to_dtype(attn_output.dtype())?
            .reshape((b_sz, q_len, self.num_heads, 1))?;
        let attn_output = attn_output
            .broadcast_mul(&gate)?
            .reshape((b_sz, q_len, ()))?;
        self.out_proj.forward(&attn_output)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            rotary_emb,
            cfg,
            layer_idx,
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
            paged_attn,
            comm,
        )?;
        let mlp = Mlp::new(
            mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq),
            cfg.hidden_size,
            cfg.intermediate_size,
            &cfg.quantization_config,
            cfg.hidden_act,
            comm,
        )?;
        let input_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_attention_layernorm"), false),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: &AttentionMask,
        sliding_attention_mask: &AttentionMask,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(xs)?;
        let attn = self.self_attn.forward(
            &normed,
            attention_mask,
            sliding_attention_mask,
            kv_cache,
            ctx,
            layer_idx,
        )?;
        // plain add + norm: the fused add_rms_norm launcher is nvcc-only in the oxide build
        let residual = (attn + xs)?;
        let normed = self.post_attention_layernorm.forward(&residual)?;
        self.mlp.forward(&normed)? + residual
    }
}

pub struct Model {
    embed_tokens: Arc<dyn QuantMethod>,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    dtype: DType,
    device: Device,
    cache: EitherCache,
    max_seq_len: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    sliding_window: usize,
    cfg: ModelConfigMetadata,
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        cfg.validate()?;
        if let Some(ref quant_cfg) = &cfg.quantization_config {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb)
            );
        }
        let mapper = normal_loading_metadata.mapper;
        let vb_m = vb.pp("model");
        let dtype = vb_m.dtype();
        let embed_tokens = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), normal_loading_metadata.loading_isq),
            &cfg.quantization_config,
        )?;

        let mut ropes = HashMap::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            if ropes.contains_key(&device.location()) {
                continue;
            }
            let full = Arc::new(rotary_embedding(
                &cfg.rope_parameters.full_attention,
                cfg,
                device,
                is_gptx,
                dtype,
            )?);
            let sliding = Arc::new(rotary_embedding(
                &cfg.rope_parameters.sliding_attention,
                cfg,
                device,
                is_gptx,
                dtype,
            )?);
            ropes.insert(device.location(), (full, sliding));
        }

        let vb_l = vb_m.pp("layers");
        let layers: Vec<DecoderLayer> = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer_idx| {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let (full, sliding) = ropes
                .get(&device.location())
                .expect("No RoPE for device location!");
            let rotary_emb = if cfg.is_sliding(layer_idx) {
                sliding.clone()
            } else {
                full.clone()
            };
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(cfg.head_dim, device, None)?)
                }
            };
            let comm = mapper.get_comm_for(layer_idx)?;
            DecoderLayer::new(
                rotary_emb,
                cfg,
                vb_l.pp(layer_idx),
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
                paged_attn,
                &comm,
            )
        })?;
        let norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;
        let lm_head = if cfg.tie_word_embeddings {
            embed_tokens.clone()
        } else {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
            )?
        };
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            dtype,
            device: normal_loading_metadata.real_device,
            cache: EitherCache::Normal(NormalCache::from_types(cfg.cache_types())),
            max_seq_len: cfg.max_position_embeddings,
            sliding_window: cfg.sliding_window,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_attn_heads: cfg.num_attention_heads,
                num_kv_heads: cfg.num_key_value_heads,
                // full-attention layers keep everything; the per-layer window lives in SdpaParams
                sliding_window: None,
                k_head_dim: cfg.head_dim,
                v_head_dim: cfg.head_dim,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            mapper,
        })
    }

    pub fn forward(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        let mut xs = self.embed_tokens.embedding_forward(input_ids, self.dtype)?;
        let cache = &mut self.cache.normal().0;
        let mask_cache = ctx.mask_cache(cache);
        let seq_len = input_ids.dim(1)?;
        let (attention_mask, sliding_attention_mask) =
            if seq_len > 1 && !crate::using_flash_attn() && !host_masks() {
                let past = mask_cache.get_past_kv_len()?;
                let (full, swa) =
                    prompt_masks(seq_len, past, self.sliding_window, xs.dtype(), xs.device())?;
                (AttentionMask::Custom(full), AttentionMask::Custom(swa))
            } else {
                (
                    CausalMasker.make_causal_mask(
                        input_ids,
                        &mask_cache,
                        xs.dtype(),
                        &CausalMaskConfig::default(),
                    )?,
                    CausalMasker.make_causal_mask(
                        input_ids,
                        &mask_cache,
                        xs.dtype(),
                        &CausalMaskConfig {
                            sliding_window: Some(self.sliding_window),
                            ..Default::default()
                        },
                    )?,
                )
            };
        // PagedAttention prompt chunking
        let (attention_mask, sliding_attention_mask) = if ctx.is_first_prompt_chunk() {
            (attention_mask, sliding_attention_mask)
        } else {
            (AttentionMask::None, AttentionMask::None)
        };
        let attention_mask = DeviceMappedMask::new(attention_mask, &*self.mapper)?;
        let sliding_attention_mask = DeviceMappedMask::new(sliding_attention_mask, &*self.mapper)?;
        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            xs = layer.forward(
                &xs,
                &attention_mask.get(xs.device()),
                &sliding_attention_mask.get(xs.device()),
                &mut cache[i],
                ctx,
                i,
            )?;
        }
        let xs = xs.to_device(&self.device)?;
        let xs = xs.apply(&self.norm)?;
        let xs = ctx.logits(&xs)?;
        ctx.lm_head(&*self.lm_head, &xs)
    }
}

impl IsqModel for Model {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let uvb_m = uvb.pp("model");
        uvb_m.pp("embed_tokens").add(&self.embed_tokens);
        uvb_m.pp("norm").add(&self.norm);
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            uvb_l.pp("input_layernorm").add(&layer.input_layernorm);
            uvb_l
                .pp("post_attention_layernorm")
                .add(&layer.post_attention_layernorm);
        }
        uvb.to_safetensors()
    }
}

impl crate::speculative::SpeculativeTargetMixin for Model {}

impl NormalModel for Model {
    fn forward(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        self.forward(input_ids, ctx)
    }
    fn xlora_forward(
        &self,
        _input_ids: &Tensor,
        _input_ids_full: &Tensor,
        _seqlen_offsets: &[usize],
        _seqlen_offsets_full: &[usize],
        _no_kv_cache: bool,
        _non_granular_state: &Option<crate::xlora_models::NonGranularState>,
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _flash_params: &FlashParams,
        _flash_params_full: &FlashParams,
    ) -> Result<Tensor> {
        unimplemented!()
    }
    fn cache(&self) -> &EitherCache {
        &self.cache
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn is_xlora(&self) -> bool {
        false
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
    // titan: every layer has a paged KV cache and plain causal attention (no alibi, sinks or softcap), so the paged
    // prompt admission plans titan's flash-prefill over the gathered KV (paged_attention::plan) instead of the eager
    // gather fallback's whole-prompt score workspace.
    fn model_config(&self) -> std::sync::Arc<dyn crate::paged_attention::ModelConfigLike + Send + Sync> {
        std::sync::Arc::new(
            crate::paged_attention::HybridPagedKvCacheConfig::new(self.cfg.clone(), vec![true; self.cfg.num_layers])
                .with_uniform_prefix_prefill_attention_features(Default::default()),
        )
    }
}

impl AnyMoeBaseModelMixin for Model {}

#[cfg(test)]
mod tests {
    use super::*;

    fn hf_config() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["Spark2_5ForCausalLM"],
            "vocab_size": 131072, "hidden_size": 2560, "intermediate_size": 10240,
            "num_hidden_layers": 4, "num_attention_heads": 16, "num_key_value_heads": 4,
            "head_dim": 256, "max_position_embeddings": 1048576, "rms_norm_eps": 1e-6,
            "sliding_window": 512, "hidden_act": "gelu", "headwise_attn_output_gate": true,
            "gate_attn_act_mode": "sigmoid", "tie_word_embeddings": true,
            "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
            "rope_parameters": {
                "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 5000000},
                "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": 10000}
            }
        })
    }

    #[test]
    fn hf_config_deserializes_with_erf_gelu() {
        let cfg: Config = serde_json::from_value(hf_config()).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.hidden_act, Activation::Gelu);
        assert_eq!(cfg.gate_attn_act_mode, GateActivation::Sigmoid);
        assert_eq!(rotary_dim(cfg.head_dim, cfg.rope_parameters.full_attention.partial_rotary_factor), 64);
        assert!(cfg.is_sliding(2) && !cfg.is_sliding(3));
        let types = cfg.cache_types();
        assert!(matches!(
            &types[3],
            NormalCacheType::Normal { max_seq_len: 1048576 }
        ));
        assert!(matches!(
            &types[0],
            NormalCacheType::SlidingWindow { window: 512 }
        ));
    }

    #[test]
    fn rejects_mismatched_layer_types() {
        let mut value = hf_config();
        value["num_hidden_layers"] = serde_json::json!(5);
        let cfg: Config = serde_json::from_value(value).unwrap();
        assert!(cfg.validate().is_err());
    }
}
