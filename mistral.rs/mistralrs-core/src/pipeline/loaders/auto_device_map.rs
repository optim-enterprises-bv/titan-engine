use std::fmt::{self, Display};

use crate::paged_attention::{
    calculate_cache_config, device_memory_cap, CacheMemoryReservations, MemoryGpuConfig,
    ModelConfigLike, DEFAULT_PAGED_ATTENTION_BLOCK_SIZE,
};
use crate::utils::debug::DeviceRepr;
use crate::{DeviceLayerMapMetadata, DeviceMapMetadata, MemoryUsage, PagedAttentionConfig};
use anyhow::{Context, Result};
use candle_core::{DType, Device};
use itertools::Itertools;
use tracing::{info, warn};

use super::DeviceMappedModelLoader;

fn saturating_memory_sum<const N: usize>(parts: [usize; N]) -> usize {
    parts
        .into_iter()
        .fold(0usize, |total, part| total.saturating_add(part))
}

fn checked_memory_sum<const N: usize>(parts: [usize; N]) -> Option<usize> {
    parts
        .into_iter()
        .try_fold(0usize, |total, part| total.checked_add(part))
}

fn post_load_memory_config(
    requested: MemoryGpuConfig,
    pre_load_budget: MemoryGpuConfig,
    resolve_utilization_after_load: bool,
) -> MemoryGpuConfig {
    match requested {
        MemoryGpuConfig::Utilization(_) if !resolve_utilization_after_load => pre_load_budget,
        _ => requested,
    }
}

#[derive(Clone, Debug)]
pub(crate) enum NonMappedSubModel {
    Vision,
    Audio,
}

impl Display for NonMappedSubModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NonMappedSubModel::Vision => write!(f, "vision"),
            NonMappedSubModel::Audio => write!(f, "audio"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum AutoDeviceMapParams {
    Text {
        max_seq_len: usize,
        max_batch_size: usize,
    },
    Multimodal {
        max_seq_len: usize,
        max_batch_size: usize,
        max_image_shape: (usize, usize),
        max_num_images: usize,
    },
}

impl AutoDeviceMapParams {
    pub fn maybe_promote_to_multimodal(&self) -> Self {
        match *self {
            Self::Text {
                max_seq_len,
                max_batch_size,
            } => Self::Multimodal {
                max_seq_len,
                max_batch_size,
                max_image_shape: (
                    Self::DEFAULT_MAX_IMAGE_LENGTH,
                    Self::DEFAULT_MAX_IMAGE_LENGTH,
                ),
                max_num_images: Self::DEFAULT_MAX_NUM_IMAGES,
            },
            Self::Multimodal {
                max_seq_len,
                max_batch_size,
                max_image_shape,
                max_num_images,
            } => Self::Multimodal {
                max_seq_len,
                max_batch_size,
                max_image_shape,
                max_num_images,
            },
        }
    }

    pub fn max_seq_len(&self) -> usize {
        match self {
            Self::Text { max_seq_len, .. } | Self::Multimodal { max_seq_len, .. } => *max_seq_len,
        }
    }

    pub fn max_batch_size(&self) -> usize {
        match self {
            Self::Text { max_batch_size, .. } | Self::Multimodal { max_batch_size, .. } => {
                *max_batch_size
            }
        }
    }
}

impl Display for AutoDeviceMapParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text {
                max_seq_len,
                max_batch_size,
            } => write!(
                f,
                "text[max_seq_len: {max_seq_len}, max_batch_size: {max_batch_size}]"
            ),
            Self::Multimodal {
                max_seq_len,
                max_batch_size,
                max_image_shape,
                max_num_images,
            } => write!(
                f,
                "multimodal[max_seq_len: {max_seq_len}, max_batch_size: {max_batch_size}, max_image_shape: {max_image_shape:?}, max_num_images: {max_num_images}]"
            ),
        }
    }
}

impl AutoDeviceMapParams {
    // Default max sequence length for memory estimation when not specified
    pub const DEFAULT_MAX_SEQ_LEN: usize = 4 * 1024;
    pub const DEFAULT_MAX_BATCH_SIZE: usize = 1;
    pub const DEFAULT_MAX_NUM_IMAGES: usize = 1;
    pub const DEFAULT_MAX_IMAGE_LENGTH: usize = 1024;

    pub fn default_text() -> Self {
        Self::Text {
            max_seq_len: Self::DEFAULT_MAX_SEQ_LEN,
            max_batch_size: Self::DEFAULT_MAX_BATCH_SIZE,
        }
    }

    pub fn default_multimodal() -> Self {
        Self::Multimodal {
            max_seq_len: Self::DEFAULT_MAX_SEQ_LEN,
            max_batch_size: Self::DEFAULT_MAX_BATCH_SIZE,
            max_num_images: Self::DEFAULT_MAX_NUM_IMAGES,
            max_image_shape: (
                Self::DEFAULT_MAX_IMAGE_LENGTH,
                Self::DEFAULT_MAX_IMAGE_LENGTH,
            ),
        }
    }
}

fn calculate_key_block_shape(
    model_config: &dyn ModelConfigLike,
    dtype: DType,
    block_size: usize,
) -> (usize, usize, usize, usize) {
    let element_size = dtype.size_in_bytes();
    let x = 16 / element_size;
    (
        model_config.num_kv_heads(),
        model_config.k_head_dim() / x,
        block_size,
        x,
    )
}

fn calculate_value_block_shape(
    model_config: &dyn ModelConfigLike,
    block_size: usize,
) -> (usize, usize, usize) {
    (
        model_config.num_kv_heads(),
        model_config.v_head_dim(),
        block_size,
    )
}

/// One chunk of eager attention scores (`attention::ATTENTION_CHUNK_SCORE_ELEMS`, 16 Mi elements) as F32 softmax
/// input and output plus its activation-dtype copy, charged as elements of a 2-byte activation dtype.
const PREFILL_ATTN_CHUNK_ELEMS: usize = 80 << 20;

/// Key count from which eager GQA attention runs grouped with score chunks capped at 16 Mi elements
/// (`attention::GQA_GROUPED_MIN_KV`); below it a chunk is 1024 query rows x every head x every key.
const EAGER_GROUPED_MIN_KV: usize = 4096;

/// Transient device elements (activation dtype) of one prefill pass over a prompt of up to `seq_len` tokens. The
/// non-paged path does not chunk prompts, so a prompt of the full context runs in one forward:
/// - `masks` `seq_len x seq_len` attention masks in the activation dtype, each built from a u8 mask of the same
///   shape (half an element of a 2-byte dtype);
/// - the widest per-token working set of one decoder layer: the MLP's gate, up and their product (`mlp_width`
///   each) or attention's q / k / v (`attn_width` = q + k + v width) with their RoPE'd and transposed copies,
///   plus the residual stream, the norm output and the layer output (`hidden` each);
/// - one chunk of eager attention scores; for models whose prompt attention is eager (`eager_heads` > 0, not the
///   flash-prefill kernels), prompts under 4096 keys take 1024-row chunks over all heads (F16 scores and their F32
///   softmax: 4 elements each), which peaks just under 4096 tokens and can exceed the long-prompt figure.
pub(crate) fn prefill_act_elems(
    seq_len: usize,
    batch: usize,
    hidden: usize,
    mlp_width: usize,
    attn_width: usize,
    masks: usize,
    eager_heads: usize,
) -> usize {
    let per_token = (3 * mlp_width).max(3 * attn_width) + 3 * hidden;
    let at = |s: usize| masks * batch * s * s * 3 / 2 + batch * s * per_token + PREFILL_ATTN_CHUNK_ELEMS;
    let short = seq_len.min(EAGER_GROUPED_MIN_KV - 1);
    let short_scores = eager_heads * batch * short.min(1024) * short * 4;
    at(seq_len).max(at(short) + short_scores)
}

/// Largest activation reserve the map makes on a device: half its usable memory. A prompt whose prefill
/// transients exceed that cannot run on the device in one pass at any mapping that also holds weights and KV;
/// reserving it all would only push every layer to the CPU. Longer prompts fail (CUDA OOM, the request errors,
/// the engine recovers) instead of the whole model running from host memory.
fn activation_reserve(act_bytes: usize, usable: usize) -> usize {
    act_bytes.min(usable / 2)
}

macro_rules! b_to_mb {
    ($x:expr) => {
        $x / (1024 * 1024)
    };
}

#[allow(
    clippy::too_many_arguments,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]
/// Core logic for automatic device mapping
pub fn get_device_layers(
    loader: &dyn DeviceMappedModelLoader,
    config: &str,
    num_layers: usize,
    mut layer_sizes_in_bytes: Vec<usize>,
    non_mapped_size_in_bytes: usize,
    total_model_size_in_bytes: usize,
    devices: &[Device],
    dtype: DType,
    params: &AutoDeviceMapParams,
    paged_attn_config: Option<&mut PagedAttentionConfig>,
) -> Result<DeviceMapMetadata> {
    let mapped_max = loader.mapped_max_act_size_elems(config, params)? * dtype.size_in_bytes();
    let non_mapped_max =
        loader.non_mapped_max_act_size_elems(config, params)? * dtype.size_in_bytes();

    let mut layer_sizes_backup = if paged_attn_config.is_some() {
        Some(layer_sizes_in_bytes.clone())
    } else {
        None
    };

    let mut remaining = total_model_size_in_bytes;
    let max_seq_len = match params {
        AutoDeviceMapParams::Text { max_seq_len, .. }
        | AutoDeviceMapParams::Multimodal { max_seq_len, .. } => *max_seq_len,
    };
    let max_batch_size = match params {
        AutoDeviceMapParams::Text { max_batch_size, .. }
        | AutoDeviceMapParams::Multimodal { max_batch_size, .. } => *max_batch_size,
    };

    let model_cfg = loader.model_config(config)?;
    let has_paged_attn = paged_attn_config.is_some();
    let base_device_memory_reservation_bytes = paged_attn_config
        .as_ref()
        .map_or(0, |config| config.base_device_memory_reservation_bytes);
    let kv_cache_elems = match paged_attn_config {
        Some(cfg) => {
            // The mapping estimate is bounded independently from the post-load memory mode.
            let requested_mem_gpu = cfg.mem_gpu;
            let effective_mem_gpu = match requested_mem_gpu {
                MemoryGpuConfig::MbAmount(user_mb) => {
                    // Clamp user's KV budget to available memory.
                    let primary_dev = &devices[0];
                    let avail_bytes = MemoryUsage.query(primary_dev)?.available();
                    let cap = device_memory_cap(avail_bytes, primary_dev);
                    let act_overhead = non_mapped_max.max(mapped_max);
                    let budget_mb = cap
                        .saturating_sub(act_overhead)
                        .saturating_sub(base_device_memory_reservation_bytes)
                        / (1024 * 1024);
                    MemoryGpuConfig::MbAmount(budget_mb.min(user_mb))
                }
                MemoryGpuConfig::BestEffortMbAmount { target_mb, min_mb } => {
                    let primary_dev = &devices[0];
                    let avail_bytes = MemoryUsage.query(primary_dev)?.available();
                    let cap = device_memory_cap(avail_bytes, primary_dev);
                    let act_overhead = non_mapped_max.max(mapped_max);
                    let budget_mb = cap
                        .saturating_sub(act_overhead)
                        .saturating_sub(base_device_memory_reservation_bytes)
                        / (1024 * 1024);
                    MemoryGpuConfig::BestEffortMbAmount {
                        target_mb: budget_mb.min(target_mb),
                        min_mb,
                    }
                }
                MemoryGpuConfig::Utilization(f) => {
                    // Prevent overallocation when total_memory > available_memory
                    // (e.g., unified memory systems, other GPU processes using VRAM).
                    // Cap the KV budget so model + activations + KV fits within
                    // the device capacity derived from *available* memory.
                    let primary_dev = &devices[0];
                    let avail_bytes = MemoryUsage.query(primary_dev)?.available();
                    let cap = device_memory_cap(avail_bytes, primary_dev);
                    let act_overhead = non_mapped_max.max(mapped_max);
                    let occupied = saturating_memory_sum([
                        remaining,
                        act_overhead,
                        base_device_memory_reservation_bytes,
                    ]);
                    let budget_mb =
                        ((cap as f64 * f as f64) as usize).saturating_sub(occupied) / (1024 * 1024);
                    MemoryGpuConfig::MbAmount(budget_mb)
                }
                // ContextSize passes through to calculate_cache_config.
                other => other,
            };
            info!(
                "Reserving {} MB on the primary device and {} MB on mapped devices for activations (predicted).",
                b_to_mb!(non_mapped_max.max(mapped_max)),
                b_to_mb!(mapped_max),
            );
            cfg.reserve_activation_memory(non_mapped_max.max(mapped_max), mapped_max);
            if base_device_memory_reservation_bytes > 0 {
                info!(
                    "Reserving {} MB on the primary device for post-load model components.",
                    base_device_memory_reservation_bytes.div_ceil(1024 * 1024)
                );
            }
            // Re-resolve utilization after recurrent serving state is allocated.
            cfg.mem_gpu = post_load_memory_config(
                requested_mem_gpu,
                effective_mem_gpu,
                cfg.resolve_memory_utilization_after_load,
            );

            let cache = calculate_cache_config(
                effective_mem_gpu,
                CacheMemoryReservations::default(),
                Some(cfg.block_size.unwrap_or(DEFAULT_PAGED_ATTENTION_BLOCK_SIZE)),
                dtype,
                cfg.cache_type,
                &*model_cfg,
                &devices[0],
                &devices.iter().map(|d| Some(d.clone())).collect::<Vec<_>>(),
                true,
                Some(total_model_size_in_bytes),
                Some(max_seq_len * max_batch_size),
            )?;
            let key_shape = calculate_key_block_shape(&*model_cfg, dtype, cache.block_size);
            let key_sz =
                cache.num_gpu_blocks * key_shape.0 * key_shape.1 * key_shape.2 * key_shape.3;
            let val_shape = calculate_value_block_shape(&*model_cfg, cache.block_size);
            let val_sz = cache.num_gpu_blocks * val_shape.0 * val_shape.1 * val_shape.2;
            key_sz + val_sz
        }
        None => {
            let key_shape = [
                max_batch_size,
                model_cfg.num_kv_heads(),
                max_seq_len,
                model_cfg.k_head_dim(),
            ];
            let val_shape = [
                max_batch_size,
                model_cfg.num_kv_heads(),
                max_seq_len,
                model_cfg.v_head_dim(),
            ];
            let per_layer = key_shape.iter().product::<usize>() + val_shape.iter().product::<usize>();
            (per_layer as f64 * loader.kv_cache_layer_fraction(config)?).ceil() as usize
        }
    };
    // Per paged layer; hybrid models leave the recurrent/linear layers out of the cache entirely. The non-paged
    // cache is held in the loader's KV dtype (paged blocks are sized in `dtype` above).
    let kv_dtype = if has_paged_attn {
        dtype
    } else {
        loader.kv_cache_dtype(config, dtype)?
    };
    let kv_cache_bytes = kv_cache_elems * kv_dtype.size_in_bytes();
    let kv_bytes_for_layer = |idx: usize| {
        if model_cfg.layer_has_paged_kv_cache(idx) {
            kv_cache_bytes
        } else {
            0
        }
    };
    // Paged layers past the mapped stack (an MTP head) are charged to the non-mapped device.
    let extra_kv_bytes = (num_layers..model_cfg.num_layers())
        .map(kv_bytes_for_layer)
        .sum::<usize>();

    // prepare available memory per device, CPU fallback last (unless unified memory)
    let has_unified_memory = devices.iter().any(crate::utils::normal::is_integrated_gpu);

    let mut avail = Vec::new();
    for dev in devices {
        let a = MemoryUsage.query(dev)?.available();
        avail.push((a, dev.clone()));
    }
    // On unified memory systems (iGPUs), GPU and CPU share the same physical RAM.
    // Don't add CPU as a fallback device since it would double-count memory.
    if !has_unified_memory {
        let a = MemoryUsage.query(&Device::Cpu)?.available();
        avail.push((a, Device::Cpu));
    }

    // Non-paged: cap the activation reserve (paged attention sized its KV budget from the uncapped value above).
    let (mapped_max, non_mapped_max) = if has_paged_attn {
        (mapped_max, non_mapped_max)
    } else {
        let usable = avail.first().map_or(usize::MAX, |(a, d)| device_memory_cap(*a, d));
        let act = non_mapped_max.max(mapped_max);
        if activation_reserve(act, usable) < act {
            warn!(
                "Prefill activations of a {max_seq_len}-token prompt (~{} MB) exceed half the primary device's usable memory; reserving {} MB, so prompts that long will not fit on the device.",
                b_to_mb!(act),
                b_to_mb!(activation_reserve(act, usable)),
            );
        }
        (
            activation_reserve(mapped_max, usable),
            activation_reserve(non_mapped_max, usable),
        )
    };
    info!(
        "Automatic device map estimate: weights {} MiB in {} layers ({}-{} MiB each) + {} MiB not mapped; KV cache {} MiB per KV layer ({:?}, {} of {} layers) = {} MiB; activations {} MiB on the primary device, {} MiB on others; {} MiB post-load reservation; primary device {} MiB available, {} MiB usable.",
        b_to_mb!(layer_sizes_in_bytes.iter().sum::<usize>()),
        layer_sizes_in_bytes.len(),
        b_to_mb!(layer_sizes_in_bytes.iter().copied().min().unwrap_or(0)),
        b_to_mb!(layer_sizes_in_bytes.iter().copied().max().unwrap_or(0)),
        b_to_mb!(non_mapped_size_in_bytes),
        b_to_mb!(kv_cache_bytes),
        kv_dtype,
        (0..num_layers).filter(|&i| model_cfg.layer_has_paged_kv_cache(i)).count(),
        num_layers,
        b_to_mb!((0..num_layers).map(kv_bytes_for_layer).sum::<usize>() + extra_kv_bytes),
        b_to_mb!(non_mapped_max.max(mapped_max)),
        b_to_mb!(mapped_max),
        b_to_mb!(base_device_memory_reservation_bytes),
        b_to_mb!(avail.first().map_or(0, |(a, _)| *a)),
        b_to_mb!(avail.first().map_or(0, |(a, d)| device_memory_cap(*a, d))),
    );

    avail.reverse();
    layer_sizes_in_bytes.reverse();

    let mut mappings = Vec::new();
    info!("Using automatic device mapping parameters: {params}.");
    if let Some(subs) = loader.non_mapped_sub_models_for_config(config)? {
        let (_, last) = avail.last().unwrap();
        info!(
            "The following sub-models will not be device mapped and will be loaded on {}: {}",
            last.device_pretty_repr(),
            subs.iter().map(|x| x.to_string()).join(", ")
        );
    }

    let mut ordinal = 0;
    let mut layer = 0;
    let avail_copy = avail.clone();
    let mut includes_cpu = false;
    while remaining > 0 && !avail.is_empty() {
        let (avail_bytes, dev) = avail
            .pop()
            .context("No more devices to map to. The model does not fit on this system.")?;

        // For GPU/accelerators: keep a small dynamic safety reserve to avoid OOMs
        let cap = device_memory_cap(avail_bytes, &dev);
        if ordinal == 0
            && checked_memory_sum([
                non_mapped_max.max(mapped_max),
                non_mapped_size_in_bytes,
                base_device_memory_reservation_bytes,
            ])
            .is_none_or(|required| required > cap)
        {
            anyhow::bail!(
                "Primary device {} cannot fit its fixed model components, activations, and post-load reservation within {} MB of usable capacity.",
                dev.device_pretty_repr(),
                b_to_mb!(cap),
            );
        }

        // Algorithm is to check the following:
        // 1) (no mapping) if *everything* fits on the first dev (non mapped and mapped)
        // 2) if the mapped activations plus remaining fits on the nth device
        // 3) common case, iteratively find the optimal amount of layers to put on the nth device
        //   - if this is the first dev: must hold the non-mapped act and non-mapped model
        //   - otherwise, must hold the mapped act
        let remaining_kv_bytes = (layer..num_layers).map(kv_bytes_for_layer).sum::<usize>();
        let required_whole_capacity = if ordinal == 0 {
            checked_memory_sum([
                remaining,
                non_mapped_max.max(mapped_max),
                remaining_kv_bytes,
                extra_kv_bytes,
                base_device_memory_reservation_bytes,
            ])
        } else {
            checked_memory_sum([
                remaining,
                if dev.is_cpu() { 0 } else { mapped_max },
                remaining_kv_bytes,
            ])
        };

        let layers_on_dev = if required_whole_capacity.is_some_and(|required| cap >= required) {
            remaining = 0;
            num_layers - layer
        } else {
            // The CPU is the last resort: its prompt transients come out of host memory, not a reserve.
            let mut used = if dev.is_cpu() { 0 } else { mapped_max };
            let mut used_weight_bytes = 0usize;
            let mut count = 0;
            if ordinal == 0 {
                used = checked_memory_sum([
                    used.max(non_mapped_max),
                    non_mapped_size_in_bytes,
                    extra_kv_bytes,
                    base_device_memory_reservation_bytes,
                ])
                .unwrap_or(usize::MAX);
                used_weight_bytes = used_weight_bytes.saturating_add(non_mapped_size_in_bytes);
            }
            while let Some(&sz) = layer_sizes_in_bytes.last() {
                let Some(delta) = sz.checked_add(kv_bytes_for_layer(layer + count)) else {
                    break;
                };
                let Some(next_used) = used.checked_add(delta) else {
                    break;
                };
                if next_used > cap {
                    break;
                }
                layer_sizes_in_bytes.pop();
                used = next_used;
                used_weight_bytes = used_weight_bytes.saturating_add(sz);
                count += 1;
            }
            if count > 0 {
                remaining = remaining.saturating_sub(used_weight_bytes);
            } else {
                warn!(
                    "Device {} can fit 0 layers. Consider reducing auto map params from current: {params} (ex. reducing max seq len or max num images)",
                    dev.device_pretty_repr(),
                );
                ordinal += 1;
                continue;
            }
            count
        };
        if !dev.is_cpu() {
            mappings.push(DeviceLayerMapMetadata {
                ordinal,
                layers: layers_on_dev,
            });
            ordinal += 1;
        } else {
            includes_cpu = true;
        }
        layer += layers_on_dev;
    }
    if remaining > 0 {
        let over = b_to_mb!(remaining);
        anyhow::bail!(
            "This model does not fit on the devices {:?}, and exceeds total capacity by {}MB. Auto device mapping params: {params}",
            avail_copy.iter().rev().map(|(a, d)| format!("{} (avail: {}MB)", d.device_pretty_repr(), b_to_mb!(a))).collect::<Vec<_>>(),
            over
        );
    }
    if has_paged_attn && includes_cpu {
        let original_layers = layer_sizes_backup
            .take()
            .expect("layer sizes backup missing for paged attention fallback");
        // The original vector was in forward order, but `get_device_layers` handles
        // reversing internally, so we can pass it along unchanged.
        return get_device_layers(
            loader,
            config,
            num_layers,
            original_layers,
            non_mapped_size_in_bytes,
            total_model_size_in_bytes,
            devices,
            dtype,
            params,
            None,
        );
    }
    Ok(DeviceMapMetadata::from_num_device_layers(mappings))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_params_promote_to_multimodal_defaults_after_detection() {
        let params = AutoDeviceMapParams::Text {
            max_seq_len: 4096,
            max_batch_size: 7,
        };

        match params.maybe_promote_to_multimodal() {
            AutoDeviceMapParams::Multimodal {
                max_seq_len,
                max_batch_size,
                max_image_shape,
                max_num_images,
            } => {
                assert_eq!(max_seq_len, 4096);
                assert_eq!(max_batch_size, 7);
                assert_eq!(
                    max_image_shape,
                    (
                        AutoDeviceMapParams::DEFAULT_MAX_IMAGE_LENGTH,
                        AutoDeviceMapParams::DEFAULT_MAX_IMAGE_LENGTH,
                    )
                );
                assert_eq!(max_num_images, AutoDeviceMapParams::DEFAULT_MAX_NUM_IMAGES);
            }
            AutoDeviceMapParams::Text { .. } => panic!("expected multimodal parameters"),
        }
    }

    #[test]
    fn multimodal_params_preserve_explicit_limits_after_detection() {
        let params = AutoDeviceMapParams::Multimodal {
            max_seq_len: 8192,
            max_batch_size: 3,
            max_image_shape: (1536, 1024),
            max_num_images: 5,
        };

        match params.maybe_promote_to_multimodal() {
            AutoDeviceMapParams::Multimodal {
                max_seq_len,
                max_batch_size,
                max_image_shape,
                max_num_images,
            } => {
                assert_eq!(max_seq_len, 8192);
                assert_eq!(max_batch_size, 3);
                assert_eq!(max_image_shape, (1536, 1024));
                assert_eq!(max_num_images, 5);
            }
            AutoDeviceMapParams::Text { .. } => panic!("expected multimodal parameters"),
        }
    }

    #[test]
    fn memory_sum_saturates_capacity_accounting_overflow() {
        assert_eq!(saturating_memory_sum([usize::MAX - 2, 1, 1]), usize::MAX);
        assert_eq!(saturating_memory_sum([usize::MAX, 1]), usize::MAX);
    }

    #[test]
    fn checked_memory_sum_distinguishes_max_from_overflow() {
        assert_eq!(checked_memory_sum([usize::MAX - 1, 1]), Some(usize::MAX));
        assert_eq!(checked_memory_sum([usize::MAX, 1]), None);
    }

    #[test]
    fn post_load_utilization_is_not_frozen_to_the_inventory_estimate() {
        let requested = MemoryGpuConfig::Utilization(0.85);
        assert!(matches!(
            post_load_memory_config(requested, MemoryGpuConfig::MbAmount(45_000), true),
            MemoryGpuConfig::Utilization(value) if value == 0.85
        ));
        assert!(matches!(
            post_load_memory_config(requested, MemoryGpuConfig::MbAmount(45_000), false,),
            MemoryGpuConfig::MbAmount(45_000)
        ));
        assert!(matches!(
            post_load_memory_config(
                MemoryGpuConfig::MbAmount(60_000),
                MemoryGpuConfig::MbAmount(45_000),
                true,
            ),
            MemoryGpuConfig::MbAmount(60_000)
        ));
        assert!(matches!(
            post_load_memory_config(
                MemoryGpuConfig::BestEffortMbAmount {
                    target_mb: 60_000,
                    min_mb: Some(40_000),
                },
                MemoryGpuConfig::BestEffortMbAmount {
                    target_mb: 45_000,
                    min_mb: Some(40_000),
                },
                true,
            ),
            MemoryGpuConfig::BestEffortMbAmount {
                target_mb: 60_000,
                min_mb: Some(40_000),
            }
        ));
    }

    // Arithmetic of the automatic map's estimate against what titan measured (m4/devmap window 2, RTX 5080 laptop,
    // 15933 MiB; TITAN_DEVMAP_LOG pool peaks, every model pinned all-GPU, prompt transient = pool peak - pool used
    // before the step - the step's new KV). Weights come from the GGUF inventory (file bytes per tensor).
    const MIB: usize = 1 << 20;
    const USABLE_MIB: usize = 15117; // "primary device 15629 MiB available, 15117 MiB usable" (qwen3 load)

    fn kv_bytes_per_token(loader: &dyn DeviceMappedModelLoader, config: &str, dtype: DType) -> usize {
        let cfg = loader.model_config(config).unwrap();
        let per_layer = cfg.num_kv_heads() * (cfg.k_head_dim() + cfg.v_head_dim());
        let frac = loader.kv_cache_layer_fraction(config).unwrap();
        let kv_dtype = loader.kv_cache_dtype(config, dtype).unwrap();
        (per_layer as f64 * frac * cfg.num_layers() as f64) as usize * kv_dtype.size_in_bytes()
    }

    fn qwen3_14b() -> String {
        r#"{"vocab_size": 151936, "hidden_size": 5120, "intermediate_size": 17408, "num_hidden_layers": 40,
            "num_attention_heads": 40, "num_key_value_heads": 8, "hidden_act": "silu",
            "max_position_embeddings": 40960, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
            "sliding_window": null, "head_dim": 128, "quantization_config": null,
            "tie_word_embeddings": false, "max_window_layers": 40, "use_sliding_window": false}"#
            .to_string()
    }

    fn text(s: usize) -> AutoDeviceMapParams {
        AutoDeviceMapParams::Text { max_seq_len: s, max_batch_size: 1 }
    }

    fn mm(s: usize) -> AutoDeviceMapParams {
        AutoDeviceMapParams::Multimodal {
            max_seq_len: s,
            max_batch_size: 1,
            max_image_shape: (1024, 1024),
            max_num_images: 1,
        }
    }

    #[test]
    fn devmap_qwen3_14b_estimate() {
        let loader = super::super::Qwen3Loader;
        let cfg = qwen3_14b();
        // KV: 40 layers x 2 x 8 heads x 128 x 2 B = 160 KiB per token (10 GiB at 64k, 2.5 GiB at 16k).
        assert_eq!(kv_bytes_per_token(&loader, &cfg, DType::F16), 40 * 4096);
        let act = |s| loader.mapped_max_act_size_elems(&cfg, &text(s)).unwrap() * 2;
        // Flash-prefill prompt attention (models/qwen3.rs), measured beyond weights and KV (window 4, prefix cache
        // off, all 40 layers): 444 / 792 / 1441 / 2063 MiB at 3992 / 7992 / 11976 / 16184 tokens.
        for (s, mib) in [(3992, 444), (7992, 792), (11976, 1441), (16184, 2063)] {
            assert!(act(s) >= mib * MIB && act(s) <= mib * MIB * 2, "{s}: {}", act(s) / MIB);
        }
        // 16k: 8902 + 1118 MiB weights + 2560 MiB KV + activations fit the usable 15117 MiB (a 16184-token prompt
        // ran with all 40 layers on the GPU, pool peak 14779 MiB).
        let total = (8902 + 1118 + 2560) * MIB + act(16384);
        assert!(total <= USABLE_MIB * MIB, "{}", total / MIB);
        assert!(total >= (14779 - 10143 + 10020) * MIB, "{}", total / MIB);
        // 64k: one full-length prompt's transients exceed half the device; the reserve is capped there.
        assert!(act(65536) > USABLE_MIB * MIB / 2);
        assert_eq!(activation_reserve(act(65536), USABLE_MIB * MIB), USABLE_MIB * MIB / 2);
    }

    fn gemma4(layers: usize, hidden: usize, inter: usize, global_kv: usize, moe: bool) -> String {
        let types = (0..layers)
            .map(|i| if (i + 1) % 6 == 0 { "\"full_attention\"" } else { "\"sliding_attention\"" })
            .collect::<Vec<_>>()
            .join(", ");
        let moe = if moe {
            r#", "enable_moe_block": true, "num_experts": 128, "top_k_experts": 8, "moe_intermediate_size": 704"#
        } else {
            ""
        };
        format!(
            r#"{{"architectures": ["Gemma4ForCausalLM"], "text_config": {{
                "hidden_size": {hidden}, "intermediate_size": {inter}, "num_hidden_layers": {layers},
                "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 256, "global_head_dim": 512,
                "num_global_key_value_heads": {global_kv}, "sliding_window": 1024, "final_logit_softcapping": 30.0,
                "vocab_size": 262144, "tie_word_embeddings": true, "layer_types": [{types}]{moe}}}}}"#
        )
    }

    #[test]
    fn devmap_gemma4_12b_estimate() {
        let loader = super::super::Gemma4Loader;
        let cfg = gemma4(48, 3840, 15360, 1, false);
        // --dtype f32 keeps the cache in F16: 40 sliding x 2 x 8 x 256 + 8 full x 2 x 1 x 512, x 2 B.
        assert_eq!(loader.kv_cache_dtype(&cfg, DType::F32).unwrap(), DType::F16);
        assert_eq!(loader.kv_cache_dtype(&cfg, DType::BF16).unwrap(), DType::BF16);
        assert_eq!(kv_bytes_per_token(&loader, &cfg, DType::F32), 344_064);
        let act = |s, b| loader.mapped_max_act_size_elems(&cfg, &mm(s)).unwrap() * b;
        // Measured transients: f32 7998 tokens 1852 MiB, bf16 12095 tokens 1403 MiB.
        assert!(act(7998, 4) >= 1852 * MIB && act(7998, 4) <= 1852 * MIB * 17 / 10, "{}", act(7998, 4) / MIB);
        assert!(act(12095, 2) >= 1403 * MIB && act(12095, 2) <= 1403 * MIB * 18 / 10, "{}", act(12095, 2) / MIB);
        // Both deployed contexts fit all 48 layers: 6637 MiB weights + KV + activations.
        for (s, b, kv) in [(11264, 4, DType::F32), (12288, 2, DType::BF16)] {
            let total = 6637 * MIB + s * kv_bytes_per_token(&loader, &cfg, kv) + act(s, b);
            assert!(total <= 15149 * MIB, "{s}: {}", total / MIB);
        }
    }

    #[test]
    fn devmap_redcell_26b_estimate() {
        let loader = super::super::Gemma4Loader;
        let cfg = gemma4(30, 2816, 2112, 2, true);
        // 25 sliding x 2 x 8 x 256 + 5 full x 2 x 2 x 512, x 2 B (bf16).
        assert_eq!(kv_bytes_per_token(&loader, &cfg, DType::BF16), 225_280);
        let act = |s| loader.mapped_max_act_size_elems(&cfg, &mm(s)).unwrap() * 2;
        // Measured: 7998 tokens, 1289 MiB.
        assert!(act(7998) >= 1289 * MIB && act(7998) <= 1289 * MIB * 13 / 10, "{}", act(7998) / MIB);
        // 11676 MiB of weights (the GGUF; the old per-binding count charged the fused gate_up experts twice).
        let total = 11676 * MIB + 8192 * kv_bytes_per_token(&loader, &cfg, DType::BF16) + act(8192);
        assert!(total <= 15149 * MIB, "{}", total / MIB);
    }

    #[test]
    fn devmap_activation_reserve_caps_at_half_the_device() {
        assert_eq!(activation_reserve(3 << 30, 15 << 30), 3 << 30);
        assert_eq!(activation_reserve(20 << 30, 15 << 30), (15 << 30) / 2);
        assert_eq!(prefill_act_elems(1, 1, 0, 0, 0, 0, 0), PREFILL_ATTN_CHUNK_ELEMS);
    }

}
