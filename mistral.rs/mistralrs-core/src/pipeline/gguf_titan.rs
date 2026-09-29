//! titan fork-local GGUF loading, registered alongside upstream's native GGUF loader.
//!
//! mistral.rs v0.9.4 (1fdb451) loads GGUF checkpoints through bindings into its native models and
//! dropped the `quantized_*` models. The titan models (tiered experts with the AVX2 CPU twin, MTP
//! from the GGUF nextn head, piecewise CUDA graphs, chunked prefill, the hybrid prefix cache) live
//! on that older path, and upstream's native models cannot run them on titan (sync/upstream-bench.md).
//! So `GGUFLoader::load_model_from_path` sends the architectures below here, as the fork's
//! `GGUFPipeline` did before the sync, and upstream's native path handles every other GGUF:
//!
//! - `qwen35moe`, `qwen35`, `qwen3next`: `quantized_qwen35_moe` (hybrid GDN + attention, MTP)
//! - `gpt-oss`: `quantized_gpt_oss` (tiered MXFP4 experts)
//! - `qwen3moe`: `quantized_qwen3_moe` when `TITAN_TIERED=1` (M1 tiering); upstream otherwise
//!
//! `TITAN_GGUF_ROUTE=upstream` sends everything to upstream's loader (comparisons). The titan path
//! runs without PagedAttention (the fork's MTP, hybrid prefix cache and chunked prefill are unpaged).

use crate::pipeline::llg::build_llg_factory;
use crate::pipeline::text_models_inputs_processor::{InputMetadata, ModelInputs};
use crate::pipeline::{
    AnyMoePipelineMixin, CacheManager, CacheManagerMixin, EitherCache, ForwardInputsResult,
    GeneralMetadata, IsqPipelineMixin, MetadataMixin, ModelCategory, ModelForwardContext,
    ModelKind, ModelPaths, PreProcessingMixin, RecurrentBatchKind, RecurrentMetadata,
};
use crate::device_map::{self, DeviceMapper};
use crate::distributed::WorkerTransferData;
use crate::gguf::{
    get_gguf_chat_template, titan_convert_gguf_to_hf_tokenizer, Content, GGUFArchitecture,
    GgufTokenizerConversion,
};
use crate::kv_cache::{FullCacheManager, HybridCacheManager, NormalCacheManager};
use crate::paged_attention::AttentionImplementation;
use crate::pipeline::chat_template::{calculate_eos_tokens, BeginEndUnkPadTok, GenerationConfig};
use crate::pipeline::loaders::DeviceMappedModelLoader;
use crate::pipeline::sampling::sample_and_add_toks;
use crate::pipeline::{get_chat_template, ChatTemplate, Modalities, SupportedModality};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::utils::gguf_metadata::{ContentConfig, GgufDeviceMapLoaderInner};
use crate::utils::model_config::FromGGUF;
use crate::utils::tokenizer::get_tokenizer;
use crate::{
    distributed, models::quantized_gpt_oss::ModelWeights as QGptOss,
    models::quantized_qwen35_moe::ModelWeights as QQwen35MoE,
    models::quantized_qwen3_moe::ModelWeights as QQwen3MoE, DeviceMapSetting, PagedAttentionConfig,
    Pipeline, Topology, TryIntoDType,
};
use anyhow::Result;
use candle_core::{Device, Tensor};
use either::Either;
use mistralrs_quant::IsqType;
use rand_isaac::Isaac64Rng;
use std::any::Any;
use std::sync::Arc;
use std::{env, fs};
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{info, warn};

/// Default VRAM the tiered auto plan leaves for CUDA graph memory when `TITAN_CUDA_GRAPHS=1`.
const TITAN_CUDA_GRAPHS_RESERVE_MIB: usize = 256;

/// The titan architecture of a GGUF file (its `general.architecture`), when it loads through the
/// fork-local models.
pub(super) fn titan_arch(paths: &dyn ModelPaths) -> Result<Option<String>> {
    if env::var("TITAN_GGUF_ROUTE").is_ok_and(|v| v == "upstream") {
        return Ok(None);
    }
    let Some(first) = paths.get_weight_filenames().first() else {
        return Ok(None);
    };
    let mut file = fs::File::open(first)?;
    let content = candle_core::quantized::gguf_file::Content::read(&mut file)
        .map_err(|e| anyhow::anyhow!("reading GGUF header of {}: {e}", first.display()))?;
    let arch = match content.metadata.get("general.architecture") {
        Some(candle_core::quantized::gguf_file::Value::String(s)) => s.clone(),
        _ => return Ok(None),
    };
    let titan = match arch.as_str() {
        "qwen35moe" | "qwen35" | "qwen3next" | "gpt-oss" => true,
        "qwen3moe" => mistralrs_quant::TieredExperts::enabled(),
        _ => false,
    };
    Ok(titan.then_some(arch))
}

enum Model {
    Qwen3MoE(QQwen3MoE),
    Qwen35MoE(QQwen35MoE),
    GptOss(QGptOss),
}

pub struct TitanGGUFPipeline {
    model: Model,
    tokenizer: Arc<Tokenizer>,
    chat_template: Arc<ChatTemplate>,
    model_id: String,
    metadata: Arc<GeneralMetadata>,
    generation_defaults: Option<crate::ModelGenerationDefaults>,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    #[cfg(feature = "cuda")]
    cuda_sparse_rejection: std::sync::Mutex<Option<crate::speculative::CudaSparseRejectionWorkspace>>,
}

/// What the titan path needs of the `GGUFLoader` that routed here.
pub(super) struct TitanLoadArgs<'a> {
    pub model_id: Option<&'a String>,
    pub quantized_model_id: &'a str,
    pub no_kv_cache: bool,
    pub chat_template: Option<&'a String>,
    pub jinja_explicit: Option<&'a String>,
    pub topology: Option<&'a Topology>,
    pub kind: ModelKind,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn load(
    args: TitanLoadArgs<'_>,
    arch_name: &str,
    paths: &dyn ModelPaths,
    dtype: &dyn TryIntoDType,
    device: &Device,
    silent: bool,
    mut mapper: DeviceMapSetting,
    in_situ_quant: Option<IsqType>,
    paged_attn_config: Option<PagedAttentionConfig>,
) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
    info!("GGUF architecture `{arch_name}`: titan fork-local loader (TITAN_GGUF_ROUTE=upstream for upstream's)");
    if in_situ_quant.is_some() {
        anyhow::bail!("You are trying to in-situ quantize a GGUF model. This will not do anything.");
    }
    if paged_attn_config.is_some() {
        warn!("titan GGUF models run without PagedAttention (MTP, hybrid prefix cache and chunked prefill are unpaged); disabling it");
    }

    let mut readers = Vec::new();
    for filename in paths.get_weight_filenames() {
        readers.push(std::fs::File::open(filename)?);
    }
    // titan tiered experts read host experts in place from the page cache (TITAN_TIERED_MMAP=1)
    let mmaps = if mistralrs_quant::TieredExperts::mmap_enabled() {
        readers
            .iter()
            .map(|f| {
                let m = unsafe { memmap2::Mmap::map(f)? };
                // TITAN_TIERED_MADV=hugepage|random: whole-file advice (mmap cost experiments)
                let _ = match std::env::var("TITAN_TIERED_MADV").as_deref() {
                    Ok("hugepage") => m.advise(memmap2::Advice::HugePage),
                    Ok("random") => m.advise(memmap2::Advice::Random),
                    _ => Ok(()),
                };
                Ok(Arc::new(m))
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        Vec::new()
    };
    let mut readers = readers.iter_mut().collect::<Vec<_>>();
    let mut model = Content::from_readers(&mut readers)?;
    if !mmaps.is_empty() {
        model = model.with_mmaps(mmaps)?;
    }
    if !silent {
        model.print_metadata()?;
    }
    let arch = model.arch();

    let mut num_layers = model.get_metadata()[&format!("{arch}.block_count")].to_u32()? as usize;
    if matches!(
        arch,
        GGUFArchitecture::Qwen35 | GGUFArchitecture::Qwen35MoE | GGUFArchitecture::Qwen3Next
    ) {
        // the nextn (MTP) blocks are not trunk layers; the model loads them on the main device
        num_layers -= crate::models::quantized_qwen35_moe::nextn_predict_layers(
            model.get_metadata(),
            &arch.to_string(),
        );
    }

    // the CLI builds multimodal params when it found a projector next to the model (ignored here)
    if let DeviceMapSetting::Auto(crate::AutoDeviceMapParams::Multimodal {
        max_seq_len,
        max_batch_size,
        ..
    }) = mapper
    {
        mapper = DeviceMapSetting::Auto(crate::AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        });
    }
    if let DeviceMapSetting::Auto(params) = mapper.clone() {
        crate::titan_monitor::set_max_seq_len(params.max_seq_len());
        let devices = device_map::get_all_similar_devices(device)?;
        let dtype = dtype.try_into_dtype(&devices.iter().collect::<Vec<_>>())?;
        let model = GgufDeviceMapLoaderInner {
            model: &model,
            arch,
        };
        // titan tiered experts: the auto GPU fraction must leave room for the KV cache of this
        // context length, sized the same way the device map sizes it.
        if mistralrs_quant::TieredExperts::wants_auto_plan() {
            let cfg = model.model_config("this is a dummy config!")?;
            let frac = model.kv_cache_layer_fraction("this is a dummy config!")?;
            let per_layer = params.max_batch_size()
                * cfg.num_kv_heads()
                * params.max_seq_len()
                * (cfg.k_head_dim() + cfg.v_head_dim())
                * dtype.size_in_bytes();
            let kv = (per_layer as f64 * cfg.num_layers() as f64 * frac).ceil() as usize;
            // TITAN_CUDA_GRAPHS=1: graph memory (captured decode segments) comes from its own pool
            let graphs = if std::env::var("TITAN_CUDA_GRAPHS").is_ok_and(|v| v == "1") {
                std::env::var("TITAN_CUDA_GRAPHS_RESERVE_MIB")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(TITAN_CUDA_GRAPHS_RESERVE_MIB)
                    << 20
            } else {
                0
            };
            mistralrs_quant::TieredExperts::set_auto_extra_reserve(kv + graphs);
        }
        let layer_sizes_in_bytes =
            model.layer_sizes_in_bytes("this is a dummy config!", dtype, 1, None)?;
        let non_mapped_size_in_bytes =
            model.non_mapped_size_in_bytes("this is a dummy config!", dtype, 1, None, None)?;
        let total_model_size_in_bytes =
            layer_sizes_in_bytes.iter().sum::<usize>() + non_mapped_size_in_bytes;
        let new = model.get_device_layers(
            "this is a dummy config!",
            num_layers,
            layer_sizes_in_bytes,
            non_mapped_size_in_bytes,
            total_model_size_in_bytes,
            &devices,
            dtype,
            &params,
            None,
        )?;
        mapper = DeviceMapSetting::Map(new);
    }

    #[cfg(feature = "cuda")]
    if let Device::Cuda(dev) = &device {
        unsafe { dev.disable_event_tracking() };
    }

    let use_nccl = mistralrs_quant::distributed::use_nccl();
    let available_devices = if let Ok(payload) = env::var(distributed::IS_DAEMON_FLAG) {
        let payload: WorkerTransferData = serde_json::from_str(&payload)?;
        let WorkerTransferData::Init { worker_rank, .. } = payload;
        vec![candle_core::Device::new_cuda(worker_rank + 1)?]
    } else if use_nccl {
        vec![candle_core::Device::new_cuda(0)?]
    } else {
        device_map::get_all_similar_devices(device)?
    };
    let pipeline_mapper =
        mapper.into_mapper(num_layers, device, args.topology, &available_devices)?;
    let mapper = mapper.into_mapper(num_layers, device, args.topology, &available_devices)?;

    let GgufTokenizerConversion {
        tokenizer,
        bos,
        eos,
        unk,
    } = if paths.get_tokenizer_filename().to_string_lossy().is_empty() {
        titan_convert_gguf_to_hf_tokenizer(&model)?
    } else {
        GgufTokenizerConversion {
            tokenizer: get_tokenizer(paths.get_tokenizer_filename(), None)?,
            bos: None,
            eos: None,
            unk: None,
        }
    };

    // Only load gguf chat template if there is nothing else
    let gguf_chat_template =
        if paths.get_template_filename().is_none() && args.chat_template.is_none() {
            get_gguf_chat_template(&model)?
        } else {
            None
        };

    let model_config_metadata: ContentConfig = (&model).into();
    let internal_dtype = mapper.get_min_dtype(dtype)?;
    let attention = AttentionImplementation::Eager;

    let model = match arch {
        GGUFArchitecture::Qwen3MoE => Model::Qwen3MoE(QQwen3MoE::from_gguf(
            model,
            device,
            mapper,
            attention,
            internal_dtype,
        )?),
        GGUFArchitecture::Qwen35 | GGUFArchitecture::Qwen35MoE | GGUFArchitecture::Qwen3Next => {
            Model::Qwen35MoE(QQwen35MoE::from_gguf(
                model,
                device,
                mapper,
                attention,
                internal_dtype,
            )?)
        }
        GGUFArchitecture::GptOss => Model::GptOss(QGptOss::from_gguf(
            model,
            device,
            mapper,
            attention,
            internal_dtype,
        )?),
        a => anyhow::bail!("titan GGUF loader: unsupported architecture `{a:?}`"),
    };

    let gen_conf: Option<GenerationConfig> = match paths.get_gen_conf_filename() {
        Some(f) => Some(serde_json::from_str(&fs::read_to_string(f)?)?),
        None => None,
    };
    let chat_template_explicit = paths
        .get_chat_template_explicit()
        .as_ref()
        .map(|x| x.to_string_lossy().to_string());
    let mut chat_template = get_chat_template(
        paths,
        args.jinja_explicit,
        chat_template_explicit.as_ref(),
        args.chat_template,
        gguf_chat_template,
    );

    let max_seq_len = match model {
        Model::Qwen3MoE(ref p) => p.max_seq_len,
        Model::Qwen35MoE(ref p) => p.max_seq_len,
        Model::GptOss(ref p) => p.max_seq_len,
    };
    let llg_factory = build_llg_factory(tokenizer.clone())?;
    let num_hidden_layers = match model {
        Model::Qwen3MoE(ref model) => model.cache.normal().0.len(),
        Model::Qwen35MoE(ref model) => model.cache.hybrid().num_layers(),
        Model::GptOss(ref model) => model.cache.normal().0.len(),
    };

    if chat_template.bos_token.is_none() {
        if let Some(v) = bos {
            chat_template.bos_token = Some(BeginEndUnkPadTok(Either::Left(v)));
        }
    }
    if chat_template.eos_token.is_none() {
        if let Some(v) = eos {
            chat_template.eos_token = Some(BeginEndUnkPadTok(Either::Left(v)));
        }
    }
    if chat_template.unk_token.is_none() {
        if let Some(v) = unk {
            chat_template.unk_token = Some(BeginEndUnkPadTok(Either::Left(v)));
        }
    }

    let generation_defaults = gen_conf
        .as_ref()
        .and_then(GenerationConfig::generation_defaults);
    let eos = calculate_eos_tokens(&chat_template, gen_conf.as_ref(), &tokenizer);
    Ok(Arc::new(Mutex::new(TitanGGUFPipeline {
        model,
        tokenizer: tokenizer.into(),
        chat_template: Arc::new(chat_template),
        model_id: args
            .model_id
            .cloned()
            .unwrap_or_else(|| args.quantized_model_id.to_string()),
        metadata: Arc::new(GeneralMetadata {
            max_seq_len,
            llg_factory: Some(llg_factory),
            no_kv_cache: args.no_kv_cache,
            no_prefix_cache: false,
            num_hidden_layers,
            eos_tok: eos,
            kind: args.kind,
            is_xlora: false,
            activation_dtype: internal_dtype,
            sliding_window: None,
            cache_config: None,
            cache_engine: None,
            model_metadata: Some(Arc::new(model_config_metadata)),
            modalities: Modalities {
                input: vec![SupportedModality::Text],
                output: vec![SupportedModality::Text],
            },
            loaded_for_uqff_write: false,
        }),
        generation_defaults,
        mapper: pipeline_mapper,
        #[cfg(feature = "cuda")]
        cuda_sparse_rejection: std::sync::Mutex::new(None),
    })))
}

impl PreProcessingMixin for TitanGGUFPipeline {
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        Some(self.chat_template.clone())
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

impl IsqPipelineMixin for TitanGGUFPipeline {
    fn re_isq_model(&mut self, _dtype: IsqType) -> Result<()> {
        anyhow::bail!("You are trying to in-situ requantize a GGML model. This will not do anything.")
    }
}

impl CacheManagerMixin for TitanGGUFPipeline {
    fn clone_in_cache(&self, seqs: &mut [&mut Sequence]) -> candle_core::Result<()> {
        match self.cache() {
            EitherCache::Full(_) => FullCacheManager.clone_in_cache(self, seqs, false),
            EitherCache::Normal(_) => NormalCacheManager.clone_in_cache(self, seqs, false),
            EitherCache::Hybrid(_) => HybridCacheManager.clone_in_cache(self, seqs, false),
        }
    }
    fn clone_out_cache(&self, seqs: &mut [&mut Sequence]) {
        match self.cache() {
            EitherCache::Full(_) => FullCacheManager.clone_out_cache(self, seqs, false),
            EitherCache::Normal(_) => NormalCacheManager.clone_out_cache(self, seqs, false),
            EitherCache::Hybrid(_) => HybridCacheManager.clone_out_cache(self, seqs, false),
        }
    }
    fn set_none_cache(
        &self,
        seqs: &mut [&mut Sequence],
        _reset_non_granular: bool,
        modify_draft_cache: bool,
        load_preallocated_cache: bool,
    ) -> candle_core::Result<()> {
        match self.cache() {
            EitherCache::Full(_) => {
                FullCacheManager.set_none_cache(self, seqs, modify_draft_cache, false)
            }
            EitherCache::Normal(_) => NormalCacheManager.set_none_cache(
                self,
                seqs,
                modify_draft_cache,
                load_preallocated_cache,
            ),
            EitherCache::Hybrid(_) => HybridCacheManager.set_none_cache(
                self,
                seqs,
                modify_draft_cache,
                load_preallocated_cache,
            ),
        }
    }
    fn cache(&self) -> &EitherCache {
        match self.model {
            Model::Qwen3MoE(ref model) => &model.cache,
            Model::Qwen35MoE(ref model) => &model.cache,
            Model::GptOss(ref model) => &model.cache,
        }
    }
}

impl MetadataMixin for TitanGGUFPipeline {
    fn device(&self) -> Device {
        match self.model {
            Model::Qwen3MoE(ref model) => model.device.clone(),
            Model::Qwen35MoE(ref model) => model.device.clone(),
            Model::GptOss(ref model) => model.device.clone(),
        }
    }
    fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        Some(self.tokenizer.clone())
    }
    fn name(&self) -> String {
        self.model_id.clone()
    }
    fn reset_non_granular_state(&self) {}
    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.metadata.clone()
    }
    fn generation_defaults(&self) -> Option<crate::ModelGenerationDefaults> {
        self.generation_defaults.clone()
    }
    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        Some(&*self.mapper)
    }
}

#[async_trait::async_trait]
impl Pipeline for TitanGGUFPipeline {
    fn requires_uniform_completion_batch(&self) -> bool {
        false
    }

    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> Result<ForwardInputsResult, candle_core::Error> {
        let ModelInputs {
            input_ids,
            input_ids_full: _,
            seqlen_offsets,
            seqlen_offsets_full: _,
            context_lens,
            position_ids,
            paged_attn_meta,
            flash_meta,
            flash_meta_full: _,
            recurrent_batch_kind,
            adapter_leases: _,
        } = *inputs.downcast().expect("Downcast failed.");
        // The titan models run unpaged: v0.9.4 builds paged metadata only with a cache engine.
        let paged_attn_meta = match (&self.metadata.cache_engine, &paged_attn_meta) {
            (None, _) => None,
            (Some(_), None) => candle_core::bail!("Forward step expected a PagedAttention input metadata."),
            (Some(engine), Some(meta)) => Some((engine.get_kv_cache().clone(), meta)),
        };
        // The fork's unpaged MTP verifies staged drafts as a (multi-row) decode step; v0.9.4 tags
        // those rows `SpeculativeDecode`, which the titan models treat as `Decode`, as before.
        let recurrent_batch_kind = match recurrent_batch_kind {
            RecurrentBatchKind::SpeculativeDecode => RecurrentBatchKind::Decode,
            other => other,
        };
        let logits = match self.model {
            Model::Qwen3MoE(ref model) => {
                model.forward(&input_ids, &seqlen_offsets, context_lens, paged_attn_meta)?
            }
            Model::GptOss(ref model) => {
                model.forward(&input_ids, &seqlen_offsets, context_lens, paged_attn_meta)?
            }
            Model::Qwen35MoE(ref model) => {
                let recurrent_metadata = {
                    let hybrid = model.cache.hybrid();
                    hybrid.state_indices().cloned().map(|indices| {
                        RecurrentMetadata::new(
                            recurrent_batch_kind,
                            indices,
                            hybrid.state_indices_host().map(ToOwned::to_owned),
                        )
                    })
                };
                let mut ctx = ModelForwardContext::new(
                    &seqlen_offsets,
                    &context_lens,
                    &position_ids,
                    paged_attn_meta
                        .as_ref()
                        .map(|(kv_cache, meta)| (kv_cache.as_slice(), *meta)),
                    &flash_meta,
                )
                .with_recurrent_batch_kind(recurrent_batch_kind)
                .with_recurrent_metadata(recurrent_metadata);
                model.forward(&input_ids, &mut ctx)?
            }
        };
        if return_raw_logits {
            Ok(ForwardInputsResult::RawLogits { logits })
        } else {
            Ok(ForwardInputsResult::CausalGeneration { logits })
        }
    }

    async fn sample_causal_gen(
        &self,
        seqs: &mut [&mut Sequence],
        logits: Vec<Tensor>,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<(), candle_core::Error> {
        sample_and_add_toks(self, seqs, logits, prefix_cacher, disable_eos_stop, rng).await
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::Text
    }

    fn attach_speculative(
        &mut self,
        config: crate::speculative::SpeculativeConfig,
    ) -> Result<(), candle_core::Error> {
        match config {
            crate::speculative::SpeculativeConfig::Off => Ok(()),
            crate::speculative::SpeculativeConfig::Mtp(cfg) => match self.mtp_model() {
                Some(m) => {
                    if cfg.n_predict.is_some_and(|n| Some(n) != m.mtp_draft_len()) {
                        warn!(
                            "GGUF MTP draft length comes from TITAN_MTP ({:?}), ignoring --mtp-n-predict",
                            m.mtp_draft_len()
                        );
                    }
                    Ok(())
                }
                None => candle_core::bail!(
                    "GGUF MTP: set TITAN_MTP=<draft tokens> for a qwen35moe GGUF with nextn layers, without PagedAttention"
                ),
            },
            #[allow(unreachable_patterns)]
            _ => candle_core::bail!("titan GGUF models support MTP speculation only"),
        }
    }

    fn verifies_unpaged_speculation(&self) -> bool {
        self.mtp_model().is_some()
    }

    fn hybrid_prefix_restore(&self, seq: &mut Sequence) -> Result<(), candle_core::Error> {
        match &self.model {
            Model::Qwen35MoE(m) if self.metadata.cache_engine.is_none() => m.prefix_restore(seq),
            _ => candle_core::bail!("this GGUF model cannot resume a hybrid prefix-cache entry"),
        }
    }

    fn hybrid_prefix_export(
        &self,
        seq: &Sequence,
    ) -> Result<Option<crate::kv_cache::HybridPrefixEntry>, candle_core::Error> {
        match &self.model {
            Model::Qwen35MoE(m) if self.metadata.cache_engine.is_none() => {
                m.prefix_export(seq.get_toks())
            }
            _ => Ok(None),
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn try_sample_speculative_causal_gen(
        &mut self,
        seqs: &mut [&mut Sequence],
        logits: &[Tensor],
        batched_logits: Option<&Tensor>,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
        _metadata: Option<crate::pipeline::text_models_inputs_processor::PagedAttentionMeta>,
        logger: &crate::IntervalLogger,
    ) -> Result<bool, candle_core::Error> {
        let Some(rollback) = self.mtp_model().map(|m| m.spec_rollback()) else {
            crate::speculative::driver::clear_staged_speculative_tokens(seqs);
            return Ok(false);
        };
        let cache = crate::speculative::unpaged::UnpagedSpeculativeCacheAccess::new(rollback);
        crate::speculative::driver::try_sample_speculative_causal_gen(
            self,
            seqs,
            logits,
            batched_logits,
            prefix_cacher,
            disable_eos_stop,
            rng,
            &cache,
            logger,
        )
        .await
    }
}

impl TitanGGUFPipeline {
    /// The qwen35moe model when it drafts with its MTP block (unpaged caches only).
    fn mtp_model(&self) -> Option<&QQwen35MoE> {
        match &self.model {
            Model::Qwen35MoE(m) if m.has_mtp() && self.metadata.cache_engine.is_none() => Some(m),
            _ => None,
        }
    }
}

impl crate::speculative::driver::SpeculativePipelineExt for TitanGGUFPipeline {
    fn has_speculative_proposer(&self) -> bool {
        self.mtp_model().is_some()
    }

    /// One batch-1 plan at the TITAN_MTP draft length, fed the trunk hidden of the last kept row.
    fn speculative_plan(&self, batch_size: usize) -> Option<crate::speculative::SpeculativeBatchPlan> {
        let m = self.mtp_model()?;
        if batch_size != 1 {
            return None;
        }
        Some(crate::speculative::SpeculativeBatchPlan::new(m.mtp_draft_len()?))
    }

    fn speculative_observe(&self, _observation: crate::speculative::SpeculativeBatchObservation) {}

    fn speculative_bypass(&mut self, _seq_ids: &[usize]) -> candle_core::Result<()> {
        Ok(())
    }

    fn speculative_target_hiddens(
        &self,
        rows: &[(usize, usize)],
    ) -> candle_core::Result<Option<Tensor>> {
        match self.mtp_model() {
            Some(m) => m.mtp_target_hiddens(rows),
            None => Ok(None),
        }
    }

    fn speculative_propose(
        &mut self,
        ctx: crate::speculative::SpeculativeProposeBatchCtx<'_>,
    ) -> candle_core::Result<Option<crate::speculative::SpeculativeProposalBatch>> {
        let Some(m) = self.mtp_model() else {
            return Ok(None);
        };
        if ctx.sequences.len() != 1 || ctx.base_lens.len() != 1 {
            return Ok(None);
        }
        Ok(m
            .mtp_propose(ctx.sequences[0].get_toks(), ctx.base_lens[0])?
            .map(|tokens| {
                crate::speculative::SpeculativeProposalBatch::new(vec![
                    crate::speculative::SpeculativeProposal::new(tokens),
                ])
            }))
    }

    fn speculative_prepare_propose(
        &mut self,
        _ctx: crate::speculative::SpeculativeProposePrepareCtx<'_>,
    ) -> candle_core::Result<Option<Box<dyn crate::speculative::SpeculativeProposePreparation>>>
    {
        Ok(None)
    }

    /// The rollback of rejected rows happens in the unpaged cache guard (`finish_verification`).
    fn speculative_commit(
        &mut self,
        _rows: &[crate::speculative::SpeculativeCommitRow],
    ) -> candle_core::Result<()> {
        Ok(())
    }

    fn build_speculative_verify_inputs(
        &self,
        input_meta: InputMetadata,
    ) -> candle_core::Result<Box<dyn Any>> {
        Ok(Box::new(ModelInputs {
            input_ids: input_meta.input,
            input_ids_full: None,
            seqlen_offsets: input_meta.positions,
            seqlen_offsets_full: None,
            context_lens: input_meta.context_lens,
            position_ids: input_meta.position_ids,
            paged_attn_meta: input_meta.paged_attn_meta,
            flash_meta: input_meta.flash_meta,
            flash_meta_full: None,
            recurrent_batch_kind: RecurrentBatchKind::Decode,
            adapter_leases: Arc::from([]),
        }))
    }

    #[cfg(feature = "cuda")]
    fn cuda_sparse_rejection_workspace(
        &self,
    ) -> &std::sync::Mutex<Option<crate::speculative::CudaSparseRejectionWorkspace>> {
        &self.cuda_sparse_rejection
    }
}

impl AnyMoePipelineMixin for TitanGGUFPipeline {}
