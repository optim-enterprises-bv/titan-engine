//! Configuration file loading for mistralrs-cli
//!
//! Supports a full TOML configuration that mirrors the CLI options while
//! allowing multiple models without aliases.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

use crate::args::{
    AdapterOptions, CacheOptions, DeviceOptions, FormatOptions, GlobalOptions, ModelSourceOptions,
    ModelType, MultimodalAdapterOptions, MultimodalOptions, PagedAttentionOptions,
    QuantizationOptions, RuntimeOptions, SandboxOptions, ServerOptions,
};
use mistralrs_core::{ModelDType, NormalLoaderType, ReasoningEffort, TokenSource};

#[derive(Deserialize)]
#[serde(tag = "command", rename_all = "kebab-case")]
pub enum CliConfig {
    Serve(ServeConfig),
    Run(RunConfig),
}

#[derive(Deserialize, Default)]
pub struct ServeConfig {
    #[serde(default)]
    pub global: GlobalOptionsToml,
    #[serde(default)]
    pub runtime: RuntimeOptions,
    #[serde(default)]
    pub server: ServerOptions,
    #[serde(default)]
    pub paged_attn: PagedAttentionOptions,
    #[serde(default)]
    pub sandbox: SandboxOptions,
    #[serde(default)]
    pub models: Vec<ModelEntry>,
    #[serde(default)]
    pub default_model_id: Option<String>,
    /// titan swap mode: models load on demand in this process (see `mistralrs_core::TitanSwapPolicy`).
    #[serde(default)]
    pub titan_swap: Option<TitanSwapToml>,
}

#[derive(Deserialize, Default, Clone)]
pub struct TitanSwapToml {
    /// Models resident at once (default 1): a request for another evicts the least recently used.
    #[serde(default)]
    pub max_resident: Option<usize>,
}

/// `[models.titan]`: the model's TITAN_* settings (applied while it is loaded; the process env stays the default,
/// an empty string unsets one), an optional idle TTL after which it is unloaded, and optional prefix cache bounds:
/// `prefix_cache_n` sequences (absent: `[runtime] prefix_cache_n`; 0 turns the prefix cache off for this model) and
/// `prefix_cache_max_mib` MiB of KV kept on the GPU by the sequence-level prefix cache (absent: no byte bound; it is
/// `TITAN_PREFIX_CACHE_DEVICE_MIB` in the model's env).
#[derive(Deserialize, Default, Clone)]
pub struct TitanEntryToml {
    #[serde(default)]
    pub env: std::collections::BTreeMap<String, toml::Value>,
    #[serde(default)]
    pub idle_ttl_secs: Option<u64>,
    #[serde(default)]
    pub prefix_cache_n: Option<usize>,
    #[serde(default)]
    pub prefix_cache_max_mib: Option<usize>,
    /// PagedAttention for this model: "on", "off" or "auto" (absent: `[paged_attn] mode`).
    #[serde(default)]
    pub paged_attn: Option<crate::args::PagedAttnMode>,
    /// Paged KV cache type: "auto" (the activation dtype) or "f8e4m3" (absent: `[paged_attn] cache_type`).
    #[serde(default)]
    pub pa_cache_type: Option<String>,
    /// Paged KV pool by context length or by MiB; either replaces the global pool sizing for this model.
    #[serde(default)]
    pub pa_context_len: Option<usize>,
    #[serde(default)]
    pub pa_memory_mb: Option<usize>,
    /// Prompt tokens per scheduler step (absent: `[runtime] max_num_batched_tokens`).
    #[serde(default)]
    pub max_num_batched_tokens: Option<std::num::NonZeroUsize>,
    /// Concurrent sequences for this model (absent: `[runtime] max_seqs`). Scheduler only: the auto device map still
    /// sizes for `[models.device] max_batch_size`.
    #[serde(default)]
    pub max_seqs: Option<std::num::NonZeroUsize>,
}

impl TitanEntryToml {
    pub fn pa_cache_type(&self) -> Result<Option<mistralrs_core::PagedCacheType>> {
        self.pa_cache_type
            .as_deref()
            .map(|s| s.parse().map_err(|e: String| anyhow::anyhow!("pa_cache_type: {e}")))
            .transpose()
    }

    /// The core swap settings of this entry.
    pub fn settings(&self) -> Result<mistralrs_core::TitanModelSettings> {
        use crate::args::PagedAttnMode;
        Ok(mistralrs_core::TitanModelSettings {
            env: self.env_strings(),
            idle_ttl: self.idle_ttl_secs.filter(|s| *s > 0).map(std::time::Duration::from_secs),
            prefix_cache_n: self.prefix_cache_n,
            paged_attn: self.paged_attn.map(|m| match m {
                PagedAttnMode::Auto => None,
                PagedAttnMode::On => Some(true),
                PagedAttnMode::Off => Some(false),
            }),
            pa_cache_type: self.pa_cache_type()?,
            pa_context_len: self.pa_context_len,
            pa_memory_mb: self.pa_memory_mb,
            max_num_batched_tokens: self.max_num_batched_tokens.map(std::num::NonZeroUsize::get),
            max_seqs: self.max_seqs.map(std::num::NonZeroUsize::get),
        })
    }

    pub fn env_strings(&self) -> std::collections::HashMap<String, String> {
        let mut env: std::collections::HashMap<String, String> = self
            .env
            .iter()
            .map(|(k, v)| {
                let v = match v {
                    toml::Value::String(s) => s.clone(),
                    toml::Value::Boolean(b) => (if *b { "1" } else { "0" }).to_string(),
                    other => other.to_string(),
                };
                (k.clone(), v)
            })
            .collect();
        if let Some(mib) = self.prefix_cache_max_mib {
            env.insert("TITAN_PREFIX_CACHE_DEVICE_MIB".to_string(), mib.to_string());
        }
        env
    }
}

#[derive(Deserialize, Default)]
pub struct RunConfig {
    #[serde(default)]
    pub global: GlobalOptionsToml,
    #[serde(default)]
    pub runtime: RuntimeOptions,
    #[serde(default)]
    pub paged_attn: PagedAttentionOptions,
    #[serde(default)]
    pub sandbox: SandboxOptions,
    #[serde(default)]
    pub models: Vec<ModelEntry>,
    #[serde(default, alias = "enable_thinking")]
    pub thinking: Option<bool>,
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(default)]
    pub adapter: Option<String>,
}

#[derive(Deserialize, Default, Clone)]
pub struct GlobalOptionsToml {
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub log: Option<PathBuf>,
    #[serde(default)]
    pub token_source: Option<String>,
}

#[derive(Deserialize, Default, Clone, Copy)]
#[serde(rename_all = "kebab-case")]
pub enum ModelKind {
    #[default]
    Auto,
    Text,
    Multimodal,
    Diffusion,
    Speech,
    Embedding,
}

#[derive(Deserialize, Clone)]
pub struct ModelEntry {
    #[serde(default)]
    pub kind: ModelKind,
    pub model_id: String,
    /// API model name (the `model` field requests send); defaults to `model_id`.
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub titan: Option<TitanEntryToml>,
    #[serde(default)]
    pub tokenizer: Option<PathBuf>,
    #[serde(default)]
    pub arch: Option<NormalLoaderType>,
    #[serde(default)]
    pub dtype: ModelDType,
    #[serde(default)]
    pub hf_overrides: Option<mistralrs_core::HfConfigOverrides>,
    #[serde(default)]
    pub max_model_len: Option<usize>,
    #[serde(default)]
    pub format: FormatOptions,
    #[serde(default)]
    pub adapter: AdapterOptions,
    #[serde(default)]
    pub quantization: QuantizationOptions,
    #[serde(default)]
    pub device: DeviceOptionsToml,
    #[serde(default)]
    pub multimodal: MultimodalOptions,
    #[serde(default)]
    pub chat_template: Option<PathBuf>,
    #[serde(default)]
    pub jinja_explicit: Option<PathBuf>,
    /// Path to a MatFormer slice config. Only meaningful for MatFormer-trained models like Gemma 3n.
    #[serde(default)]
    pub matformer_config_path: Option<PathBuf>,
    /// Named slice to load from the MatFormer config.
    #[serde(default)]
    pub matformer_slice_name: Option<String>,
}

#[derive(Deserialize, Default, Clone)]
pub struct DeviceOptionsToml {
    #[serde(default)]
    pub cpu: Option<bool>,
    #[serde(default)]
    pub device_layers: Option<Vec<String>>,
    #[serde(default)]
    pub topology: Option<PathBuf>,
    #[serde(default)]
    pub hf_cache: Option<PathBuf>,
    #[serde(default)]
    pub max_seq_len: Option<usize>,
    #[serde(default)]
    pub max_batch_size: Option<usize>,
}

pub fn load_cli_config(path: &Path) -> Result<CliConfig> {
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("toml"))
        != Some(true)
    {
        anyhow::bail!("mistralrs-cli config files must be .toml");
    }

    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config file {}", path.to_string_lossy()))?;

    let config: CliConfig =
        toml::from_str(&contents).context("Failed to parse TOML config file")?;
    validate_config(&config)?;
    Ok(config)
}

fn validate_config(config: &CliConfig) -> Result<()> {
    let (models, default_model_id) = match config {
        CliConfig::Serve(cfg) => (&cfg.models, cfg.default_model_id.as_ref()),
        CliConfig::Run(cfg) => (&cfg.models, None),
    };

    if models.is_empty() {
        anyhow::bail!("Config must define at least one model in [[models]]");
    }

    if let CliConfig::Run(cfg) = config {
        let _ = crate::commands::normalize_requested_adapter(
            &models[0].to_model_type(false),
            cfg.adapter.as_deref(),
        )?;
    }

    if let Some(default_id) = default_model_id {
        let has_model = models
            .iter()
            .any(|model| model.model_id == *default_id || model.name.as_deref() == Some(default_id.as_str()));
        if !has_model {
            anyhow::bail!(
                "default_model_id '{}' does not match any model_id in [[models]]",
                default_id
            );
        }
    }

    let mut cpu_setting: Option<bool> = None;
    for model in models {
        if model.max_model_len == Some(0) {
            anyhow::bail!("max_model_len must be greater than zero");
        }
        model
            .adapter
            .validate()
            .map_err(|error| anyhow::anyhow!("invalid adapter configuration: {error}"))?;
        if matches!(model.kind, ModelKind::Multimodal)
            && (model.adapter.legacy_lora.is_some() || model.adapter.xlora.is_some())
        {
            anyhow::bail!(
                "multimodal models support dynamic language-model LoRA, but not legacy LoRA or X-LoRA"
            );
        }
        if let Some(titan) = &model.titan {
            titan.pa_cache_type().with_context(|| format!("[models.titan] of {}", model.model_id))?;
            if titan.pa_context_len.is_some() && titan.pa_memory_mb.is_some() {
                anyhow::bail!("[models.titan] of {}: set pa_context_len or pa_memory_mb, not both", model.model_id);
            }
        }
        if let Some(cpu) = model.device.cpu {
            match cpu_setting {
                None => cpu_setting = Some(cpu),
                Some(existing) if existing != cpu => {
                    anyhow::bail!(
                        "cpu must be consistent across all models (found both true and false)"
                    );
                }
                _ => {}
            }
        }
    }

    Ok(())
}

impl GlobalOptionsToml {
    pub fn to_global_options(&self) -> Result<GlobalOptions> {
        let token_source = match &self.token_source {
            Some(value) => value
                .parse()
                .map_err(|err| anyhow::anyhow!("Invalid token_source: {err}"))?,
            None => TokenSource::CacheToken,
        };

        Ok(GlobalOptions {
            seed: self.seed,
            log: self.log.clone(),
            token_source,
            verbose: 0,
        })
    }
}

impl DeviceOptionsToml {
    pub fn to_device_options(&self, cpu: bool) -> DeviceOptions {
        DeviceOptions {
            cpu,
            device_layers: self.device_layers.clone(),
            topology: self.topology.clone(),
            hf_cache: self.hf_cache.clone(),
            // DeviceOptions::default() is derived (0 / 0), not the CLI defaults; a 0 batch sizes the KV plans to nothing
            max_seq_len: self.max_seq_len.unwrap_or(mistralrs_core::AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN),
            max_batch_size: self.max_batch_size.unwrap_or(mistralrs_core::AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE),
        }
    }
}

impl ModelEntry {
    pub fn to_model_type(&self, cpu: bool) -> ModelType {
        let model = ModelSourceOptions {
            model_id: self.model_id.clone(),
            tokenizer: self.tokenizer.clone(),
            arch: self.arch.clone(),
            dtype: self.dtype,
            hf_overrides: self.hf_overrides.clone(),
            max_model_len: self.max_model_len,
        };

        let device = self.device.to_device_options(cpu);
        let cache = CacheOptions::default();

        match self.kind {
            ModelKind::Auto => ModelType::Auto {
                model,
                format: self.format.clone(),
                adapter: self.adapter.clone(),
                quantization: self.quantization.clone(),
                device,
                cache,
                multimodal: self.multimodal.clone(),
            },
            ModelKind::Text => ModelType::Text {
                model,
                format: self.format.clone(),
                adapter: self.adapter.clone(),
                quantization: self.quantization.clone(),
                device,
                cache,
            },
            ModelKind::Multimodal => ModelType::Multimodal {
                model,
                format: self.format.clone(),
                adapter: MultimodalAdapterOptions::from_adapter_options(&self.adapter),
                quantization: self.quantization.clone(),
                device,
                cache,
                multimodal: self.multimodal.clone(),
            },
            ModelKind::Diffusion => ModelType::Diffusion { model, device },
            ModelKind::Speech => ModelType::Speech { model, device },
            ModelKind::Embedding => ModelType::Embedding {
                model,
                format: self.format.clone(),
                quantization: self.quantization.clone(),
                device,
                cache,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omitted_model_kind_defaults_to_auto() {
        let config: CliConfig = toml::from_str(
            r#"
command = "serve"

[[models]]
model_id = "google/gemma-4-E4B-it"
"#,
        )
        .unwrap();

        let cfg = match config {
            CliConfig::Serve(cfg) => cfg,
            CliConfig::Run(_) => panic!("expected serve config"),
        };

        assert!(matches!(cfg.models[0].kind, ModelKind::Auto));
        assert!(matches!(
            cfg.models[0].to_model_type(false),
            ModelType::Auto { .. }
        ));
    }

    #[test]
    fn explicit_model_kind_is_preserved() {
        let config: CliConfig = toml::from_str(
            r#"
command = "serve"

[[models]]
kind = "multimodal"
model_id = "google/gemma-4-E4B-it"
"#,
        )
        .unwrap();

        let cfg = match config {
            CliConfig::Serve(cfg) => cfg,
            CliConfig::Run(_) => panic!("expected serve config"),
        };

        assert!(matches!(cfg.models[0].kind, ModelKind::Multimodal));
        assert!(matches!(
            cfg.models[0].to_model_type(false),
            ModelType::Multimodal { .. }
        ));
    }

    #[test]
    fn run_config_uses_structured_lora_preloads_and_explicit_selection() {
        let config: CliConfig = toml::from_str(
            r#"
command = "run"
adapter = "code"

[[models]]
model_id = "org/base"

[models.adapter]
enable_lora = true
lora = [{ alias = "code", source = "org/code-lora", revision = "refs/pr/7" }]
lora_max_adapters = 4
lora_max_rank = 64
lora_max_bytes = 1048576
"#,
        )
        .unwrap();

        validate_config(&config).unwrap();
        let cfg = match config {
            CliConfig::Run(cfg) => cfg,
            CliConfig::Serve(_) => panic!("expected run config"),
        };
        assert_eq!(cfg.adapter.as_deref(), Some("code"));
        assert_eq!(cfg.models[0].adapter.lora[0].alias, "code");
        assert_eq!(cfg.models[0].adapter.lora[0].source, "org/code-lora");
        assert_eq!(cfg.models[0].adapter.lora[0].revision(), "refs/pr/7");
        assert_eq!(cfg.models[0].adapter.lora_runtime_config().max_rank, 64);
    }

    #[test]
    fn duplicate_toml_lora_aliases_are_rejected() {
        let config: CliConfig = toml::from_str(
            r#"
command = "serve"

[[models]]
model_id = "org/base"

[models.adapter]
lora = [
    { alias = "code", source = "org/first" },
    { alias = "code", source = "org/second" },
]
"#,
        )
        .unwrap();

        assert!(validate_config(&config)
            .unwrap_err()
            .to_string()
            .contains("more than once"));
    }

    #[test]
    fn multimodal_toml_rejects_legacy_adapter_modes() {
        let config: CliConfig = toml::from_str(
            r#"
command = "serve"

[[models]]
kind = "multimodal"
model_id = "org/vision"

[models.adapter]
legacy_lora = "org/legacy"
legacy_lora_order = "order.json"
"#,
        )
        .unwrap();

        assert!(validate_config(&config)
            .unwrap_err()
            .to_string()
            .contains("not legacy LoRA or X-LoRA"));
    }

    fn serve(toml_src: &str) -> ServeConfig {
        match toml::from_str::<CliConfig>(toml_src).unwrap() {
            CliConfig::Serve(cfg) => cfg,
            CliConfig::Run(_) => panic!("expected serve config"),
        }
    }

    #[test]
    fn titan_prefix_cache_settings_are_per_model() {
        let cfg = serve(
            r#"
command = "serve"

[runtime]
prefix_cache_n = 8

[[models]]
name = "big"
model_id = "/m/big"
[models.titan]
idle_ttl_secs = 1800
[models.titan.env]
TITAN_MTP = "2"

[[models]]
name = "gemma"
model_id = "/m/gemma"
[models.titan]
prefix_cache_n = 0
prefix_cache_max_mib = 1024

[[models]]
name = "plain"
model_id = "/m/plain"
"#,
        );
        assert_eq!(cfg.runtime.prefix_cache_n, 8);
        let big = cfg.models[0].titan.clone().unwrap();
        assert_eq!(big.prefix_cache_n, None);
        assert_eq!(big.prefix_cache_max_mib, None);
        assert_eq!(
            big.env_strings(),
            std::collections::HashMap::from([("TITAN_MTP".to_string(), "2".to_string())])
        );
        let gemma = cfg.models[1].titan.clone().unwrap();
        assert_eq!(gemma.prefix_cache_n, Some(0));
        assert_eq!(gemma.prefix_cache_max_mib, Some(1024));
        assert_eq!(
            gemma.env_strings().get("TITAN_PREFIX_CACHE_DEVICE_MIB").map(String::as_str),
            Some("1024")
        );
        assert!(cfg.models[2].titan.is_none());
    }

    #[test]
    fn titan_prefix_cache_n_absent_is_the_global_default() {
        let cfg = serve(
            r#"
command = "serve"

[[models]]
model_id = "/m/a"
[models.titan]
idle_ttl_secs = 60
"#,
        );
        // no [runtime]: the CLI default (16), which every model without its own value gets
        assert_eq!(cfg.runtime.prefix_cache_n, 16);
        let t = cfg.models[0].titan.clone().unwrap();
        assert_eq!(t.prefix_cache_n, None);
        assert!(t.env_strings().is_empty());
        let policy = mistralrs_core::TitanSwapPolicy {
            default_model: "/m/a".to_string(),
            max_resident: 1,
            models: std::collections::HashMap::from([
                (
                    "/m/a".to_string(),
                    mistralrs_core::TitanModelSettings { prefix_cache_n: t.prefix_cache_n, ..Default::default() },
                ),
                (
                    "/m/b".to_string(),
                    mistralrs_core::TitanModelSettings { prefix_cache_n: Some(0), ..Default::default() },
                ),
            ]),
        };
        assert_eq!(policy.prefix_cache_n("/m/a", cfg.runtime.prefix_cache_n), 16);
        assert_eq!(policy.prefix_cache_n("/m/b", cfg.runtime.prefix_cache_n), 0);
        assert_eq!(policy.prefix_cache_n("/m/unknown", 16), 16);
    }

    #[test]
    fn titan_serving_settings_are_per_model() {
        let cfg = serve(
            r#"
command = "serve"

[paged_attn]
mode = "off"

[[models]]
name = "big"
model_id = "/m/big"
[models.titan.env]
TITAN_MTP = "2"

[[models]]
name = "spark"
model_id = "/m/spark"
[models.titan]
paged_attn = "on"
max_seqs = 4
max_num_batched_tokens = 16384
pa_cache_type = "f8e4m3"

[[models]]
name = "qwen"
model_id = "/m/qwen"
[models.titan]
paged_attn = "auto"
pa_context_len = 16384

[[models]]
name = "gemma"
model_id = "/m/gemma"
[models.titan]
paged_attn = "off"
pa_memory_mb = 2048
"#,
        );
        let s: Vec<_> = cfg.models.iter().map(|m| m.titan.clone().unwrap_or_default().settings().unwrap()).collect();
        // absent: every serving setting falls back to the global one (None), the TITAN_* env untouched
        assert_eq!(
            (s[0].paged_attn, s[0].pa_cache_type, s[0].pa_context_len, s[0].pa_memory_mb),
            (None, None, None, None)
        );
        assert_eq!((s[0].max_seqs, s[0].max_num_batched_tokens), (None, None));
        assert_eq!(s[0].env.get("TITAN_MTP").map(String::as_str), Some("2"));
        assert_eq!(s[1].paged_attn, Some(Some(true)));
        assert_eq!(s[1].pa_cache_type, Some(mistralrs_core::PagedCacheType::F8E4M3));
        assert_eq!((s[1].max_seqs, s[1].max_num_batched_tokens), (Some(4), Some(16384)));
        assert_eq!((s[2].paged_attn, s[2].pa_context_len), (Some(None), Some(16384)));
        assert_eq!((s[3].paged_attn, s[3].pa_memory_mb), (Some(Some(false)), Some(2048)));
        // a model without [models.titan] gets the defaults
        let plain = TitanEntryToml::default().settings().unwrap();
        assert_eq!((plain.paged_attn, plain.max_seqs, plain.max_num_batched_tokens), (None, None, None));
        assert!(plain.env.is_empty());
    }

    #[test]
    fn titan_serving_settings_reject_bad_values() {
        let bad = |titan: &str| {
            let src = format!("command = \"serve\"\n[[models]]\nmodel_id = \"/m/a\"\n[models.titan]\n{titan}\n");
            match toml::from_str::<CliConfig>(&src) {
                Err(_) => true,
                Ok(cfg) => validate_config(&cfg).is_err(),
            }
        };
        assert!(bad("paged_attn = \"yes\""));
        assert!(bad("pa_cache_type = \"f8e5m2\""));
        assert!(bad("max_seqs = 0"));
        assert!(bad("max_num_batched_tokens = 0"));
        assert!(bad("pa_context_len = 8192\npa_memory_mb = 1024"));
        assert!(!bad("pa_cache_type = \"auto\"\nmax_seqs = 1"));
    }

    #[test]
    fn titan_prefix_cache_n_rejects_bad_values() {
        assert!(toml::from_str::<CliConfig>(
            r#"
command = "serve"
[[models]]
model_id = "/m/a"
[models.titan]
prefix_cache_n = -1
"#
        )
        .is_err());
    }

    #[test]
    fn run_config_rejects_unconfigured_adapter_selection() {
        let config: CliConfig = toml::from_str(
            r#"
command = "run"
adapter = "math"

[[models]]
model_id = "org/base"

[models.adapter]
lora = [{ alias = "code", source = "org/code-lora" }]
"#,
        )
        .unwrap();

        assert!(validate_config(&config)
            .unwrap_err()
            .to_string()
            .contains("not configured"));
    }
}
