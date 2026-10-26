//! ollama-style client commands (`ps`, `ls`, `pull`) that talk to a running titan swap server over its HTTP API.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use comfy_table::{presets::UTF8_FULL, Cell, ContentArrangement, Table};
use serde::Deserialize;

use crate::args::MatformerSelection;

#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<ModelRow>,
}

#[derive(Deserialize)]
struct ModelRow {
    id: String,
    #[serde(default)]
    status: Option<String>,
}

/// GET the server's model list, dropping the `default` alias row (a duplicate of the resident model).
fn fetch_models(url: &str) -> Result<Vec<ModelRow>> {
    let endpoint = format!("{}/v1/models", url.trim_end_matches('/'));
    let text = reqwest::blocking::Client::new()
        .get(&endpoint)
        .send()
        .and_then(|r| r.error_for_status())
        .and_then(|r| r.text())
        .with_context(|| format!("GET {endpoint} failed; is the titan server running at {url}?"))?;
    let parsed: ModelsResponse = serde_json::from_str(&text).context("parsing /v1/models response")?;
    Ok(parsed.data.into_iter().filter(|m| m.id != "default").collect())
}

fn human_size(bytes: u64) -> String {
    let gb = bytes as f64 / 1_073_741_824.0;
    if gb >= 1.0 {
        format!("{gb:.1} GB")
    } else {
        format!("{:.0} MB", bytes as f64 / 1_048_576.0)
    }
}

#[derive(Deserialize)]
struct SizeConfig {
    #[serde(default)]
    models: Vec<SizeEntry>,
}

#[derive(Deserialize)]
struct SizeEntry {
    name: Option<String>,
    model_id: Option<String>,
    format: Option<SizeFormat>,
}

#[derive(Deserialize)]
struct SizeFormat {
    quantized_file: Option<String>,
}

/// Map model name -> on-disk GGUF size from the swap config, for the SIZE column. Lenient: missing/unparsable -> empty.
fn model_sizes(config: &Path) -> HashMap<String, u64> {
    let mut sizes = HashMap::new();
    let Ok(text) = std::fs::read_to_string(config) else {
        return sizes;
    };
    let Ok(parsed) = toml::from_str::<SizeConfig>(&text) else {
        return sizes;
    };
    for entry in parsed.models {
        if let (Some(name), Some(dir), Some(file)) =
            (entry.name, entry.model_id, entry.format.and_then(|f| f.quantized_file))
        {
            if let Ok(meta) = std::fs::metadata(Path::new(&dir).join(&file)) {
                sizes.insert(name, meta.len());
            }
        }
    }
    sizes
}

fn render(rows: Vec<ModelRow>, sizes: &HashMap<String, u64>, header_status: bool) {
    if rows.is_empty() {
        println!("(no models)");
        return;
    }
    let mut table = Table::new();
    table.load_preset(UTF8_FULL).set_content_arrangement(ContentArrangement::Dynamic);
    let mut head = vec![Cell::new("NAME"), Cell::new("SIZE")];
    if header_status {
        head.push(Cell::new("STATUS"));
    }
    table.set_header(head);
    for row in rows {
        let size = sizes.get(&row.id).map(|b| human_size(*b)).unwrap_or_else(|| "-".to_string());
        let mut cells = vec![Cell::new(&row.id), Cell::new(size)];
        if header_status {
            cells.push(Cell::new(row.status.as_deref().unwrap_or("-")));
        }
        table.add_row(cells);
    }
    println!("{table}");
}

/// `ps`: models resident in VRAM now.
pub async fn run_ps(url: &str) -> Result<()> {
    let url = url.to_string();
    let config = default_config();
    tokio::task::spawn_blocking(move || -> Result<()> {
        let rows: Vec<ModelRow> = fetch_models(&url)?
            .into_iter()
            .filter(|m| m.status.as_deref() == Some("loaded"))
            .collect();
        render(rows, &model_sizes(&config), false);
        Ok(())
    })
    .await??;
    Ok(())
}

/// `ls`: all registered models with their load status.
pub async fn run_ls(url: &str, config: &Path) -> Result<()> {
    let url = url.to_string();
    let config = config.to_path_buf();
    tokio::task::spawn_blocking(move || -> Result<()> {
        let rows = fetch_models(&url)?;
        render(rows, &model_sizes(&config), true);
        Ok(())
    })
    .await??;
    Ok(())
}

fn default_config() -> PathBuf {
    PathBuf::from("/usr/local/etc/titan-engine/models.toml")
}

/// The serve dir (where GGUFs live) = the common `model_id` of the config's models.
fn serve_dir(config: &Path) -> Result<PathBuf> {
    let text = std::fs::read_to_string(config)
        .with_context(|| format!("reading {}", config.display()))?;
    let parsed: SizeConfig = toml::from_str(&text).context("parsing config")?;
    let dir = parsed
        .models
        .into_iter()
        .find_map(|m| m.model_id)
        .context("no [[models]] with model_id in config to locate the serve dir")?;
    Ok(PathBuf::from(dir))
}

/// `pull`: download a GGUF, place it in the serve dir, register it live, and persist it to the config.
pub async fn run_pull(
    repo: &str,
    file: Option<&str>,
    quant: &str,
    name: Option<&str>,
    url: &str,
    config: &Path,
) -> Result<()> {
    // Resolve the GGUF filename up front (async: lists repo files via the shared --quant path) if not given.
    let file = match file {
        Some(f) => f.to_string(),
        None => resolve_repo_file(repo, quant).await.with_context(|| {
            format!("auto-selecting a {quant}-bit GGUF in {repo} (or pass --file <name>)")
        })?,
    };
    let (repo, name, url, config) = (
        repo.to_string(),
        name.map(str::to_string),
        url.to_string(),
        config.to_path_buf(),
    );
    tokio::task::spawn_blocking(move || pull_blocking(&repo, &file, name.as_deref(), &url, &config)).await??;
    Ok(())
}

/// Resolve a repo + quant spec to a concrete GGUF filename (possibly a multi-file spec) via the engine's own
/// `--quant` selection, so `pull` picks the same file `serve -m <repo> --quant <q>` would.
async fn resolve_repo_file(repo: &str, quant: &str) -> Result<String> {
    let entry_toml = format!(
        "kind = \"text\"\nmodel_id = \"{repo}\"\n[format]\nformat = \"gguf\"\n[quantization]\nquant = \"{quant}\"\n"
    );
    let entry: crate::config::ModelEntry = toml::from_str(&entry_toml).context("building the probe entry")?;
    let mut model_type = entry.to_model_type(false);
    let matformer = MatformerSelection {
        config_path: None,
        slice_name: None,
    };
    crate::commands::serve::apply_quant_resolution(
        &mut model_type,
        &mistralrs_core::TokenSource::CacheToken,
        &matformer,
    )
    .await?;
    let model_selected = crate::commands::serve::convert_to_model_selected(&model_type, &matformer)?;
    match model_selected {
        mistralrs_core::ModelSelected::GGUF {
            quantized_filename, ..
        } => Ok(quantized_filename),
        _ => anyhow::bail!("{repo} did not resolve to a GGUF artifact; pass --file explicitly"),
    }
}

fn pull_blocking(repo: &str, file: &str, name: Option<&str>, url: &str, config: &Path) -> Result<()> {
    let name = name
        .map(str::to_string)
        .unwrap_or_else(|| repo.rsplit('/').next().unwrap_or(repo).to_lowercase());
    let serve = serve_dir(config)?;
    std::fs::create_dir_all(&serve).with_context(|| format!("creating {}", serve.display()))?;

    println!("pulling {repo} :: {file} ...");
    let api = hf_hub::api::sync::ApiBuilder::new()
        .build()
        .context("building the Hugging Face API client")?;
    let model = api.model(repo.to_string());
    // `file` may be a multi-file (split GGUF) spec joined by the delimiter; fetch and link each part.
    for part in file.split(mistralrs_core::GGUF_MULTI_FILE_DELIMITER).map(str::trim) {
        let cached = model
            .get(part)
            .with_context(|| format!("downloading {part} from {repo}"))?;
        let dest = serve.join(part);
        if std::fs::symlink_metadata(&dest).is_ok() {
            std::fs::remove_file(&dest).ok();
        }
        std::os::unix::fs::symlink(&cached, &dest)
            .with_context(|| format!("linking {} -> {}", dest.display(), cached.display()))?;
    }

    // Build the model's ModelSelected via the same path a config entry uses, so loading matches exactly.
    let entry_toml = format!(
        "kind = \"text\"\nmodel_id = \"{}\"\nname = \"{name}\"\n[format]\nformat = \"gguf\"\nquantized_file = \"{file}\"\n",
        serve.display()
    );
    let entry: crate::config::ModelEntry = toml::from_str(&entry_toml).context("building the model entry")?;
    let model_type = entry.to_model_type(false);
    let matformer = MatformerSelection {
        config_path: None,
        slice_name: None,
    };
    let model_selected = crate::commands::serve::convert_to_model_selected(&model_type, &matformer)?;

    // Register it live on the running server (no restart).
    let body = serde_json::json!({ "name": name, "model": model_selected });
    let endpoint = format!("{}/v1/models/create", url.trim_end_matches('/'));
    let resp = reqwest::blocking::Client::new()
        .post(&endpoint)
        .header("content-type", "application/json")
        .body(serde_json::to_string(&body)?)
        .send()
        .with_context(|| format!("POST {endpoint} failed; is the titan server running at {url}?"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().unwrap_or_default();
        anyhow::bail!("server rejected the model ({status}): {text}");
    }

    // Persist to the config so it survives a restart (root-owned file; append via sudo).
    append_to_config(config, &name, &serve, file)?;

    println!("ready: `{name}` is registered. Any request naming it will load it (swapping out the resident model).");
    Ok(())
}

fn append_to_config(config: &Path, name: &str, serve: &Path, file: &str) -> Result<()> {
    use std::io::Write;
    let snippet = format!(
        "\n[[models]]\nname = \"{name}\"\nkind = \"text\"\nmodel_id = \"{}\"\n[models.format]\nformat = \"gguf\"\nquantized_file = \"{file}\"\n[models.titan]\nidle_ttl_secs = 1800\n",
        serve.display()
    );
    let mut child = std::process::Command::new("sudo")
        .arg("tee")
        .arg("-a")
        .arg(config)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .spawn()
        .context("appending to the config needs sudo (tee -a)")?;
    child
        .stdin
        .take()
        .context("sudo tee stdin")?
        .write_all(snippet.as_bytes())?;
    let status = child.wait()?;
    if !status.success() {
        anyhow::bail!("failed to append the model to {}", config.display());
    }
    Ok(())
}
