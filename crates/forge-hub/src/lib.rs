use anyhow::{Context, Result};
use hf_hub::api::tokio::Api;
use std::path::PathBuf;

/// HuggingFace Hub integration: search, download, model discovery
pub struct HubClient {
    api: Api,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub author: String,
    pub name: String,
    pub downloads: u64,
    pub likes: u64,
    pub pipeline_tag: Option<String>,
    pub tags: Vec<String>,
}

impl HubClient {
    pub async fn new() -> Result<Self> {
        let api = Api::new()?;
        Ok(Self { api })
    }

    /// Search for models on HuggingFace Hub
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<ModelInfo>> {
        let url = format!(
            "https://huggingface.co/api/models?search={}&sort=downloads&direction=-1&limit={}",
            urlencoding::encode(query),
            limit
        );

        let response = reqwest::get(&url).await?;
        let models: Vec<serde_json::Value> = response.json().await?;

        Ok(models.into_iter().map(|m| {
            let id = m["id"].as_str().unwrap_or("").to_string();
            let parts: Vec<&str> = id.split('/').collect();
            ModelInfo {
                id: id.clone(),
                author: parts.first().unwrap_or(&"").to_string(),
                name: parts.last().unwrap_or(&"").to_string(),
                downloads: m["downloads"].as_u64().unwrap_or(0),
                likes: m["likes"].as_u64().unwrap_or(0),
                pipeline_tag: m["pipeline_tag"].as_str().map(String::from),
                tags: m["tags"].as_array()
                    .map(|a| a.iter().filter_map(|t| t.as_str().map(String::from)).collect())
                    .unwrap_or_default(),
            }
        }).collect())
    }

    /// Download a full model (weights + tokenizer + config) into `output_dir`.
    /// Uses symlinks to HF cache where possible to avoid 2x disk usage.
    /// Returns the output dir.
    pub async fn download(&self, model_id: &str, output_dir: &PathBuf) -> Result<PathBuf> {
        std::fs::create_dir_all(output_dir)?;
        let repo = self.api.model(model_id.to_string());

        // Always fetch sidecars first (small, fail fast if repo missing)
        let sidecars = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",
        ];
        for f in sidecars {
            match repo.get(f).await {
                Ok(cached) => {
                    let dst = output_dir.join(f);
                    if cached != dst {
                        link_or_copy(&cached, &dst)?;
                    }
                    eprintln!("  fetched {}", f);
                }
                Err(e) => {
                    eprintln!("  skip {}: {}", f, e);
                }
            }
        }

        // Try sharded index first, then single-file weights
        let mut shards: Vec<String> = vec![];
        match repo.get("model.safetensors.index.json").await {
            Ok(cached_index) => {
                let dst = output_dir.join("model.safetensors.index.json");
                if cached_index != dst {
                    link_or_copy(&cached_index, &dst)?;
                }
                eprintln!("  fetched model.safetensors.index.json");
                let data = std::fs::read_to_string(&dst)
                    .with_context(|| "reading downloaded index.json")?;
                let v: serde_json::Value = serde_json::from_str(&data)?;
                if let Some(map) = v.get("weight_map").and_then(|m| m.as_object()) {
                    let mut uniq = std::collections::BTreeSet::new();
                    for (_, fname) in map.iter() {
                        if let Some(s) = fname.as_str() {
                            uniq.insert(s.to_string());
                        }
                    }
                    shards.extend(uniq.into_iter());
                }
            }
            Err(_) => {
                eprintln!("  no model.safetensors.index.json, trying single file");
            }
        }

        if shards.is_empty() {
            // Single-file or unknown layout — try common weight filenames
            // Also try model-*.safetensors naming (e.g. NeoHorse, Clownius)
            for candidate in ["model.safetensors", "pytorch_model.safetensors"] {
                match repo.get(candidate).await {
                    Ok(cached) => {
                        let dst = output_dir.join(candidate);
                        if cached != dst {
                            link_or_copy(&cached, &dst)?;
                        }
                        eprintln!("  fetched {}", candidate);
                        shards.push(candidate.to_string());
                        break;
                    }
                    Err(e) => eprintln!("  skip {}: {}", candidate, e),
                }
            }
            // Fallback: sharded without index (model-00001-of-*.safetensors)
            if shards.is_empty() {
                eprintln!("  trying sharded pattern without index...");
                for i in 1..=8 {
                    for total in [4usize, 2, 8] {
                        let name = format!("model-{:05}-of-{:05}.safetensors", i, total);
                        if let Ok(cached) = repo.get(&name).await {
                            let dst = output_dir.join(&name);
                            if cached != dst {
                                link_or_copy(&cached, &dst)?;
                            }
                            eprintln!("  fetched {}", name);
                            shards.push(name);
                        }
                    }
                    if !shards.is_empty() && i >= 4 {
                        break;
                    }
                }
            }
        } else {
            for shard in &shards {
                let cached = repo.get(shard).await
                    .with_context(|| format!("downloading shard {}", shard))?;
                let dst = output_dir.join(shard);
                if cached != dst {
                    link_or_copy(&cached, &dst)?;
                }
                eprintln!("  fetched {}", shard);
            }
        }

        if shards.is_empty() {
            anyhow::bail!("no weights found for '{}' (tried index + model.safetensors)", model_id);
        }

        Ok(output_dir.clone())
    }
}

/// Link HF cache file to destination (symlink → hardlink → copy fallback).
/// Saves 2x disk for 19GB models.
fn link_or_copy(src: &std::path::Path, dst: &std::path::Path) -> Result<()> {
    let _ = std::fs::remove_file(dst);
    #[cfg(unix)]
    {
        if std::os::unix::fs::symlink(src, dst).is_ok() {
            return Ok(());
        }
        if std::fs::hard_link(src, dst).is_ok() {
            return Ok(());
        }
    }
    std::fs::copy(src, dst)?;
    Ok(())
}
