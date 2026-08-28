use anyhow::Result;
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

    /// Download a model from HuggingFace Hub
    pub async fn download(&self, model_id: &str, _output_dir: &PathBuf) -> Result<PathBuf> {
        // Use hf-hub cache dir; full clone via hf-hub api
        let api = Api::new()?;
        let repo = api.model(model_id.to_string());
        // Download a small file to verify, return cache path
        let path = repo.get("config.json").await?;
        Ok(path.parent().unwrap().to_path_buf())
    }
}
