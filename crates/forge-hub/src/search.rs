use anyhow::Result;

pub async fn search_models(_query: &str, _limit: usize) -> Result<Vec<ModelInfo>> {
    Ok(vec![])
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub author: String,
    pub downloads: u64,
    pub likes: u64,
    pub tags: Vec<String>,
}