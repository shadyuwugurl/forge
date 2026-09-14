use anyhow::Result;
use forge_core::{ArchitectureFamily, DType};

pub async fn get_model_info(_model_id: &str) -> Result<ModelInfo> {
    Ok(ModelInfo {
        id: "unknown".to_string(),
        architecture: ArchitectureFamily::Unknown,
        dtype: DType::BF16,
        param_count: 0,
    })
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub architecture: ArchitectureFamily,
    pub dtype: DType,
    pub param_count: usize,
}