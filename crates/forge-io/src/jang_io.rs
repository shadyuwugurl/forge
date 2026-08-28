use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// JangQ v2 format support
/// MLX-native mixed-precision quantized weights in standard safetensors

#[derive(Debug, Serialize, Deserialize)]
pub struct JangConfig {
    pub format: String,
    pub format_version: String,
    pub quantization: JangQuantization,
    pub source_model: JangSourceModel,
    pub runtime: JangRuntime,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JangQuantization {
    pub method: String,
    pub profile: String,
    pub target_bits: f32,
    pub actual_bits: f32,
    pub block_size: usize,
    pub bit_widths_used: Vec<u8>,
    pub quantization_scheme: String,
    pub quantization_backend: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JangSourceModel {
    pub name: String,
    pub dtype: String,
    pub parameters: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JangRuntime {
    pub total_weight_bytes: u64,
    pub total_weight_gb: f64,
}

/// JangQ sensitivity tiers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JangTier {
    /// Attention, MoE routers, MLA latent → 6-8 bit
    Critical,
    /// Embeddings, linear attention → 4-6 bit
    Important,
    /// MLP, MoE experts → 2-4 bit
    Compress,
}

impl JangTier {
    /// Classify a tensor by its name into a sensitivity tier
    pub fn classify(tensor_name: &str) -> Self {
        let name = tensor_name.to_lowercase();

        // Critical: attention projections, routers, MLA
        if name.contains("q_proj") || name.contains("k_proj") || name.contains("v_proj")
            || name.contains("o_proj") || name.contains("gate") && name.contains("router")
            || name.contains("mlp.gate") && !name.contains("up") && !name.contains("down")
            || name.contains("latent_proj") || name.contains("ssm")
            || name.contains("lm_head")
        {
            return JangTier::Critical;
        }

        // Important: embeddings, linear attention, shared experts
        if name.contains("embed") || name.contains("word_embeddings")
            || name.contains("shared_expert") || name.contains("wte")
            || name.contains("norm")
        {
            return JangTier::Important;
        }

        // Compress: MLP, MoE experts (bulk of parameters)
        JangTier::Compress
    }

    /// Get recommended bit width for this tier
    pub fn bits(&self, profile: &str) -> u8 {
        match (self, profile) {
            (JangTier::Critical, "JANG_1L") => 8,
            (JangTier::Critical, "JANG_2L") => 8,
            (JangTier::Critical, "JANG_4K") => 5,
            (JangTier::Important, "JANG_1L") => 6,
            (JangTier::Important, "JANG_2L") => 6,
            (JangTier::Important, "JANG_4K") => 4,
            (JangTier::Compress, "JANG_1L") => 4,
            (JangTier::Compress, "JANG_2L") => 2,
            (JangTier::Compress, "JANG_4K") => 3,
            _ => 4,
        }
    }
}

/// Load JangQ config from a model directory
pub fn load_jang_config(model_dir: &Path) -> Result<JangConfig> {
    let config_path = model_dir.join("jang_config.json");
    let data = std::fs::read_to_string(&config_path)?;
    let config: JangConfig = serde_json::from_str(&data)?;
    Ok(config)
}

/// Save JangQ config to a model directory
pub fn save_jang_config(model_dir: &Path, config: &JangConfig) -> Result<()> {
    let config_path = model_dir.join("jang_config.json");
    let data = serde_json::to_string_pretty(config)?;
    std::fs::write(&config_path, data)?;
    Ok(())
}
