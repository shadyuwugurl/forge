use std::path::PathBuf;

/// Architecture family of a model
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ArchitectureFamily {
    Transformer {
        heads: usize,
        hidden_size: usize,
        layers: usize,
        intermediate_size: usize,
    },
    Mamba {
        d_model: usize,
        d_state: usize,
        d_conv: usize,
        layers: usize,
    },
    Hybrid {
        transformer_layers: Vec<usize>,
        mamba_layers: Vec<usize>,
        hidden_size: usize,
    },
    MoE {
        num_experts: usize,
        experts_per_token: usize,
        hidden_size: usize,
        layers: usize,
    },
    Unknown,
}

/// A tensor's metadata (shape, dtype, location)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TensorMeta {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: crate::DType,
    pub offset: u64,
    pub size: usize,
}

impl TensorMeta {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn byte_size(&self) -> usize {
        self.num_elements() * self.dtype.byte_size()
    }
}

/// A loaded model's metadata
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Model {
    pub path: PathBuf,
    pub name: String,
    pub architecture: ArchitectureFamily,
    pub dtype: crate::DType,
    pub tensors: Vec<TensorMeta>,
    pub param_count: usize,
    pub config: serde_json::Value,
}

impl Model {
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.iter().any(|t| t.name == name)
    }

    pub fn get_tensor(&self, name: &str) -> Option<&TensorMeta> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn layer_tensors(&self, layer_idx: usize) -> Vec<&TensorMeta> {
        let prefix = format!("layers.{}.", layer_idx);
        self.tensors
            .iter()
            .filter(|t| t.name.starts_with(&prefix))
            .collect()
    }
}

/// Source model entry for merge configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelEntry {
    pub path: PathBuf,
    #[serde(default = "default_weight")]
    pub weight: f32,
    #[serde(default = "default_density")]
    pub density: f32,
    #[serde(default)]
    pub epsilon: f32,
}

fn default_weight() -> f32 { 1.0 }
fn default_density() -> f32 { 1.0 }

/// Slice specification for FrankenMerge (passthrough layer stacking)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SliceSpec {
    pub model: PathBuf,
    pub layer_range: (usize, usize),
}

/// Hardware profile for Apple Silicon
#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub chip: ChipType,
    pub gpu_cores: usize,
    pub has_neural_engine: bool,
    pub unified_memory_bytes: usize,
    pub memory_bandwidth_gbs: f64,
    pub performance_cores: usize,
    pub efficiency_cores: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChipType {
    M1,
    M1Pro,
    M1Max,
    M1Ultra,
    M2,
    M2Pro,
    M2Max,
    M2Ultra,
    M3,
    M3Pro,
    M3Max,
    M3Ultra,
    M4,
    M4Pro,
    M4Max,
    M4Ultra,
    M5,
    M5Pro,
    M5Max,
    M6,
    Unknown(String),
}

impl HardwareProfile {
    pub fn total_memory_gb(&self) -> f64 {
        self.unified_memory_bytes as f64 / 1e9
    }

    pub fn max_model_size_gb(&self) -> f64 {
        // Reserve ~4GB for OS + apps
        self.total_memory_gb() - 4.0
    }

    pub fn is_apple_silicon(&self) -> bool {
        !matches!(self.chip, ChipType::Unknown(_))
    }
}
