use thiserror::Error;

#[derive(Error, Debug)]
pub enum ForgeError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Safetensors error: {0}")]
    Safetensors(String),

    #[error("Config error: {0}")]
    Config(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Tensor not found: {0} in model")]
    TensorNotFound(String),

    #[error("Architecture mismatch: {0} vs {1}")]
    ArchitectureMismatch(String, String),

    #[error("Dimension mismatch: tensor {name} has shape {found:?}, expected {expected:?}")]
    DimensionMismatch {
        name: String,
        expected: Vec<usize>,
        found: Vec<usize>,
    },

    #[error("Merge error: {0}")]
    Merge(String),

    #[error("Quantization error: {0}")]
    Quantization(String),

    #[error("Evaluation error: {0}")]
    Evaluation(String),

    #[error("Hardware error: {0}")]
    Hardware(String),

    #[error("Memory error: need {needed_gb:.1} GB but only {available_gb:.1} GB available")]
    InsufficientMemory {
        needed_gb: f64,
        available_gb: f64,
    },

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
}

pub type Result<T> = std::result::Result<T, ForgeError>;
