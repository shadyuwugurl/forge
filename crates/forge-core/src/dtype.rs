use std::fmt;

/// Supported data types for model weights
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DType {
    F32,
    F16,
    BF16,
    FP8,
    Int8,
    Int4,
    UInt8,
    UInt32,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "float32"),
            DType::F16 => write!(f, "float16"),
            DType::BF16 => write!(f, "bfloat16"),
            DType::FP8 => write!(f, "float8"),
            DType::Int8 => write!(f, "int8"),
            DType::Int4 => write!(f, "int4"),
            DType::UInt8 => write!(f, "uint8"),
            DType::UInt32 => write!(f, "uint32"),
        }
    }
}

impl DType {
    pub fn byte_size(self) -> usize {
        match self {
            DType::F32 | DType::UInt32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::FP8 | DType::Int8 | DType::Int4 | DType::UInt8 => 1,
        }
    }
}
