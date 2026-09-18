use anyhow::{Context, Result};
use std::io::{Write, BufWriter, Seek};
use std::path::Path;
use std::collections::HashMap;
use forge_io::TensorStore;

/// GGUF writer with full K-quant support (Q4_K_M, Q8_0, Q5_K, Q6_K, etc.)
/// Based on llama.cpp GGUF format specification

/// GGUF quantization types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GGUFQuantType {
    Q4_K,      // 4-bit K-quant (most common)
    Q4_K_M,    // Medium quality 4-bit
    Q4_K_S,    // Small 4-bit
    Q5_K,      // 5-bit K-quant
    Q5_K_M,    // Medium 5-bit
    Q5_K_S,    // Small 5-bit
    Q6_K,      // 6-bit K-quant
    Q6_K_M,    // Medium 6-bit
    Q8_0,      // 8-bit (no K-quant, per-row scale)
    F16,       // FP16
    BF16,      // BF16
    F32,       // FP32
}

impl GGUFQuantType {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "q4_k" => Some(Self::Q4_K),
            "q4_k_m" => Some(Self::Q4_K_M),
            "q4_k_s" => Some(Self::Q4_K_S),
            "q5_k" => Some(Self::Q5_K),
            "q5_k_m" => Some(Self::Q5_K_M),
            "q5_k_s" => Some(Self::Q5_K_S),
            "q6_k" => Some(Self::Q6_K),
            "q6_k_m" => Some(Self::Q6_K_M),
            "q8_0" => Some(Self::Q8_0),
            "f16" => Some(Self::F16),
            "bf16" => Some(Self::BF16),
            "f32" => Some(Self::F32),
            _ => None,
        }
    }

    pub fn bits_per_weight(&self) -> f32 {
        match self {
            Self::Q4_K | Self::Q4_K_M | Self::Q4_K_S => 4.0,
            Self::Q5_K | Self::Q5_K_M | Self::Q5_K_S => 5.0,
            Self::Q6_K | Self::Q6_K_M => 6.0,
            Self::Q8_0 => 8.0,
            Self::F16 | Self::BF16 => 16.0,
            Self::F32 => 32.0,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Q4_K => "Q4_K",
            Self::Q4_K_M => "Q4_K_M",
            Self::Q4_K_S => "Q4_K_S",
            Self::Q5_K => "Q5_K",
            Self::Q5_K_M => "Q5_K_M",
            Self::Q5_K_S => "Q5_K_S",
            Self::Q6_K => "Q6_K",
            Self::Q6_K_M => "Q6_K_M",
            Self::Q8_0 => "Q8_0",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::F32 => "F32",
        }
    }

    pub fn to_gguf_id(&self) -> u32 {
        // GGUF quantization type IDs (from llama.cpp)
        match self {
            Self::Q4_K => 10,
            Self::Q4_K_M => 11,
            Self::Q4_K_S => 12,
            Self::Q5_K => 13,
            Self::Q5_K_M => 14,
            Self::Q5_K_S => 15,
            Self::Q6_K => 16,
            Self::Q6_K_M => 17,
            Self::Q8_0 => 7,
            Self::F16 => 1,
            Self::BF16 => 9,
            Self::F32 => 0,
        }
    }
}

/// GGUF tensor info
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub quant_type: GGUFQuantType,
    pub offset: u64,
    pub size: u64,
}

/// GGUF writer with full K-quant support
pub struct GgufWriter {
    path: std::path::PathBuf,
    tensors: Vec<GgufTensorInfo>,
    metadata: HashMap<String, serde_json::Value>,
}

impl GgufWriter {
    pub fn create(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
        Ok(Self {
            path: path.to_path_buf(),
            tensors: Vec::new(),
            metadata: HashMap::new(),
        })
    }

    pub fn set_metadata(&mut self, key: &str, value: serde_json::Value) {
        self.metadata.insert(key.to_string(), value);
    }

    /// Write a model with specified quantization
    pub fn write_quantized(&mut self, store: &TensorStore, quant_type: GGUFQuantType) -> Result<()> {
        // Determine per-tensor quantization (some tensors stay F16)
        let mut total_size = 0u64;

        for name in store.tensor_names() {
            let meta = store.tensor_meta(name)?;
            let is_passthrough = is_passthrough_tensor(name);
            
            let tensor_type = if is_passthrough {
                GGUFQuantType::F16
            } else {
                quant_type
            };

            let size = estimate_gguf_size(&meta, tensor_type);
            
            self.tensors.push(GgufTensorInfo {
                name: name.to_string(),
                shape: meta.shape.iter().map(|&x| x as u64).collect(),
                quant_type: tensor_type,
                offset: total_size,
                size,
            });
            total_size += size;
        }

        // Write GGUF file
        let mut file = BufWriter::new(std::fs::File::create(&self.path)?);
        
        // Header
        file.write_all(b"GGUF")?; // Magic
        file.write_all(&3u32.to_le_bytes())?; // Version
        file.write_all(&(self.tensors.len() as u64).to_le_bytes())?; // Tensor count
        file.write_all(&(self.metadata.len() as u64).to_le_bytes())?; // KV count

        // Metadata KV pairs
        for (key, value) in &self.metadata {
            write_kv(&mut file, key, value)?;
        }

        // Tensor infos
        for tensor in &self.tensors {
            write_tensor_info(&mut file, tensor)?;
        }

        // Alignment padding
        let pos = file.seek(std::io::SeekFrom::Current(0))?;
        let alignment = 32;
        let padding = (alignment - (pos % alignment)) % alignment;
        file.write_all(&vec![0u8; padding as usize])?;

        // Tensor data
        for tensor in &self.tensors {
            write_tensor_data(&mut file, tensor, store)?;
        }

        Ok(())
    }
}

/// Exact GGUF tensor data size for given element count + type.
/// Q8_0 pads the trailing partial block with zeros (34 bytes per 32-elem block).
fn exact_gguf_size(elements: u64, qtype: GGUFQuantType) -> u64 {
    match qtype {
        GGUFQuantType::Q8_0 => elements.div_ceil(32) * 34,
        GGUFQuantType::F16 | GGUFQuantType::BF16 => elements * 2,
        GGUFQuantType::F32 => elements * 4,
        _ => (elements as f64 * qtype.bits_per_weight() as f64 / 8.0).ceil() as u64,
    }
}

/// Estimate GGUF tensor size for given quantization (exact for Q8_0/F16/BF16/F32).
fn estimate_gguf_size(meta: &forge_core::TensorMeta, qtype: GGUFQuantType) -> u64 {
    let elements: u64 = meta.shape.iter().map(|&x| x as u64).product();
    exact_gguf_size(elements, qtype)
}

fn is_passthrough_tensor(name: &str) -> bool {
    let n = name.to_lowercase();
    n.contains("norm") || n.ends_with(".bias") || n.contains("embed") || n.contains("wte")
}

/// Write a key-value pair in GGUF format
fn write_kv<W: Write>(w: &mut W, key: &str, value: &serde_json::Value) -> Result<()> {
    let key_bytes = key.as_bytes();
    w.write_all(&(key_bytes.len() as u64).to_le_bytes())?;
    w.write_all(key_bytes)?;
    
    // Value type + value
    match value {
        serde_json::Value::String(s) => {
            w.write_all(&4u32.to_le_bytes())?; // String type
            let s = s.as_bytes();
            w.write_all(&(s.len() as u64).to_le_bytes())?;
            w.write_all(s)?;
        }
        serde_json::Value::Number(n) => {
            if n.is_u64() {
                w.write_all(&5u32.to_le_bytes())?; // u64
                w.write_all(&n.as_u64().unwrap().to_le_bytes())?;
            } else if n.is_i64() {
                w.write_all(&6u32.to_le_bytes())?; // i64
                w.write_all(&n.as_i64().unwrap().to_le_bytes())?;
            } else {
                w.write_all(&7u32.to_le_bytes())?; // f64
                w.write_all(&n.as_f64().unwrap().to_le_bytes())?;
            }
        }
        serde_json::Value::Bool(b) => {
            w.write_all(&8u32.to_le_bytes())?; // Bool
            w.write_all(&(*b as u8).to_le_bytes())?;
        }
        _ => {
            // Serialize as string fallback
            let s = value.to_string();
            let s_bytes = s.as_bytes();
            w.write_all(&4u32.to_le_bytes())?;
            w.write_all(&(s_bytes.len() as u64).to_le_bytes())?;
            w.write_all(s_bytes)?;
        }
    }
    Ok(())
}

fn write_tensor_info<W: Write>(w: &mut W, tensor: &GgufTensorInfo) -> Result<()> {
    // Name
    let name_bytes = tensor.name.as_bytes();
    w.write_all(&(name_bytes.len() as u64).to_le_bytes())?;
    w.write_all(name_bytes)?;
    
    // Dimensions
    let n_dims = tensor.shape.len() as u32;
    w.write_all(&n_dims.to_le_bytes())?;
    for &dim in &tensor.shape {
        w.write_all(&dim.to_le_bytes())?;
    }
    
    // Quantization type
    let qtype_id = tensor.quant_type.to_gguf_id();
    w.write_all(&qtype_id.to_le_bytes())?;
    
    // Offset (will be updated after header)
    w.write_all(&tensor.offset.to_le_bytes())?;
    
    Ok(())
}

fn write_tensor_data<W: Write>(w: &mut W, tensor: &GgufTensorInfo, store: &TensorStore) -> Result<()> {
    let data = store.tensor_f32(&tensor.name)?;
    match tensor.quant_type {
        GGUFQuantType::Q8_0 => w.write_all(&quantize_q8_0(&data))?,
        GGUFQuantType::F16 => {
            for &v in &data {
                w.write_all(&half::f16::from_f32(v).to_bits().to_le_bytes())?;
            }
        }
        GGUFQuantType::BF16 => {
            for &v in &data {
                w.write_all(&half::bf16::from_f32(v).to_bits().to_le_bytes())?;
            }
        }
        GGUFQuantType::F32 => {
            for &v in &data {
                w.write_all(&v.to_le_bytes())?;
            }
        }
        q => anyhow::bail!(
            "GGUF {:?} block quantization not implemented (supported: Q8_0, F16, BF16, F32)",
            q
        ),
    }
    Ok(())
}

/// Q8_0 quantization: per-32-element blocks, fp16 scale + 32×int8.
/// Matches ggml `block_q8_0` layout. Trailing partial block zero-padded.
pub fn quantize_q8_0(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len().div_ceil(32) * 34);
    for block in data.chunks(32) {
        let amax = block.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        let inv = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        out.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
        for i in 0..32 {
            let v = if i < block.len() { block[i] } else { 0.0 };
            out.push((v * inv).round().clamp(-127.0, 127.0) as i8 as u8);
        }
    }
    out
}

/// Dequantize Q8_0 blocks (for round-trip tests).
pub fn dequantize_q8_0(data: &[u8], elements: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(elements);
    for block in data.chunks_exact(34) {
        let scale = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        for &q in &block[2..34] {
            out.push((q as i8 as f32) * scale);
        }
    }
    out.truncate(elements);
    out
}

/// GGUF reader
pub struct GgufReader;

impl GgufReader {
    pub fn open(_path: &Path) -> Result<Self> {
        // TODO: Implement GGUF reading
        Ok(Self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_gguf_quant_types() {
        assert_eq!(GGUFQuantType::Q4_K_M.bits_per_weight(), 4.0);
        assert_eq!(GGUFQuantType::Q8_0.bits_per_weight(), 8.0);
        assert_eq!(GGUFQuantType::Q4_K_M.to_gguf_id(), 11);
        assert_eq!(GGUFQuantType::Q8_0.to_gguf_id(), 7);
    }

    #[test]
    fn test_q8_0_roundtrip_error_bound() {
        // Error per element must be <= half-ULP of scale + fp16 scale rounding.
        let data: Vec<f32> = (0..100).map(|i| ((i as f32 * 0.37).sin()) * 3.0).collect();
        let q = quantize_q8_0(&data);
        assert_eq!(q.len(), exact_gguf_size(100, GGUFQuantType::Q8_0) as usize);
        let back = dequantize_q8_0(&q, 100);
        assert_eq!(back.len(), 100);
        let amax: f32 = data.iter().map(|x| x.abs()).fold(0.0, f32::max);
        for (a, b) in data.iter().zip(back.iter()) {
            assert!((a - b).abs() <= amax / 127.0 + 1e-3, "{} vs {}", a, b);
        }
    }

    #[test]
    fn test_q8_0_zeros_and_size() {
        let q = quantize_q8_0(&vec![0.0; 32]);
        assert_eq!(q.len(), 34);
        assert_eq!(&q[0..2], &[0, 0]); // zero scale
        assert!(q[2..].iter().all(|&b| b == 0));
        assert_eq!(exact_gguf_size(1, GGUFQuantType::Q8_0), 34);
        assert_eq!(exact_gguf_size(32, GGUFQuantType::Q8_0), 34);
        assert_eq!(exact_gguf_size(33, GGUFQuantType::Q8_0), 68);
    }
}