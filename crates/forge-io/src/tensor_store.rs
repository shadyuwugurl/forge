use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use anyhow::{Context, Result};
use memmap2::Mmap;
use safetensors::{SafeTensors, Dtype};

use forge_core::{TensorMeta, DType};

/// Memory-mapped tensor store for zero-copy reads.
/// Processes one tensor at a time — peak RAM = largest tensor only.
pub struct TensorStore {
    path: PathBuf,
    mmap: Mmap,
    index: HashMap<String, TensorInfo>,
    total_params: usize,
}

struct TensorInfo {
    dtype: DType,
    shape: Vec<usize>,
    offset: usize,
    size: usize,
}

impl TensorStore {
    /// Open a safetensors file with memory-mapped access
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("opening {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file) }
            .context("memory mapping file")?;

        // Parse the header to build tensor index
        let (offset, metadata) = SafeTensors::read_metadata(&mmap)
            .map_err(|e| anyhow::anyhow!("safetensors parse error: {}", e))?;

        let mut index = HashMap::new();
        let mut total_params = 0;

        for (name, info) in metadata.tensors() {
            let dtype = match info.dtype {
                Dtype::F32 => DType::F32,
                Dtype::F16 => DType::F16,
                Dtype::BF16 => DType::BF16,
                Dtype::F64 => DType::F32, // promote
                Dtype::U8 => DType::UInt8,
                Dtype::I8 => DType::Int8,
                Dtype::U16 => DType::UInt8,
                Dtype::I16 => DType::Int8,
                Dtype::U32 => DType::UInt32,
                Dtype::I32 => DType::Int8,
                Dtype::U64 => DType::UInt32,
                Dtype::I64 => DType::Int8,
                _ => DType::F16,
            };

            let shape: Vec<usize> = info.shape.clone();
            let num_elements: usize = shape.iter().product();
            let (start, end) = info.data_offsets;
            let size = end - start;

            index.insert(name.clone(), TensorInfo {
                dtype,
                shape,
                offset: start,
                size,
            });

            total_params += num_elements;
        }

        Ok(Self {
            path: path.to_path_buf(),
            mmap,
            index,
            total_params,
        })
    }

    /// List all tensor names
    pub fn tensor_names(&self) -> Vec<&str> {
        self.index.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a tensor exists
    pub fn has_tensor(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// Get tensor metadata
    pub fn tensor_meta(&self, name: &str) -> Result<TensorMeta> {
        let info = self.index.get(name)
            .with_context(|| format!("tensor '{}' not found in {}", name, self.path.display()))?;

        Ok(TensorMeta {
            name: name.to_string(),
            shape: info.shape.clone(),
            dtype: info.dtype,
            offset: info.offset as u64,
            size: info.size,
        })
    }

    /// Get raw bytes for a tensor (zero-copy slice of mmap)
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let info = self.index.get(name)
            .with_context(|| format!("tensor '{}' not found", name))?;

        let start = info.offset;
        let end = start + info.size;
        Ok(&self.mmap[start..end])
    }

    /// Get raw bytes for a tensor as a typed slice
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self.index.get(name)
            .with_context(|| format!("tensor '{}' not found", name))?;

        let start = info.offset;
        let end = start + info.size;
        let bytes = &self.mmap[start..end];

        match info.dtype {
            DType::F32 => {
                let floats: Vec<f32> = bytes.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                Ok(floats)
            }
            DType::F16 => {
                let floats: Vec<f32> = bytes.chunks_exact(2)
                    .map(|c| {
                        let h = half::f16::from_bits(u16::from_le_bytes([c[0], c[1]]));
                        h.to_f32()
                    })
                    .collect();
                Ok(floats)
            }
            DType::BF16 => {
                let floats: Vec<f32> = bytes.chunks_exact(2)
                    .map(|c| {
                        let h = half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]]));
                        h.to_f32()
                    })
                    .collect();
                Ok(floats)
            }
            _ => Err(anyhow::anyhow!("unsupported dtype for f32 conversion: {:?}", info.dtype).into()),
        }
    }

    pub fn total_params(&self) -> usize {
        self.total_params
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Iterator over all tensor names and their metadata
    pub fn iter_tensors(&self) -> impl Iterator<Item = (&str, TensorMeta)> + '_ {
        self.index.iter().map(|(name, info)| {
            let meta = TensorMeta {
                name: name.clone(),
                shape: info.shape.clone(),
                dtype: info.dtype,
                offset: info.offset as u64,
                size: info.size,
            };
            (name.as_str(), meta)
        })
    }
}
