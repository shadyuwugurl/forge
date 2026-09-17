use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use anyhow::{Context, Result};
use memmap2::Mmap;
use safetensors::{SafeTensors, Dtype};

use forge_core::{TensorMeta, DType};

/// Memory-mapped tensor store for zero-copy reads.
/// Supports single-file (`model.safetensors`) and sharded dirs
/// (`model-00001-of-00004.safetensors` + index.json).
/// Processes one tensor at a time — peak RAM = largest tensor only.
pub struct TensorStore {
    path: PathBuf,
    mmaps: Vec<Mmap>,
    index: HashMap<String, TensorInfo>,
    total_params: usize,
    /// Per-file byte offset where tensor data starts (8 + header len).
    data_bases: Vec<usize>,
}

struct TensorInfo {
    file_idx: usize,
    dtype: DType,
    shape: Vec<usize>,
    offset: usize,
    size: usize,
}

fn dtype_of(d: Dtype) -> DType {
    match d {
        Dtype::F32 => DType::F32,
        Dtype::F16 => DType::F16,
        Dtype::BF16 => DType::BF16,
        Dtype::F64 => DType::F32,
        Dtype::U8 => DType::UInt8,
        Dtype::I8 => DType::Int8,
        Dtype::U16 => DType::UInt8,
        Dtype::I16 => DType::Int8,
        Dtype::U32 => DType::UInt32,
        Dtype::I32 => DType::Int8,
        Dtype::U64 => DType::UInt32,
        Dtype::I64 => DType::Int8,
        _ => DType::F16,
    }
}

impl TensorStore {
    /// Open a safetensors file OR a sharded model directory.
    pub fn open(path: &Path) -> Result<Self> {
        if path.is_dir() {
            Self::open_dir(path)
        } else {
            Self::open_single(path)
        }
    }

    /// Open a single safetensors file
    pub fn open_single(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("opening {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file) }
            .context("memory mapping file")?;

        let (header_len, metadata) = SafeTensors::read_metadata(&mmap)
            .map_err(|e| anyhow::anyhow!("safetensors parse error: {}", e))?;
        let data_base = 8 + header_len;

        let mut index = HashMap::new();
        let mut total_params = 0;

        for (name, info) in metadata.tensors() {
            let shape: Vec<usize> = info.shape.clone();
            let num_elements: usize = shape.iter().product();
            let (start, end) = info.data_offsets;
            index.insert(name.clone(), TensorInfo {
                file_idx: 0,
                dtype: dtype_of(info.dtype),
                shape,
                offset: start,
                size: end - start,
            });
            total_params += num_elements;
        }

        Ok(Self {
            path: path.to_path_buf(),
            mmaps: vec![mmap],
            index,
            total_params,
            data_bases: vec![data_base],
        })
    }

    /// Open a sharded model directory (all `*.safetensors`, sorted).
    pub fn open_dir(dir: &Path) -> Result<Self> {
        // Prefer single-file layout when present
        let single = dir.join("model.safetensors");
        if single.exists() {
            let mut s = Self::open_single(&single)?;
            s.path = dir.to_path_buf();
            return Ok(s);
        }
        let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)
            .with_context(|| format!("reading dir {}", dir.display()))?
            .filter_map(|e| e.ok().map(|x| x.path()))
            .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
            .collect();
        shards.sort();
        if shards.is_empty() {
            anyhow::bail!("no .safetensors files in {}", dir.display());
        }

        let mut mmaps = Vec::with_capacity(shards.len());
        let mut data_bases = Vec::with_capacity(shards.len());
        let mut index: HashMap<String, TensorInfo> = HashMap::new();
        let mut total_params = 0;

        for (file_idx, shard) in shards.iter().enumerate() {
            let file = File::open(shard)
                .with_context(|| format!("opening {}", shard.display()))?;
            let mmap = unsafe { Mmap::map(&file) }
                .context("memory mapping shard")?;
            let (header_len, metadata) = SafeTensors::read_metadata(&mmap)
                .map_err(|e| anyhow::anyhow!("safetensors parse error in {}: {}", shard.display(), e))?;
            let data_base = 8 + header_len;
            for (name, info) in metadata.tensors() {
                let shape: Vec<usize> = info.shape.clone();
                let num_elements: usize = shape.iter().product();
                let (start, end) = info.data_offsets;
                // First occurrence wins; shards should be disjoint
                if !index.contains_key(name.as_str()) {
                    total_params += num_elements;
                }
                index.insert(name.clone(), TensorInfo {
                    file_idx,
                    dtype: dtype_of(info.dtype),
                    shape,
                    offset: start,
                    size: end - start,
                });
            }
            mmaps.push(mmap);
            data_bases.push(data_base);
        }

        Ok(Self {
            path: dir.to_path_buf(),
            mmaps,
            index,
            total_params,
            data_bases,
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

        let start = self.data_bases[info.file_idx] + info.offset;
        let end = start + info.size;
        Ok(&self.mmaps[info.file_idx][start..end])
    }

    /// Get tensor as f32 vector (handles F32/F16/BF16)
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self.index.get(name)
            .with_context(|| format!("tensor '{}' not found", name))?;

        let start = self.data_bases[info.file_idx] + info.offset;
        let end = start + info.size;
        let bytes = &self.mmaps[info.file_idx][start..end];

        match info.dtype {
            DType::F32 => {
                #[cfg(target_endian = "little")]
                {
                    let n = bytes.len() / 4;
                    let mut out = Vec::with_capacity(n);
                    unsafe {
                        out.set_len(n);
                        std::ptr::copy_nonoverlapping(
                            bytes.as_ptr(),
                            out.as_mut_ptr() as *mut u8,
                            n * 4,
                        );
                    }
                    Ok(out)
                }
                #[cfg(not(target_endian = "little"))]
                {
                    let floats: Vec<f32> = bytes.chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    Ok(floats)
                }
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
