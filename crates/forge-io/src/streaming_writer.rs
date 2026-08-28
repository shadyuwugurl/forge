use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use anyhow::{Context, Result};
use safetensors::tensor::TensorView;
use safetensors::serialize;

/// Streaming writer that outputs sharded safetensors files.
/// Processes one tensor at a time, writes to disk incrementally.
pub struct StreamingWriter {
    output_dir: PathBuf,
    shard_size: usize,
    current_shard: BufWriter<File>,
    current_shard_size: usize,
    shard_index: usize,
    tensor_index: Vec<TensorEntry>,
}

struct TensorEntry {
    name: String,
    shard: usize,
    offset: usize,
    size: usize,
    dtype: String,
    shape: Vec<usize>,
}

impl StreamingWriter {
    pub fn new(output_dir: &Path, shard_size: usize) -> Result<Self> {
        fs::create_dir_all(output_dir)?;

        let first_shard = output_dir.join(format!("model-{:05}-of-99999.safetensors", 0));
        let file = File::create(&first_shard)
            .with_context(|| format!("creating shard {}", first_shard.display()))?;

        Ok(Self {
            output_dir: output_dir.to_path_buf(),
            shard_size,
            current_shard: BufWriter::new(file),
            current_shard_size: 0,
            shard_index: 0,
            tensor_index: Vec::new(),
        })
    }

    /// Write a tensor to the current shard
    pub fn write_tensor(&mut self, name: &str, data: &[u8], dtype: &str, shape: &[usize]) -> Result<()> {
        if self.current_shard_size + data.len() > self.shard_size && self.current_shard_size > 0 {
            self.flush_shard()?;
        }

        let offset = self.current_shard_size;
        self.current_shard.write_all(data)?;

        self.tensor_index.push(TensorEntry {
            name: name.to_string(),
            shard: self.shard_index,
            offset,
            size: data.len(),
            dtype: dtype.to_string(),
            shape: shape.to_vec(),
        });

        self.current_shard_size += data.len();
        Ok(())
    }

    fn flush_shard(&mut self) -> Result<()> {
        self.current_shard.flush()?;
        self.shard_index += 1;

        let shard_path = self.output_dir.join(format!("model-{:05}-of-99999.safetensors", self.shard_index));
        let file = File::create(&shard_path)?;
        self.current_shard = BufWriter::new(file);
        self.current_shard_size = 0;

        Ok(())
    }

    /// Finalize all shards and write the index file
    pub fn finalize(mut self, model_name: &str) -> Result<()> {
        self.current_shard.flush()?;

        let total_shards = self.shard_index + 1;

        // Build weight map
        let weight_map: serde_json::Map<String, serde_json::Value> = self.tensor_index.iter()
            .map(|e| {
                let val = serde_json::json!({
                    "shard": format!("model-{:05}-of-{:05}.safetensors", e.shard, total_shards),
                    "offset": e.offset,
                    "size": e.size,
                    "dtype": e.dtype,
                    "shape": e.shape,
                });
                (e.name.clone(), val)
            })
            .collect();

        let index = serde_json::json!({
            "metadata": {
                "total_tensors": self.tensor_index.len(),
            },
            "weight_map": weight_map,
        });

        let index_path = self.output_dir.join("model.safetensors.index.json");
        fs::write(&index_path, serde_json::to_string_pretty(&index)?)?;

        // Rename shards with correct total count
        for i in 0..total_shards {
            let old = self.output_dir.join(format!("model-{:05}-of-99999.safetensors", i));
            let new = self.output_dir.join(format!("model-{:05}-of-{:05}.safetensors", i, total_shards));
            if old.exists() && old != new {
                fs::rename(&old, &new)?;
            }
        }

        Ok(())
    }

    pub fn current_shard_size(&self) -> usize {
        self.current_shard_size
    }

    pub fn total_bytes_written(&self) -> usize {
        self.tensor_index.iter().map(|e| e.size).sum()
    }
}
