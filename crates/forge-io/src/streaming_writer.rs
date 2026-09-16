use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use anyhow::{Context, Result};

/// Streaming writer that outputs sharded, standards-compliant safetensors files.
///
/// Each shard is a valid standalone `.safetensors` file (8-byte header length
/// + JSON header + raw tensor bytes), so outputs reload with [`crate::TensorStore`]
/// or any HuggingFace-compatible loader. Peak RAM stays bounded: tensor bytes
/// are spilled to per-shard temp files and only the header metadata lives in
/// memory; headers are prepended at [`StreamingWriter::finalize`] time.
pub struct StreamingWriter {
    output_dir: PathBuf,
    shard_size: usize,
    current_tmp: Option<BufWriter<File>>,
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

/// Validate a dtype string against the safetensors dtype vocabulary.
fn check_dtype(dtype: &str) -> Result<()> {
    match dtype {
        "F64" | "F32" | "F16" | "BF16" | "I64" | "I32" | "I16" | "I8"
        | "U64" | "U32" | "U16" | "U8" | "BOOL" => Ok(()),
        other => Err(anyhow::anyhow!("unsupported dtype for safetensors shard: '{other}'")),
    }
}

impl StreamingWriter {
    pub fn new(output_dir: &Path, shard_size: usize) -> Result<Self> {
        fs::create_dir_all(output_dir)?;
        let tmp = Self::tmp_path(output_dir, 0);
        let file = File::create(&tmp)
            .with_context(|| format!("creating shard temp {}", tmp.display()))?;
        Ok(Self {
            output_dir: output_dir.to_path_buf(),
            shard_size,
            current_tmp: Some(BufWriter::new(file)),
            current_shard_size: 0,
            shard_index: 0,
            tensor_index: Vec::new(),
        })
    }

    fn tmp_path(output_dir: &Path, shard: usize) -> PathBuf {
        output_dir.join(format!(".shard-{shard:05}.tmp"))
    }

    fn final_path(&self, shard: usize, total_shards: usize) -> PathBuf {
        self.output_dir.join(format!(
            "model-{0:05}-of-{1:05}.safetensors",
            shard + 1,
            total_shards
        ))
    }

    /// Write a tensor's raw bytes to the current shard spill file.
    pub fn write_tensor(&mut self, name: &str, data: &[u8], dtype: &str, shape: &[usize]) -> Result<()> {
        check_dtype(dtype)?;
        if self.current_shard_size + data.len() > self.shard_size && self.current_shard_size > 0 {
            self.flush_shard()?;
        }

        let offset = self.current_shard_size;
        self.current_tmp
            .as_mut()
            .context("writer already finalized")?
            .write_all(data)?;

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
        if let Some(mut w) = self.current_tmp.take() {
            w.flush()?;
        }
        self.shard_index += 1;

        let tmp = Self::tmp_path(&self.output_dir, self.shard_index);
        let file = File::create(&tmp)?;
        self.current_tmp = Some(BufWriter::new(file));
        self.current_shard_size = 0;

        Ok(())
    }

    /// Seal every shard (header + data), write the HF-standard index file,
    /// and remove temp spill files.
    pub fn finalize(mut self, _model_name: &str) -> Result<()> {
        if let Some(mut w) = self.current_tmp.take() {
            w.flush()?;
        }

        let total_shards = self.shard_index + 1;

        for shard in 0..total_shards {
            let entries: Vec<&TensorEntry> =
                self.tensor_index.iter().filter(|e| e.shard == shard).collect();
            let mut header = serde_json::Map::new();
            for e in &entries {
                header.insert(
                    e.name.clone(),
                    serde_json::json!({
                        "dtype": e.dtype,
                        "shape": e.shape,
                        "data_offsets": [e.offset, e.offset + e.size],
                    }),
                );
            }
            let header_bytes = serde_json::to_vec(&header)?;

            let final_path = self.final_path(shard, total_shards);
            let mut out = BufWriter::new(
                File::create(&final_path)
                    .with_context(|| format!("creating {}", final_path.display()))?,
            );
            out.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
            out.write_all(&header_bytes)?;

            let tmp = Self::tmp_path(&self.output_dir, shard);
            if tmp.exists() {
                let mut src = File::open(&tmp)?;
                std::io::copy(&mut src, &mut out)?;
                fs::remove_file(&tmp)?;
            }
            out.flush()?;
        }

        // HuggingFace-standard index: tensor name -> shard filename.
        let total_size: usize = self.tensor_index.iter().map(|e| e.size).sum();
        let weight_map: serde_json::Map<String, serde_json::Value> = self
            .tensor_index
            .iter()
            .map(|e| {
                let file = format!(
                    "model-{:05}-of-{:05}.safetensors",
                    e.shard + 1,
                    total_shards
                );
                (e.name.clone(), serde_json::Value::String(file))
            })
            .collect();
        let index = serde_json::json!({
            "metadata": {
                "total_size": total_size,
                "total_tensors": self.tensor_index.len(),
            },
            "weight_map": weight_map,
        });
        fs::write(
            self.output_dir.join("model.safetensors.index.json"),
            serde_json::to_string_pretty(&index)?,
        )?;

        // Single-shard compat: also expose `model.safetensors` so single-file
        // readers (TensorStore::open) can load the output directly.
        if total_shards == 1 {
            fs::copy(
                self.final_path(0, 1),
                self.output_dir.join("model.safetensors"),
            )?;
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
