use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use forge_io::TensorStore;
use forge_io::StreamingWriter;

use forge_core::{DType, TensorMeta};

/// Trait for merge operations
pub trait MergeOp {
    fn merge_tensor(&self, name: &str, meta: &TensorMeta) -> Result<Vec<f32>>;
}

/// Options for merge execution
pub struct MergeOptions {
    pub output_dtype: DType,
    pub base_model_dir: Option<std::path::PathBuf>,
    pub quiet: bool,
    pub verbose: bool,
}

/// Execute a merge operation, writing results to the streaming writer
pub fn execute_merge(
    op: &dyn MergeOp,
    stores: &[&TensorStore],
    output_dir: &std::path::Path,
    options: &MergeOptions,
) -> Result<()> {
    // Get union of all tensor names
    let mut all_names: Vec<String> = stores.iter()
        .flat_map(|s| s.tensor_names().into_iter().map(String::from))
        .collect::<std::collections::HashSet<String>>()
        .into_iter()
        .collect();
    all_names.sort();

    let pb = if !options.quiet {
        let pb = ProgressBar::new(all_names.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} tensors")
            .unwrap());
        Some(pb)
    } else {
        None
    };

    let mut writer = StreamingWriter::new(output_dir, 5 * 1024 * 1024 * 1024)?; // 5GB shards

    for name in &all_names {
        if let Some(meta) = stores[0].tensor_meta(name).ok() {
            let result = op.merge_tensor(name, &meta)?;

            // Convert f32 result to bytes based on output dtype
            let bytes = match options.output_dtype {
                DType::F16 => {
                    let mut buf = Vec::with_capacity(result.len() * 2);
                    for &val in &result {
                        let h = half::f16::from_f32(val);
                        buf.extend_from_slice(&h.to_bits().to_le_bytes());
                    }
                    buf
                }
                DType::BF16 => {
                    let mut buf = Vec::with_capacity(result.len() * 2);
                    for &val in &result {
                        let h = half::bf16::from_f32(val);
                        buf.extend_from_slice(&h.to_bits().to_le_bytes());
                    }
                    buf
                }
                _ => {
                    let mut buf = Vec::with_capacity(result.len() * 4);
                    for &val in &result {
                        buf.extend_from_slice(&val.to_le_bytes());
                    }
                    buf
                }
            };

            let dtype_str = match options.output_dtype {
                DType::F32 => "F32",
                DType::F16 => "F16",
                DType::BF16 => "BF16",
                _ => "F16",
            };

            writer.write_tensor(name, &bytes, dtype_str, &meta.shape)?;
        }

        if let Some(ref pb) = pb {
            pb.inc(1);
        }
    }

    if let Some(pb) = pb {
        pb.finish_with_message("merge complete");
    }

    writer.finalize("merged")?;
    Ok(())
}
