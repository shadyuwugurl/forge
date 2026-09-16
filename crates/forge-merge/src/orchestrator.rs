use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use forge_io::TensorStore;
use forge_io::StreamingWriter;
use forge_core::{DType, TensorMeta};
use forge_metal::{MetalMerge, HardwareInfo};

/// Trait for merge operations
pub trait MergeOp {
    fn merge_tensor(&self, name: &str, meta: &TensorMeta) -> Result<Vec<f32>>;

    /// Merge one tensor from per-model columns (streaming: one tensor at a
    /// time, never whole models). The default forwards to `merge_tensor`,
    /// ignoring the inputs, so single-model ops keep working unchanged.
    fn merge_tensors(&self, name: &str, meta: &TensorMeta, _inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
        self.merge_tensor(name, meta)
    }
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
    op: &(dyn MergeOp + Sync),
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

    // Try to init Metal backend for GPU-accelerated merges
    let metal = MetalMerge::new().ok();

    let mut writer = StreamingWriter::new(output_dir, 5 * 1024 * 1024 * 1024)?; // 5GB shards

    // M5c: parallel tensor pipeline. Workers load + merge + dtype-convert;
    // the main thread owns the StreamingWriter and writes in arrival order
    // (shard layout is order-independent). Guard-aware: a shared byte
    // budget caps in-flight tensors so peak stays bounded:
    //   peak ≈ budget (4GB) + largest tensor, never whole models.
    // FORGE_WORKERS overrides the worker count (default: ncpu, min 1).
    let workers = std::env::var("FORGE_WORKERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
        .max(1)
        .min(all_names.len().max(1));
    execute_names_parallel(
        op,
        stores,
        &all_names,
        workers,
        options.output_dtype,
        pb.as_ref(),
        |name, shape, dtype_str, bytes| writer.write_tensor(name, &bytes, dtype_str, &shape),
    )?;

    if let Some(pb) = pb {
        pb.finish_with_message("merge complete");
    }

    writer.finalize("merged")?;
    Ok(())
}

/// Parallel driver shared by merge (and mirrored in forge-cli quantize):
/// `work(i)` runs on workers, `emit` runs serially on the caller thread.
fn execute_names_parallel(
    op: &(dyn MergeOp + Sync),
    stores: &[&TensorStore],
    all_names: &[String],
    workers: usize,
    output_dtype: DType,
    pb: Option<&ProgressBar>,
    mut emit: impl FnMut(&str, Vec<usize>, &str, Vec<u8>) -> Result<()>,
) -> Result<()> {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::mpsc::sync_channel;

    const BUDGET: u64 = 4 * 1024 * 1024 * 1024; // 4GB in-flight cap
    let next = AtomicUsize::new(0);
    let in_flight = AtomicU64::new(0);
    // (name, shape, dtype_str, bytes); None payload = skipped tensor.
    let (tx, rx) =
        sync_channel::<Option<(String, Vec<usize>, &'static str, Vec<u8>)>>(workers * 2);

    std::thread::scope(|scope| {
        for _ in 0..workers {
            let next = &next;
            let in_flight = &in_flight;
            let tx = tx.clone();
            scope.spawn(move || {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(name) = all_names.get(i) else { break };
                    let meta = match stores[0].tensor_meta(name).ok() {
                        Some(m) => m,
                        None => {
                            let _ = tx.send(None);
                            continue;
                        }
                    };
                    // Acquire byte budget: (n_models inputs + 1 output) x tensor.
                    let cost = (meta.size as u64).saturating_mul(stores.len() as u64 + 1).max(1);
                    while in_flight.fetch_add(cost, Ordering::SeqCst) + cost > BUDGET {
                        in_flight.fetch_sub(cost, Ordering::SeqCst);
                        std::thread::yield_now();
                    }
                    let msg = (|| {
                        let inputs: Vec<Vec<f32>> = stores
                            .iter()
                            .filter_map(|s| s.tensor_f32(name).ok())
                            .collect();
                        if inputs.is_empty() {
                            return None;
                        }
                        let result = op.merge_tensors(name, &meta, &inputs).ok()?;
                        let (bytes, dtype_str) = dtype_bytes(&result, output_dtype);
                        Some((name.clone(), meta.shape.clone(), dtype_str, bytes))
                    })();
                    in_flight.fetch_sub(cost, Ordering::SeqCst);
                    if tx.send(msg).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);

        let mut done = 0usize;
        for msg in rx {
            if let Some((name, shape, dtype_str, bytes)) = msg {
                emit(&name, shape, dtype_str, bytes)?;
            }
            done += 1;
            if let Some(pb) = pb {
                pb.inc(1);
            }
            if done >= all_names.len() {
                break;
            }
        }
        Ok::<(), anyhow::Error>(())
    })?;
    Ok(())
}

/// F32->output-dtype byte conversion (M5: F32 arm is one memcpy).
fn dtype_bytes(result: &[f32], output_dtype: DType) -> (Vec<u8>, &'static str) {
    match output_dtype {
        DType::F16 => {
            let mut buf = Vec::with_capacity(result.len() * 2);
            for &val in result {
                buf.extend_from_slice(&half::f16::from_f32(val).to_bits().to_le_bytes());
            }
            (buf, "F16")
        }
        DType::BF16 => {
            let mut buf = Vec::with_capacity(result.len() * 2);
            for &val in result {
                buf.extend_from_slice(&half::bf16::from_f32(val).to_bits().to_le_bytes());
            }
            (buf, "BF16")
        }
        DType::F32 => {
            #[cfg(target_endian = "little")]
            {
                let mut buf = Vec::with_capacity(result.len() * 4);
                unsafe {
                    buf.set_len(result.len() * 4);
                    std::ptr::copy_nonoverlapping(
                        result.as_ptr() as *const u8,
                        buf.as_mut_ptr(),
                        result.len() * 4,
                    );
                }
                (buf, "F32")
            }
            #[cfg(not(target_endian = "little"))]
            {
                let mut buf = Vec::with_capacity(result.len() * 4);
                for &val in result {
                    buf.extend_from_slice(&val.to_le_bytes());
                }
                (buf, "F32")
            }
        }
        _ => {
            let mut buf = Vec::with_capacity(result.len() * 2);
            for &val in result {
                buf.extend_from_slice(&half::f16::from_f32(val).to_bits().to_le_bytes());
            }
            (buf, "F16")
        }
    }
}
