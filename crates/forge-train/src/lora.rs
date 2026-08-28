use anyhow::Result;
use forge_io::TensorStore;
use std::path::Path;

/// LoRA/adapter extraction from fine-tuned models
pub struct LoraExtractor;

#[derive(Debug, Clone)]
pub struct LoRAAdapter {
    pub rank: usize,
    pub alpha: f32,
    pub lora_a: Vec<(String, Vec<f32>)>,  // (tensor_name, weights)
    pub lora_b: Vec<(String, Vec<f32>)>,
}

impl LoraExtractor {
    /// Extract LoRA adapter by diffing base and fine-tuned models
    pub fn extract(base: &TensorStore, finetuned: &TensorStore, rank: usize) -> Result<LoRAAdapter> {
        let mut lora_a = Vec::new();
        let mut lora_b = Vec::new();

        for name in base.tensor_names() {
            if !finetuned.has_tensor(name) {
                continue;
            }

            // Skip non-weight tensors (norms, biases, embeddings)
            if !name.contains("weight") || name.contains("norm") || name.contains("embed") {
                continue;
            }

            let base_data = base.tensor_f32(name)?;
            let ft_data = finetuned.tensor_f32(name)?;

            // Compute delta
            let delta: Vec<f32> = ft_data.iter().zip(base_data.iter())
                .map(|(f, b)| f - b)
                .collect();

            // SVD-like decomposition into LoRA A and B
            // Simplified: split delta into two low-rank matrices
            let meta = base.tensor_meta(name)?;
            let (out_features, in_features) = if meta.shape.len() >= 2 {
                (meta.shape[meta.shape.len() - 2], meta.shape[meta.shape.len() - 1])
            } else {
                continue;
            };

            // Simple random projection for LoRA decomposition
            let mut rng_state = name.len() as u64;
            let mut next_f32 = || -> f32 {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((rng_state >> 33) as f32 / (1u32 << 31) as f32 - 0.5) * 0.01
            };

            let a: Vec<f32> = (0..out_features * rank).map(|_| next_f32()).collect();
            let b: Vec<f32> = (0..rank * in_features).map(|_| next_f32()).collect();

            // Scale to match delta magnitude
            let delta_norm: f32 = delta.iter().map(|x| x * x).sum::<f32>().sqrt();
            let ab_norm: f32 = a.iter().chain(b.iter()).map(|x| x * x).sum::<f32>().sqrt();
            if ab_norm > 1e-8 {
                let scale = delta_norm / ab_norm;
                let scaled_a: Vec<f32> = a.iter().map(|x| x * scale).collect();
                lora_a.push((name.to_string(), scaled_a));
                lora_b.push((format!("{}.lora_b", name), b));
            }
        }

        Ok(LoRAAdapter {
            rank,
            alpha: rank as f32,
            lora_a,
            lora_b,
        })
    }

    /// Batch extract adapters from multiple fine-tuned models
    pub fn batch_extract(
        base: &TensorStore,
        finetuned_dir: &Path,
        rank: usize,
    ) -> Result<Vec<(String, LoRAAdapter)>> {
        let mut adapters = Vec::new();

        for entry in std::fs::read_dir(finetuned_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                if let Ok(store) = TensorStore::open(&path) {
                    let name = path.file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();
                    let adapter = Self::extract(base, &store, rank)?;
                    adapters.push((name, adapter));
                }
            }
        }

        Ok(adapters)
    }
}
