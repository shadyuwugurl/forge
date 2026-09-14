use anyhow::Result;
use forge_core::TensorMeta;
use crate::orchestrator::MergeOp;
use crate::slerp_utils::normalize_weights;

/// Model Stock: geometric weight search for linear interpolation.
/// Computes pairwise cosine similarities between task vectors and assigns
/// each model weight ∝ its mean similarity to the others (models near the
/// "center" of the group get up-weighted), then returns the weighted average.
/// Requires ≥3 models to be meaningful; falls back to uniform linear merge.
pub struct ModelStockMerge {
    pub base_index: Option<usize>,
    pub weights: Vec<f32>,
}

impl ModelStockMerge {
    pub fn new() -> Self {
        Self { base_index: None, weights: vec![] }
    }
}

impl Default for ModelStockMerge {
    fn default() -> Self {
        Self::new()
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    dot / (na * nb)
}

impl MergeOp for ModelStockMerge {
    fn merge_tensor(&self, _name: &str, meta: &TensorMeta) -> Result<Vec<f32>> {
        Ok(vec![0.0f32; meta.num_elements()])
    }

    fn merge_tensors(&self, _name: &str, meta: &TensorMeta, inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
        if inputs.is_empty() {
            anyhow::bail!("model_stock: no input tensors");
        }
        let n = meta.num_elements();
        for (i, t) in inputs.iter().enumerate() {
            if t.len() != n {
                anyhow::bail!("model_stock: input {} size mismatch", i);
            }
        }
        // Explicit weights win; otherwise geometric (mean-similarity) weights
        let w = if self.weights.len() == inputs.len() {
            normalize_weights(&self.weights, inputs.len())
        } else if inputs.len() >= 3 {
            let mut sims = vec![0.0f32; inputs.len()];
            for i in 0..inputs.len() {
                let mut s = 0.0f32;
                for j in 0..inputs.len() {
                    if i != j {
                        s += cosine(&inputs[i], &inputs[j]);
                    }
                }
                sims[i] = (s / (inputs.len() - 1) as f32).max(0.0);
            }
            normalize_weights(&sims, inputs.len())
        } else {
            normalize_weights(&[], inputs.len())
        };
        let mut result = vec![0.0f32; n];
        for (t, x) in inputs.iter().zip(w.iter()) {
            for (r, v) in result.iter_mut().zip(t.iter()) {
                *r += x * v;
            }
        }
        Ok(result)
    }
}
