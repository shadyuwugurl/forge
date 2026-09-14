use anyhow::Result;
use forge_core::TensorMeta;
use crate::orchestrator::MergeOp;

/// NearSwap: interpolate only where parameters are similar.
/// Per-parameter: if relative distance |a−b| / max(|a|,|b|,ε) ≤ threshold,
/// blend with weight t; otherwise keep model A's value. Preserves
/// distinctive features while merging consensus regions.
pub struct NearSwapMerge {
    /// Blend weight for similar parameters
    pub t: f32,
    /// Max relative distance to count as "near"
    pub threshold: f32,
}

impl NearSwapMerge {
    pub fn new(t: f32, threshold: f32) -> Self {
        Self { t, threshold }
    }
}

impl MergeOp for NearSwapMerge {
    fn merge_tensor(&self, _name: &str, meta: &TensorMeta) -> Result<Vec<f32>> {
        Ok(vec![0.0f32; meta.num_elements()])
    }

    fn merge_tensors(&self, _name: &str, meta: &TensorMeta, inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
        if inputs.len() < 2 {
            anyhow::bail!("nearswap: need 2 models");
        }
        let (a, b) = (&inputs[0], &inputs[1]);
        if a.len() != meta.num_elements() || b.len() != a.len() {
            anyhow::bail!("nearswap: tensor size mismatch");
        }
        Ok(a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let denom = x.abs().max(y.abs()).max(1e-12);
                if (x - y).abs() / denom <= self.threshold {
                    (1.0 - self.t) * x + self.t * y
                } else {
                    *x
                }
            })
            .collect())
    }
}
